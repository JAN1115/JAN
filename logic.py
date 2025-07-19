import os
import random
import time
from collections import Counter, defaultdict, deque
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

# --- 패턴 감지기 클래스 ---
class PatternDetector:
    def __init__(self, window_size=20, trend_threshold=0.6, chop_threshold=0.6):
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.chop_threshold = chop_threshold

    def detect(self, history):
        pb_history = [r for r in history if r in 'PB']
        if len(pb_history) < self.window_size:
            return 'NEUTRAL'
        window = pb_history[-self.window_size:]
        trend_count = sum(1 for i in range(1, len(window)) if window[i] == window[i-1])
        total_transitions = len(window) - 1
        trend_ratio = trend_count / total_transitions
        chop_ratio = 1 - trend_ratio
        if trend_ratio >= self.trend_threshold:
            return 'TRENDING'
        elif chop_ratio >= self.chop_threshold:
            return 'CHOPPY'
        else:
            return 'NEUTRAL'

# --- 특징(Feature) 추출 함수 ---
def create_features(history):
    features = {}
    pb_history = [r for r in history if r in 'PB']
    if len(pb_history) < 6:
        return None
    window = pb_history[-10:]
    features['recent_p_ratio'] = window.count('P') / len(window)
    features['volatility'] = sum(1 for i in range(1, len(window)) if window[i] != window[i-1]) / (len(window)-1)
    grams2 = ''.join(pb_history[-2:])
    grams3 = ''.join(pb_history[-3:])
    for g in ['PP','PB','BP','BB']:
        features[f'2_gram_{g}'] = int(grams2 == g)
    for g in ['PPP','PPB','PBP','PBB','BPP','BPB','BBP','BBB']:
        features[f'3_gram_{g}'] = int(grams3 == g)
    last4 = ''.join(pb_history[-4:])
    last6 = ''.join(pb_history[-6:])
    features['is_3_1_pattern'] = int(last4 in ['PPPB','BBBP'])
    features['is_3_3_pattern'] = int(last6 in ['PPPBBB','BBBPPP'])
    return features

# --- 지수평활 ---
class ExpSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ema = None
    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

# --- 컨셉 드리프트 감지 ---
class DriftDetector:
    def __init__(self, windows=(10,50), threshold=0.2):
        self.deques = [deque(maxlen=w) for w in windows]
        self.threshold = threshold
    def add(self, value):
        drifted = False
        for dq in self.deques:
            dq.append(value)
            if len(dq) == dq.maxlen:
                mid = dq.maxlen//2
                if abs(np.mean(list(dq)[:mid]) - np.mean(list(dq)[mid:])) > self.threshold:
                    drifted = True
        return drifted

# --- 중국점 헬퍼 ---
def calc_big_road(history):
    road, prev = [], None
    for r in [h for h in history if h in 'PB']:
        if prev is None or r != prev:
            road.append([r])
        else:
            road[-1].append(r)
        prev = r
    return road

def calc_derived_road(big_road, offset):
    derived = []
    for i in range(offset, len(big_road)):
        derived.append('P' if len(big_road[i]) == len(big_road[i-offset]) else 'B')
    return derived

# --- 가중치 앙상블 ---
class WeightedMajorityCombiner:
    def __init__(self, beta=0.95, min_beta=0.5, decay=0.99):
        self.beta = beta
        self.min_beta = min_beta
        self.decay = decay
        self.weights = {s:1.0 for s in ['ML','LGBM','Rule','RL','EMA']}
        self.loss_count = 0
    def update(self, preds, actual, hit):
        if not hit:
            self.loss_count += 1
            if self.loss_count % 3 == 0:
                self.beta = max(self.beta * self.decay, self.min_beta)
        else:
            self.loss_count = 0
        for strat, p in preds.items():
            if p in ('P','B') and p != actual:
                self.weights[strat] *= self.beta
        total = sum(self.weights.values())
        if total > 1e-9:
            for strat in self.weights:
                self.weights[strat] /= total

# ---------- 통합 AI 클래스 ----------
class MLConcordanceAI:
    def __init__(self, model_path='baccarat_model.joblib',
                 lgbm_model_path='baccarat_lgbm_model.joblib'):
        self.history = []
        self.game_history = []
        self.hit_record = []
        self.bet_count = self.correct = self.incorrect = 0
        self.current_win = self.max_win = 0
        self.current_loss = self.max_loss = 0
        self.last_bet_result = None
        self.analysis_text = "AI: 초기 데이터 수집 중..."
        self.next_prediction = None
        self.should_bet_now = False
        self.rl_mode = False
        self.q_table = defaultdict(lambda:{'P':0.0,'B':0.0})
        self.consecutive_miss = 0

        self.model = joblib.load(model_path)
        self.lgbm_model = joblib.load(lgbm_model_path)
        
        self.exp_smoother = ExpSmoother(alpha=0.2)
        self.drift = DriftDetector(windows=(10,50), threshold=0.2)
        self.combiner = WeightedMajorityCombiner(beta=0.95, min_beta=0.5, decay=0.99)
        self.pattern_detector = PatternDetector()
        
        self.last_preds = {s:None for s in self.combiner.weights.keys()}
        self.last_strategy = None
        self.predict_next()

    def rule_predict(self):
        ph = [h for h in self.game_history if h in 'PB']
        votes = []
        if len(ph)>=2 and ph[-1]==ph[-2]:
            votes.append(ph[-1])
        br = calc_big_road(self.game_history)
        for off in (1,2,3):
            dr = calc_derived_road(br, off)
            if dr:
                votes.append(dr[-1])
        return Counter(votes).most_common(1)[0][0] if votes else None
        
    def predict_next(self):
        ph = [h for h in self.game_history if h in 'PB']
        if len(ph) < 6:
            self.analysis_text = "AI: 초기 데이터 수집 중..."
            self.should_bet_now = False
            return

        preds = {}
        ema_val = self.exp_smoother.update(1 if ph[-1]=='P' else 0)
        preds['EMA'] = 'P' if ema_val > 0.5 else 'B'
        
        feats = create_features(self.game_history)
        if feats:
            df_for_predict = pd.DataFrame([feats])
            preds['ML'] = self.model.predict(df_for_predict)[0]
            preds['LGBM'] = self.lgbm_model.predict(df_for_predict)[0]
        else:
            preds['ML'], preds['LGBM'] = None, None
        
        preds['Rule'] = self.rule_predict()
        if self.current_loss >= 3: self.rl_mode = True
        if self.rl_mode and feats:
            state_tuple = (ph[-2], ph[-1], round(feats['recent_p_ratio'], 1), round(feats['volatility'], 1))
            q_values = self.q_table[state_tuple]
            preds['RL'] = max(q_values, key=q_values.get) if q_values else random.choice(['P','B'])
        else: preds['RL'] = None

        game_phase = self.pattern_detector.detect(self.game_history)
        dynamic_weights = self.combiner.weights.copy()
        boost_factor = 1.5
        
        if game_phase == 'TRENDING':
            dynamic_weights['EMA'] *= boost_factor
        elif game_phase == 'CHOPPY':
            dynamic_weights['LGBM'] *= boost_factor
        
        votes = {'P': 0.0, 'B': 0.0}
        for strat, p in preds.items():
            if p in votes:
                votes[p] += dynamic_weights.get(strat, 1.0)
        
        if votes['P'] == votes['B']:
            final_prediction = preds.get('LGBM', 'B')
        else:
            final_prediction = 'P' if votes['P'] > votes['B'] else 'B'
        
        # --- 신뢰도 계산 및 베팅 결정 (새로 추가된 핵심 로직) ---
        total_votes = sum(votes.values())
        confidence = 0.0
        if total_votes > 0:
            confidence = abs(votes['P'] - votes['B']) / total_votes
        
        # 신뢰도 임계값 설정 (0.5는 득표율 75% 이상을 의미)
        CONFIDENCE_THRESHOLD = 0.5 
        
        # 신뢰도가 임계값 이상일 때만 베팅을 결정
        self.should_bet_now = (final_prediction is not None) and (confidence >= CONFIDENCE_THRESHOLD)
        # --- 로직 종료 ---
        
        hit_val = 1 if self.last_bet_result else 0
        if self.drift.add(hit_val) and self.consecutive_miss >= 4:
            self.rl_mode = False; self.q_table.clear()

        winning_strats = {s: dynamic_weights.get(s, 0) for s, p in preds.items() if p == final_prediction}
        strategy = max(winning_strats, key=winning_strats.get) if winning_strats else None
        
        # 분석 메시지에 신뢰도와 베팅 여부 표시
        bet_or_skip = "베팅" if self.should_bet_now else "스킵"
        self.analysis_text = f"AI 예측: [Phase: {game_phase}] [Conf: {confidence:.0%}] ➡️ {final_prediction} ({bet_or_skip})"
        
        self.last_preds = preds
        self.next_prediction = final_prediction
        self.last_strategy = strategy

    def handle_input(self, result):
        prev_game_history = [h for h in self.game_history if h in 'PB']
        if result == 'T':
            self.history.append(result); self.hit_record.append(None); self.analysis_text = "AI: TIE, 베팅 스킵."
        else:
            if self.should_bet_now:
                hit = (self.next_prediction == result); self.bet_count += 1
                if hit:
                    self.correct += 1; self.current_win += 1; self.current_loss = 0; self.consecutive_miss = 0
                    self.max_win = max(self.max_win, self.current_win)
                else:
                    self.incorrect += 1; self.current_win = 0; self.current_loss += 1; self.consecutive_miss += 1
                    self.max_loss = max(self.max_loss, self.current_loss)
                self.hit_record.append('O' if hit else 'X')
                self.combiner.update(self.last_preds, result, hit)
                update_feats = create_features(prev_game_history)
                if self.last_strategy == 'RL' and update_feats:
                    state_tuple = (prev_game_history[-2], prev_game_history[-1], round(update_feats['recent_p_ratio'], 1), round(update_feats['volatility'], 1))
                    reward = 1 if hit else -1; q = self.q_table[state_tuple]
                    next_q_val = max(q.values()) if q else 0
                    q[self.next_prediction] = q.get(self.next_prediction,0) + 0.1 * (reward + 0.9 * next_q_val - q.get(self.next_prediction,0))
                
                self.last_bet_result = hit
            else:
                self.hit_record.append(None)
            self.history.append(result)
            self.game_history.append(result)
        self.predict_next()
        
    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return {
            "총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
            "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss
        }
