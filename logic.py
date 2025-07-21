import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np

# --- 헬퍼 클래스 및 함수 정의 (이전과 동일) ---
def create_features(history):
    features = {}
    pb_history = [r for r in history if r in 'PB']
    if len(pb_history) < 13: return None
    window = pb_history[-10:]
    features['recent_p_ratio'] = window.count('P') / len(window)
    features['volatility'] = sum(1 for i in range(1, len(window)) if window[i] != window[i-1]) / (len(window)-1)
    grams2, grams3 = ''.join(pb_history[-2:]), ''.join(pb_history[-3:])
    for g in ['PP','PB','BP','BB']: features[f'2_gram_{g}'] = int(grams2 == g)
    for g in ['PPP','PPB','PBP','PBB','BPP','BPB','BBP','BBB']: features[f'3_gram_{g}'] = int(grams3 == g)
    last_4, last_6 = ''.join(pb_history[-4:]), ''.join(pb_history[-6:])
    features['is_3_1_pattern'] = 1 if last_4 in ['PPPB', 'BBBP'] else 0
    features['is_3_3_pattern'] = 1 if last_6 in ['PPPBBB', 'BBBPPP'] else 0
    return features

class WeightedMajorityCombiner:
    def __init__(self, strategies, beta=0.95):
        self.beta, self.weights = beta, {s: 1.0 for s in strategies}
    def predict(self, preds):
        votes = {'P': 0.0, 'B': 0.0}
        for strat, p in preds.items():
            if p in votes: votes[p] += self.weights.get(strat, 1.0)
        if votes['P'] == votes['B']: return preds.get('LGBM')
        return 'P' if votes['P'] > votes['B'] else 'B'
    def update(self, preds, actual, hit):
        for strat, p in preds.items():
            if p in ('P', 'B') and p != actual: self.weights[strat] *= self.beta
        total = sum(self.weights.values())
        if total > 1e-9:
            for strat in self.weights: self.weights[strat] /= total

# ---------- 통합 AI 클래스 (최종 진화 버전) ----------
class MLConcordanceAI:
    def __init__(self, lgbm_model_path='baccarat_lgbm_model.joblib',
                 chaos_model_path='baccarat_chaos_model.joblib'):
        self.history, self.game_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.max_win, self.max_loss, self.current_win, self.current_loss = 0, 0, 0, 0, 0, 0
        self.analysis_text, self.next_prediction, self.should_bet_now = "AI 초기화 중...", None, False
        
        self.lgbm_model = joblib.load(lgbm_model_path)
        self.chaos_model = joblib.load(chaos_model_path)
        self.online_model = SGDClassifier(loss='log_loss', random_state=42)
        self.online_model_ready, self.last_features = False, None
        self.combiner = WeightedMajorityCombiner(strategies=['LGBM', 'Online'])
        self.last_preds = {}

        # --- [신규] 전략 AI를 위한 기록부 ---
        self.mode_performance_history = []
        self.last_mode_for_bet = "안정"
        # --------------------------------

        self.predict_next()

    def _calculate_chaos_index(self):
        recent_bet_results = [h for h in self.hit_record[-15:] if h is not None]
        if len(recent_bet_results) < 10: return 0.0
        numeric_results = [1 if r == 'O' else 0 for r in recent_bet_results]
        chaos = np.std(numeric_results) * 2
        return chaos

    def handle_input(self, result):
        if result == 'T': self.history.append('T'); self.hit_record.append(None)
        else:
            if self.should_bet_now:
                hit = (self.next_prediction == result)
                self.bet_count += 1
                
                # [신규] 전략 AI가 학습할 수 있도록, 어떤 모드의 예측이었고 결과가 어땠는지 기록
                self.mode_performance_history.append(f"{self.last_mode_for_bet}-{'O' if hit else 'X'}")

                if self.last_mode_for_bet == "안정":
                    self.combiner.update(self.last_preds, result, hit)
                
                if self.last_features:
                    self.online_model.partial_fit(pd.DataFrame([self.last_features]), [result], classes=['P','B'])
                    self.online_model_ready = True

                if hit: self.correct += 1; self.current_win += 1; self.current_loss = 0
                else: self.current_win = 0; self.current_loss += 1
                self.max_win = max(self.max_win, self.current_win)
                self.max_loss = max(self.max_loss, self.current_loss)
                self.hit_record.append('O' if hit else 'X')
            else:
                self.hit_record.append(None)
            self.history.append(result); self.game_history.append(result)
        self.predict_next()

    def predict_next(self):
        pb_history = [r for r in self.game_history if r in 'PB']
        if len(pb_history) < 13:
            self.analysis_text, self.should_bet_now = "AI가 패턴을 학습하고 있습니다...", False
            return
        
        chaos_index = self._calculate_chaos_index()
        final_prediction, mode = None, "안정"
        features = create_features(self.game_history)
        self.last_features = features
        
        # 1. 혼돈 지수에 따른 1차 모드 결정
        CHAOS_THRESHOLD = 0.8
        if chaos_index >= CHAOS_THRESHOLD:
            mode = "혼돈"
        
        # --- [신규] 전략 AI의 최종 판단 ---
        strategist_override = False
        if len(self.mode_performance_history) >= 3:
            # 최근 3번의 모드 성과를 분석
            last_3_modes = self.mode_performance_history[-3:]
            # 만약 안정 모드가 3연패 중이라면, 강제로 혼돈 모드로 변경
            if last_3_modes == ['안정-X', '안정-X', '안정-X'] and mode == "안정":
                mode = "혼돈"
                strategist_override = True
            # 만약 혼돈 모드가 3연패 중이라면, 강제로 안정 모드로 변경
            elif last_3_modes == ['혼돈-X', '혼돈-X', '혼돈-X'] and mode == "혼돈":
                mode = "안정"
                strategist_override = True
        # --------------------------------

        # 2. 최종 결정된 모드에 따라 예측 수행
        if mode == "혼돈" and features:
            final_prediction = self.chaos_model.predict(pd.DataFrame([features]))[0]
        elif mode == "안정" and features:
            preds = {}
            preds['LGBM'] = self.lgbm_model.predict(pd.DataFrame([features]))[0]
            if self.online_model_ready:
                preds['Online'] = self.online_model.predict(pd.DataFrame([features]))[0]
            if preds: final_prediction = self.combiner.predict(preds)
            self.last_preds = preds
        
        self.next_prediction = final_prediction
        self.should_bet_now = (final_prediction is not None)
        self.last_mode_for_bet = mode # 베팅 시점의 모드를 기록
        
        override_note = " (전략 AI 개입)" if strategist_override else ""
        self.analysis_text = f"AI 분석: [국면: {mode}{override_note} / 혼돈지수: {chaos_index:.0%}] ➡️ {self.next_prediction}"

    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return {"총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
                "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}
