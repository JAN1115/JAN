import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np

# --- 헬퍼 클래스 및 함수 정의 (이전과 동일) ---

def _calculate_longest_streaks(history):
    if not history:
        return 0, 0
    max_p_streak, max_b_streak = 0, 0
    current_p_streak, current_b_streak = 0, 0
    for result in history:
        if result == 'P':
            current_p_streak += 1
            current_b_streak = 0
            max_p_streak = max(max_p_streak, current_p_streak)
        elif result == 'B':
            current_b_streak += 1
            current_p_streak = 0
            max_b_streak = max(max_b_streak, current_b_streak)
    return max_p_streak, max_b_streak

def create_features(game_history, shoe_history):
    features = {}
    pb_history = [r for r in game_history if r in 'PB']
    if len(pb_history) < 13:
        return None

    # --- 단기 특징 ---
    window = pb_history[-10:]
    features['recent_p_ratio'] = window.count('P') / len(window) if len(window) > 0 else 0.5
    features['volatility'] = sum(1 for i in range(1, len(window)) if window[i] != window[i-1]) / (len(window)-1) if len(window) > 1 else 0
    grams2, grams3 = ''.join(pb_history[-2:]), ''.join(pb_history[-3:])
    for g in ['PP','PB','BP','BB']: features[f'2_gram_{g}'] = int(grams2 == g)
    for g in ['PPP','PPB','PBP','PBB','BPP','BPB','BBP','BBB']: features[f'3_gram_{g}'] = int(grams3 == g)
    last_4, last_6 = ''.join(pb_history[-4:]), ''.join(pb_history[-6:])
    features['is_3_1_pattern'] = 1 if last_4 in ['PPPB', 'BBBP'] else 0
    features['is_3_3_pattern'] = 1 if last_6 in ['PPPBBB', 'BBBPPP'] else 0
    
    # --- 슈(Shoe) 전체 장기 특징 ---
    shoe_pb_history = [r for r in shoe_history if r in 'PB']
    if len(shoe_pb_history) > 0:
        features['shoe_p_ratio'] = shoe_pb_history.count('P') / len(shoe_pb_history)
        features['shoe_b_ratio'] = shoe_pb_history.count('B') / len(shoe_pb_history)
        longest_p, longest_b = _calculate_longest_streaks(shoe_pb_history)
        features['longest_p_streak_in_shoe'] = longest_p
        features['longest_b_streak_in_shoe'] = longest_b
        features['game_position_in_shoe'] = len(shoe_pb_history) / 80.0
    else:
        features['shoe_p_ratio'] = 0.5
        features['shoe_b_ratio'] = 0.5
        features['longest_p_streak_in_shoe'] = 0
        features['longest_b_streak_in_shoe'] = 0
        features['game_position_in_shoe'] = 0.0
        
    return features

class WeightedMajorityCombiner:
    def __init__(self, strategies, initial_beta=0.95):
        self.beta = initial_beta
        self.weights = {s: 1.0 for s in strategies}
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

# ---------- 통합 AI 클래스 (오류 수정 버전) ----------
class MLConcordanceAI:
    def __init__(self, lgbm_model_path='baccarat_lgbm_model.joblib',
                 chaos_model_path='baccarat_chaos_model.joblib'):
        self.lgbm_model = joblib.load(lgbm_model_path)
        self.chaos_model = joblib.load(chaos_model_path)
        self.online_model = SGDClassifier(loss='log_loss', random_state=42)
        self.combiner = WeightedMajorityCombiner(strategies=['LGBM', 'Online'], initial_beta=0.95)
        
        # [# 수정된 부분] 기존 모델이 학습한 특징들의 이름 목록을 정의
        self.original_feature_names = [
            'recent_p_ratio', 'volatility', '2_gram_PP', '2_gram_PB', '2_gram_BP', '2_gram_BB',
            '3_gram_PPP', '3_gram_PPB', '3_gram_PBP', '3_gram_PBB', '3_gram_BPP', 
            '3_gram_BPB', '3_gram_BBP', '3_gram_BBB', 'is_3_1_pattern', 'is_3_3_pattern'
        ]

        self.chaos_threshold = 0.8
        self.shoe_history = []
        self.reset_game()
        self.predict_next()

    def reset_game(self):
        self.history, self.game_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.max_win, self.max_loss, self.current_win, self.current_loss = 0, 0, 0, 0, 0, 0
        self.analysis_text, self.next_prediction, self.should_bet_now = "AI 초기화 중...", None, False
        self.online_model_ready, self.last_features = False, None
        self.last_preds = {}
        self.mode_performance_history = []
        self.last_mode_for_bet = "안정"
        self.shoe_history = []
        print("--- 새로운 게임(슈)을 시작합니다. 모든 기록이 초기화되었습니다. ---")

    def _calculate_chaos_index(self):
        recent_bet_results = [h for h in self.hit_record[-15:] if h is not None]
        if len(recent_bet_results) < 10: return 0.0
        numeric_results = [1 if r == 'O' else 0 for r in recent_bet_results]
        return min(np.std(numeric_results) * 2.0, 1.0)

    def _update_dynamic_parameters(self, chaos_index):
        if self.current_loss >= 2: self.chaos_threshold = min(0.9, self.chaos_threshold + 0.05)
        elif self.current_win >= 2: self.chaos_threshold = max(0.7, self.chaos_threshold - 0.05)
        self.combiner.beta = 0.90 if chaos_index > 0.7 else 0.95

    def handle_input(self, result):
        if result == 'T': 
            self.history.append('T'); self.hit_record.append(None); self.shoe_history.append('T')
        else:
            if self.should_bet_now:
                hit = (self.next_prediction == result)
                self.bet_count += 1
                self.mode_performance_history.append(f"{self.last_mode_for_bet}-{'O' if hit else 'X'}")
                if self.last_mode_for_bet == "안정": self.combiner.update(self.last_preds, result, hit)
                
                # [# 중요] 온라인 모델은 모든 특징(21개)을 사용하여 실시간으로 학습
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
                
            self.history.append(result); self.game_history.append(result); self.shoe_history.append(result)
        
        self.predict_next()

    def predict_next(self):
        pb_history = [r for r in self.game_history if r in 'PB']
        if len(pb_history) < 13:
            self.analysis_text, self.should_bet_now = "AI가 패턴을 학습하고 있습니다...", False
            return
        
        chaos_index = self._calculate_chaos_index()
        self._update_dynamic_parameters(chaos_index)
        
        # 1. [# 수정된 부분] 모든 특징(21개)을 생성하여 `online_model`을 위해 저장
        all_features = create_features(self.game_history, self.shoe_history)
        self.last_features = all_features
        if not all_features: return

        # 2. [# 핵심 수정] 미리 학습된 모델을 위해서는 원래의 16개 특징만 필터링
        features_for_pretrained = {k: all_features[k] for k in self.original_feature_names}
        df_for_pretrained = pd.DataFrame([features_for_pretrained])
        
        mode = "혼돈" if chaos_index >= self.chaos_threshold else "안정"
        
        strategist_override = False
        if len(self.mode_performance_history) >= 3:
            last_3 = self.mode_performance_history[-3:]
            if last_3 == ['안정-X']*3 and mode == "안정": mode = "혼돈"; strategist_override = True
            elif last_3 == ['혼돈-X']*3 and mode == "혼돈": mode = "안정"; strategist_override = True

        final_prediction = None
        if mode == "혼돈":
            # 미리 학습된 chaos_model은 16개 특징만 사용
            final_prediction = self.chaos_model.predict(df_for_pretrained)[0]
        elif mode == "안정":
            preds = {}
            # 미리 학습된 lgbm_model은 16개 특징만 사용
            preds['LGBM'] = self.lgbm_model.predict(df_for_pretrained)[0]
            if self.online_model_ready:
                # 온라인 모델은 모든 특징(21개)을 사용하여 예측
                preds['Online'] = self.online_model.predict(pd.DataFrame([all_features]))[0]
            if preds: final_prediction = self.combiner.predict(preds)
            self.last_preds = preds
        
        self.next_prediction = final_prediction
        self.should_bet_now = (final_prediction is not None)
        self.last_mode_for_bet = mode
        
        override_note = " (전략 AI 개입)" if strategist_override else ""
        self.analysis_text = (f"AI 분석: [국면: {mode}{override_note} / "
                              f"혼돈지수: {chaos_index:.0%} / 임계값: {self.chaos_threshold:.0%}] ➡️ {self.next_prediction}")

    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return {"총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
                "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}
