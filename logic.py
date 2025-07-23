import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np

# --- 헬퍼 클래스 및 함수 ---

def _calculate_longest_streaks(history):
    if not history: return 0, 0
    max_p_streak, max_b_streak, current_p_streak, current_b_streak = 0, 0, 0, 0
    for result in history:
        if result == 'P':
            current_p_streak += 1; current_b_streak = 0
            max_p_streak = max(max_p_streak, current_p_streak)
        elif result == 'B':
            current_b_streak += 1; current_p_streak = 0
            max_b_streak = max(max_b_streak, current_b_streak)
    return max_p_streak, max_b_streak

def create_features(game_history, shoe_history):
    features = {}
    pb_history = [r for r in game_history if r in 'PB']
    if len(pb_history) < 13: return None
    
    # 단기 특징
    window10 = pb_history[-10:]
    features['recent_p_ratio'] = window10.count('P') / len(window10) if len(window10) > 0 else 0.5
    features['volatility'] = sum(1 for i in range(1, len(window10)) if window10[i] != window10[i-1]) / (len(window10)-1) if len(window10) > 1 else 0
    grams2, grams3 = ''.join(pb_history[-2:]), ''.join(pb_history[-3:])
    for g in ['PP','PB','BP','BB']: features[f'2_gram_{g}'] = int(grams2 == g)
    for g in ['PPP','PPB','PBP','PBB','BPP','BPB','BBP','BBB']: features[f'3_gram_{g}'] = int(grams3 == g)
    last_4, last_6 = ''.join(pb_history[-4:]), ''.join(pb_history[-6:])
    features['is_3_1_pattern'] = 1 if last_4 in ['PPPB', 'BBBP'] else 0
    features['is_3_3_pattern'] = 1 if last_6 in ['PPPBBB', 'BBBPPP'] else 0
    
    # 장기 특징
    shoe_pb_history = [r for r in shoe_history if r in 'PB']
    if len(shoe_pb_history) > 0:
        features['shoe_p_ratio'] = shoe_pb_history.count('P') / len(shoe_pb_history)
        features['shoe_b_ratio'] = shoe_pb_history.count('B') / len(shoe_pb_history)
        longest_p, longest_b = _calculate_longest_streaks(shoe_pb_history)
        features['longest_p_streak_in_shoe'] = longest_p; features['longest_b_streak_in_shoe'] = longest_b
        features['game_position_in_shoe'] = len(shoe_pb_history) / 80.0
    else:
        features.update({'shoe_p_ratio': 0.5, 'shoe_b_ratio': 0.5, 'longest_p_streak_in_shoe': 0, 'longest_b_streak_in_shoe': 0, 'game_position_in_shoe': 0.0})
    return features

class WeightedMajorityCombiner:
    def __init__(self, strategies, initial_beta=0.95):
        self.beta = initial_beta
        self.weights = {s: 1.0 for s in strategies}
    def predict(self, preds):
        votes = {'P': 0.0, 'B': 0.0}
        for strat, p in preds.items():
            if p in votes: votes[p] += self.weights.get(strat, 1.0)
        if votes['P'] == votes['B']:
            for strat in ['LGBM', 'Flow', 'Online_Short', 'Online_Long']:
                if strat in preds: return preds[strat]
            return None
        return 'P' if votes['P'] > votes['B'] else 'B'
    def update(self, preds, actual):
        for strat, p in preds.items():
            if p in ('P', 'B') and p != actual: self.weights[strat] *= self.beta
        total = sum(self.weights.values())
        if total > 1e-9:
            for strat in self.weights: self.weights[strat] /= total

# [강화] 메타 분석 (슈 유형 진단)
def profile_the_shoe(shoe_pb_history):
    total_games = len(shoe_pb_history)
    if total_games < 30: return "데이터 축적 중"
    chop_index = sum(1 for i in range(1, total_games) if shoe_pb_history[i] != shoe_pb_history[i-1]) / (total_games - 1)
    streaks, current_streak = [], 1
    for i in range(1, total_games):
        if shoe_pb_history[i] == shoe_pb_history[i-1]: current_streak += 1
        else: streaks.append(current_streak); current_streak = 1
    streaks.append(current_streak)
    avg_streak_length = np.mean(streaks) if streaks else 0
    if avg_streak_length >= 3.2 and chop_index < 0.5: return "줄타기형"
    if avg_streak_length < 2.4 and chop_index > 0.6: return "퐁당퐁당형"
    return "혼합/불규칙형"

# [강화] 통계 흐름 분석
def analyze_statistical_flow(shoe_pb_history):
    total_games = len(shoe_pb_history)
    if total_games < 20: return None, "[흐름분석: 데이터 축적 중]"
    shoe_type = profile_the_shoe(shoe_pb_history)
    prediction = None
    if shoe_type == "퐁당퐁당형": prediction = 'B' if shoe_pb_history[-1] == 'P' else 'P'
    elif shoe_type == "줄타기형":
        if shoe_pb_history[-1] == shoe_pb_history[-2]: prediction = shoe_pb_history[-1]
    else:
        gap = shoe_pb_history.count('P') - shoe_pb_history.count('B')
        momentum = shoe_pb_history[-20:].count('P') - shoe_pb_history[-20:].count('B')
        tension_threshold = max(4, total_games // 10)
        if abs(gap) > tension_threshold and np.sign(gap) == np.sign(momentum): prediction = 'P' if gap > 0 else 'B'
        elif abs(gap) > tension_threshold and np.sign(gap) != np.sign(momentum) and abs(momentum) > 2: prediction = 'P' if momentum > 0 else 'B'
        elif abs(gap) <= tension_threshold and abs(momentum) >= 4: prediction = 'P' if momentum > 0 else 'B'
    return prediction, f"[슈 유형: {shoe_type}]"

# ---------- [최종] 통합 AI 클래스 ----------
class MLConcordanceAI:
    def __init__(self, lgbm_model_path='baccarat_lgbm_model.joblib', chaos_model_path='baccarat_chaos_model.joblib'):
        self.lgbm_model = joblib.load(lgbm_model_path)
        self.chaos_model = joblib.load(chaos_model_path)
        # [강화] 온라인 전문가 분리
        self.online_model_short = SGDClassifier(loss='log_loss', random_state=10)
        self.online_model_long = SGDClassifier(loss='log_loss', random_state=20)
        # [강화] AI 위원회 구성
        self.combiner = WeightedMajorityCombiner(strategies=['LGBM', 'Flow', 'Online_Short', 'Online_Long'], initial_beta=0.95)
        
        # [강화] 온라인 모델별 학습 데이터 정의
        self.short_term_feature_names = ['recent_p_ratio', 'volatility', 'is_3_1_pattern', 'is_3_3_pattern'] + [f'2_gram_{g}' for g in ['PP','PB','BP','BB']] + [f'3_gram_{g}' for g in ['PPP','PPB','PBP','PBB','BPP','BPB','BBP','BBB']]
        self.long_term_feature_names = ['shoe_p_ratio', 'shoe_b_ratio', 'longest_p_streak_in_shoe', 'longest_b_streak_in_shoe', 'game_position_in_shoe']
        
        self.chaos_threshold = 0.8; self.shoe_history = []
        self.reset_game()

    def reset_game(self):
        self.history, self.game_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.max_win, self.max_loss, self.current_win, self.current_loss = 0, 0, 0, 0, 0, 0
        self.analysis_text, self.next_prediction, self.should_bet_now = "AI 초기화 중...", None, False
        self.online_short_ready, self.online_long_ready, self.last_features = False, False, None
        self.last_preds = {}; self.mode_performance_history = []; self.last_mode_for_bet = "안정"
        self.shoe_history = []; print("--- 새로운 게임(슈)을 시작합니다. ---")

    def _calculate_chaos_index(self):
        recent_bet_results = [h for h in self.hit_record[-15:] if h is not None]
        if len(recent_bet_results) < 10: return 0.0
        return min(np.std([1 if r == 'O' else 0 for r in recent_bet_results]) * 2.0, 1.0)

    # [해결책] 위기 대응 로직 수정
    def _update_dynamic_parameters(self, chaos_index):
        if self.current_loss >= 2: self.chaos_threshold = max(0.60, self.chaos_threshold - 0.05)
        elif self.current_win >= 3: self.chaos_threshold = min(0.90, self.chaos_threshold + 0.05)
        self.combiner.beta = 0.85 if chaos_index > 0.75 else 0.95

    def handle_input(self, result):
        if result == 'T':
            self.history.append('T'); self.hit_record.append(None); self.shoe_history.append('T')
        else:
            if self.should_bet_now:
                hit = (self.next_prediction == result)
                self.bet_count += 1
                self.mode_performance_history.append(f"{self.last_mode_for_bet}-{'O' if hit else 'X'}")
                if self.last_mode_for_bet == "안정": self.combiner.update(self.last_preds, result)
                
                # [강화] 분리된 온라인 전문가들을 각각 학습
                if self.last_features:
                    features_short = pd.DataFrame([{k: self.last_features[k] for k in self.short_term_feature_names}])
                    features_long = pd.DataFrame([{k: self.last_features[k] for k in self.long_term_feature_names}])
                    self.online_model_short.partial_fit(features_short, [result], classes=['P','B'])
                    self.online_model_long.partial_fit(features_long, [result], classes=['P','B'])
                    self.online_short_ready = True; self.online_long_ready = True

                if hit: self.correct += 1; self.current_win += 1; self.current_loss = 0
                else: self.current_win = 0; self.current_loss += 1
                self.max_win = max(self.max_win, self.current_win); self.max_loss = max(self.max_loss, self.current_loss)
                self.hit_record.append('O' if hit else 'X')
            else:
                self.hit_record.append(None)
            self.history.append(result); self.game_history.append(result); self.shoe_history.append(result)
        self.predict_next()

    def predict_next(self):
        if len([r for r in self.game_history if r in 'PB']) < 13:
            self.analysis_text, self.should_bet_now = "AI가 패턴을 학습하고 있습니다...", False; return
        
        chaos_index = self._calculate_chaos_index()
        self._update_dynamic_parameters(chaos_index)
        
        all_features = create_features(self.game_history, self.shoe_history)
        self.last_features = all_features
        if not all_features: return
        
        flow_prediction, flow_analysis_text = analyze_statistical_flow([r for r in self.shoe_history if r in 'PB'])

        mode = "혼돈" if chaos_index >= self.chaos_threshold else "안정"
        strategist_override = False
        if self.current_loss >= 5 and mode == "안정": mode = "혼돈"; strategist_override = True # [해결책] 서킷 브레이커
        if len(self.mode_performance_history) >= 3 and not strategist_override:
            last_3 = self.mode_performance_history[-3:]
            if last_3 == ['안정-X']*3 and mode == "안정": mode = "혼돈"; strategist_override = True
            elif last_3 == ['혼돈-X']*3 and mode == "혼돈": mode = "안정"; strategist_override = True

        final_prediction = None
        if mode == "혼돈":
            df_for_pretrained = pd.DataFrame([{k: all_features[k] for k in self.lgbm_model.feature_name_}])
            final_prediction = self.chaos_model.predict(df_for_pretrained)[0]
        else: # '안정' 모드: AI 위원회 가동
            preds = {}
            # 1. 고정 모델 예측
            df_for_pretrained = pd.DataFrame([{k: all_features[k] for k in self.lgbm_model.feature_name_}])
            preds['LGBM'] = self.lgbm_model.predict(df_for_pretrained)[0]
            # 2. 통계 흐름 분석 예측
            if flow_prediction: preds['Flow'] = flow_prediction
            # 3. 온라인 전문가 예측
            if self.online_short_ready:
                features_short = pd.DataFrame([{k: all_features[k] for k in self.short_term_feature_names}])
                preds['Online_Short'] = self.online_model_short.predict(features_short)[0]
            if self.online_long_ready:
                features_long = pd.DataFrame([{k: all_features[k] for k in self.long_term_feature_names}])
                preds['Online_Long'] = self.online_model_long.predict(features_long)[0]
            
            if preds: final_prediction = self.combiner.predict(preds)
            self.last_preds = preds
        
        self.next_prediction = final_prediction
        self.should_bet_now = (final_prediction is not None)
        self.last_mode_for_bet = mode
        
        override_note = " (강제 개입)" if strategist_override else ""
        base_analysis = f"AI 분석: [국면: {mode}{override_note} / 혼돈지수: {chaos_index:.0%} / 임계값: {self.chaos_threshold:.0%}]"
        final_pred_text = f"➡️ 최종 예측: {self.next_prediction}" if self.next_prediction else "➡️ 최종 예측: 관망"
        self.analysis_text = f"{base_analysis}\n{flow_analysis_text}\n{final_pred_text}"

    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return {"총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}", "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}
