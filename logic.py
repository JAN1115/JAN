import numpy as np
import pandas as pd
import joblib
from collections import Counter, deque
import random
import xgboost as xgb
from sklearn.ensemble import VotingClassifier

class BaccaratShoe:
    def __init__(self):
        self.shoe = []
        self.reset_shoe()
    def reset_shoe(self):
        single_deck = [2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 1] * 4
        self.shoe = single_deck * 8
        random.shuffle(self.shoe)
        self.burn_cards()
        self.set_cut_card()
    def burn_cards(self):
        if not self.shoe: return
        first_card = self.shoe.pop(0)
        num_to_burn = 10 if first_card == 0 else first_card
        for _ in range(num_to_burn):
            if self.shoe: self.shoe.pop(0)
    def set_cut_card(self):
        cut_position = random.randint(70 * 5, 75 * 5)
        if cut_position < len(self.shoe): self.shoe = self.shoe[:-cut_position]
    def _get_card_value(self):
        if not self.shoe: self.reset_shoe()
        return self.shoe.pop(0)
    def deal(self):
        player_cards = [self._get_card_value(), self._get_card_value()]
        banker_cards = [self._get_card_value(), self._get_card_value()]
        player_score = sum(player_cards) % 10
        banker_score = sum(banker_cards) % 10
        if player_score < 8 and banker_score < 8:
            if player_score <= 5:
                player_third_card = self._get_card_value()
                player_cards.append(player_third_card)
                if (banker_score <= 2) or \
                   (banker_score == 3 and player_third_card != 8) or \
                   (banker_score == 4 and player_third_card in [2,3,4,5,6,7]) or \
                   (banker_score == 5 and player_third_card in [4,5,6,7]) or \
                   (banker_score == 6 and player_third_card in [6,7]):
                    banker_cards.append(self._get_card_value())
            elif banker_score <= 5:
                banker_cards.append(self._get_card_value())
        player_final = sum(player_cards) % 10
        banker_final = sum(banker_cards) % 10
        if player_final > banker_final: return 'P'
        elif banker_final > player_final: return 'B'
        else: return 'T'
    def get_remaining_cards_info(self):
        total = len(self.shoe)
        if total == 0: return {'rem_cards': 0, 'rem_4_ratio': 0, 'rem_5_ratio': 0, 'rem_6_ratio': 0, 'rem_7_ratio': 0}
        counts = Counter(self.shoe)
        return {'rem_cards': total, 'rem_4_ratio': counts.get(4, 0)/total, 'rem_5_ratio': counts.get(5, 0)/total, 'rem_6_ratio': counts.get(6, 0)/total, 'rem_7_ratio': counts.get(7, 0)/total}

class BaccaratAI:
    def __init__(self):
        self.models = {}
        self.shoe = BaccaratShoe()
        self.full_history = []
        self.prediction_history = [] # AI 자신의 예측 기록을 저장
        self.load_models()
        self.analysis_text = "AI 대기 중..."
        self.system_mode = "AI"

    def load_models(self):
        model_names = ['general_1', 'general_2', 'streak', 'choppy', 'breaker', 'ensemble']
        for name in model_names:
            try:
                self.models[name] = joblib.load(f"baccarat_model_{name}.joblib")
            except FileNotFoundError:
                print(f"Warning: Model file baccarat_model_{name}.joblib not found.")
        if not self.models:
            print("Error: No models were loaded.")

    def _extract_features(self, full_history, prediction_history):
        pb_history = [r for r in full_history if r in 'PB']
        if len(pb_history) < 10: return None
        
        features = {}
        # --- 기존 특징 ---
        features['p_ratio'] = pb_history.count('P') / len(pb_history)
        features['b_ratio'] = pb_history.count('B') / len(pb_history)
        features['last_is_p'] = 1 if pb_history[-1] == 'P' else 0
        
        current_streak_len = 0
        if len(pb_history) > 1:
            last_result = pb_history[-1]
            for i in range(len(pb_history) - 2, -1, -1):
                if pb_history[i] == last_result: current_streak_len += 1
                else: break
        features['streak_len'] = current_streak_len + 1
        
        # --- 새로운 메타 특징 (Meta-Features) ---
        
        # 1. 패턴 밀림 (Lag) 감지
        features['prediction_lag_1'] = 0
        if len(prediction_history) >= 2 and len(pb_history) >= 2:
            if prediction_history[-2] == pb_history[-1]:
                features['prediction_lag_1'] = 1

        # 2. 통계적 모멘텀 (변화율)
        if len(pb_history) >= 10:
            p_ratio_last_5 = pb_history[-5:].count('P') / 5
            p_ratio_prev_5 = pb_history[-10:-5].count('P') / 5
            features['p_ratio_momentum'] = p_ratio_last_5 - p_ratio_prev_5
        else:
            features['p_ratio_momentum'] = 0

        # 3. 변동성 가속도
        if len(pb_history) >= 20:
            changes_last_10 = np.diff([1 if r == 'P' else 0 for r in pb_history[-10:]])
            volatility_last_10 = np.sum(np.abs(changes_last_10))
            changes_prev_10 = np.diff([1 if r == 'P' else 0 for r in pb_history[-20:-10:]])
            volatility_prev_10 = np.sum(np.abs(changes_prev_10))
            features['volatility_acceleration'] = volatility_last_10 - volatility_prev_10
        else:
            features['volatility_acceleration'] = 0
            
        features.update(self.shoe.get_remaining_cards_info())
        return features

    def predict(self, game_history):
        self.full_history = game_history
        pb_history = [r for r in self.full_history if r in 'PB']
        
        if len(self.models) < 6: return None, 0.5, [], "AI"

        features = self._extract_features(self.full_history, self.prediction_history)
        if features is None:
            self.analysis_text = f"{10 - len(pb_history)}턴의 데이터가 더 필요합니다."
            return None, 0.5, [], "AI"

        feature_df = pd.DataFrame([features])
        
        # 1. 모든 전문가 모델로부터 1차 예측 생성
        specialist_predictions = {}
        for name, model in self.models.items():
            if name == 'ensemble': continue # 최종 모델은 제외
            try:
                model_features = model.get_booster().feature_names
                feature_df_ordered = feature_df[model_features]
                # P가 나올 확률만 추출
                specialist_predictions[name] = model.predict_proba(feature_df_ordered)[0][1]
            except Exception as e:
                print(f"Specialist model '{name}' prediction error: {e}")
                specialist_predictions[name] = 0.5 # 오류 발생 시 중간값

        # 2. 1차 예측 결과를 입력 데이터로 하여 최종 앙상블(상황실장) 모델이 예측
        ensemble_input = pd.DataFrame([specialist_predictions])
        ensemble_model = self.models['ensemble']

        # --- ▼ [수정됨] 컬럼 순서를 강제로 고정 ---
        model_names = ['general_1', 'general_2', 'breaker', 'streak', 'choppy']
        ensemble_input = ensemble_input[model_names]
        # --- ▲ [수정됨] 여기까지 ---
        
        final_proba = ensemble_model.predict_proba(ensemble_input)[0]
        p_proba_index = np.where(ensemble_model.classes_ == 1)[0][0]
        
        final_prediction = 'P' if final_proba[p_proba_index] > 0.5 else 'B'
        final_confidence = final_proba[p_proba_index] if final_prediction == 'P' else 1 - final_proba[p_proba_index]
        
        # AI의 예측 기록 업데이트
        self.prediction_history.append(final_prediction)
        
        self.analysis_text = f"AI 예측: {final_prediction} (신뢰도: {final_confidence:.1%})"
        
        return final_prediction, final_confidence, [], "AI"
