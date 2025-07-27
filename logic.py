import numpy as np
import xgboost
import joblib
from collections import deque, Counter
import pandas as pd
import json
from scipy.stats import mode

class FeatureExtractor:
    def __init__(self): self.reset()
    def reset(self): self.current_loss = 0; self.p_count = 0; self.b_count = 0; self.gap_history = deque(maxlen=10)
    def update_state_from_history(self, pb_history):
        self.reset();
        if not pb_history: return
        for i in range(1, len(pb_history)):
            prev_slice = pb_history[:i]; current_val = pb_history[i]
            if not prev_slice: continue
            pred = self._predict_chaos(prev_slice)
            if pred is not None and pred != current_val: self.current_loss += 1
            else: self.current_loss = 0
        self.p_count = pb_history.count('P'); self.b_count = pb_history.count('B')
        gaps = []; p, b = 0, 0
        for r in pb_history:
            if r == 'P': p += 1
            else: b += 1
            gaps.append(p - b)
        self.gap_history.extend(gaps[-10:])
    def get_features(self, pb_history):
        if len(pb_history) < 11: return None
        self.update_state_from_history(pb_history)
        pred_normal = self._predict_chaos(pb_history); pred_fast = pb_history[-1]; pred_delayed = pb_history[-2]; pred_momentum = self._predict_momentum()
        vol_recent = sum(1 for i in range(-5, 0) if pb_history[i] != pb_history[i-1]); vol_prior = sum(1 for i in range(-10, -5) if pb_history[i] != pb_history[i-1]); volatility_delta = vol_recent - vol_prior
        streak_events_10 = sum(1 for i in range(-10, 0) if pb_history[i] == pb_history[i-1]); chop_events_10 = 10 - streak_events_10; trend_dominance_score = streak_events_10 - chop_events_10
        volatility_10 = sum(1 for i in range(-10, -1) if pb_history[i] != pb_history[i+1])
        preds = [pred_normal, pred_fast, pred_delayed, pred_momentum]; valid_preds = [p for p in preds if p is not None]; consensus_score = 0
        if len(valid_preds) > 1: p_votes = valid_preds.count('P'); b_votes = valid_preds.count('B'); consensus_score = max(p_votes, b_votes) / len(valid_preds)
        last_val = pb_history[-1]; current_streak_len = 1
        for i in range(len(pb_history) - 2, -1, -1):
            if pb_history[i] == last_val: current_streak_len += 1
            else: break
        current_streak_type = (1 if last_val == 'P' else -1) if current_streak_len > 1 else 0
        is_choppy = 1 if len(pb_history) >= 4 and pb_history[-1] != pb_history[-2] and pb_history[-2] != pb_history[-3] and pb_history[-3] != pb_history[-4] else 0
        return {'pred_normal': 1 if pred_normal == 'P' else 0, 'pred_momentum': 1 if pred_momentum == 'P' else (0 if pred_momentum == 'B' else 0.5),'current_loss': self.current_loss, 'gap': self.p_count - self.b_count, 'total_count': len(pb_history),'lag_3': 1 if pb_history[-3] == 'P' else 0, 'lag_5': 1 if pb_history[-5] == 'P' else 0,'gap_ma_5': np.mean(list(self.gap_history)[-5:]) if len(self.gap_history) >= 5 else 0,'loss_x_gap': self.current_loss * (self.p_count - self.b_count), 'volatility_10': volatility_10,'consensus_score': consensus_score, 'current_streak_type': current_streak_type,'current_streak_length': current_streak_len, 'is_choppy': is_choppy, 'volatility_delta': volatility_delta, 'trend_dominance_score': trend_dominance_score}
    def _predict_chaos(self, pb_history):
        if len(pb_history) < 2: return None
        return 'B' if pb_history[-1] == 'P' else 'P'
    def _predict_momentum(self):
        if len(self.gap_history) < 5: return None
        return 'P' if self.p_count > self.b_count else 'B'

class BaccaratAI_Ensemble:
    def __init__(self, num_models=5):
        self.models = []
        self.feature_extractor = FeatureExtractor()
        self.elite_features = None
        self.analysis_text = "AI 대기 중..."
        try:
            with open('elite_features.json', 'r') as f:
                self.elite_features = json.load(f)
            for i in range(1, num_models + 1):
                model_filename = f"baccarat_model_{i}.joblib"
                self.models.append(joblib.load(model_filename))
            print(f"✅ {num_models}개의 앙상블 모델 로딩 완료!")
        except Exception as e:
            print(f"🚨 모델 또는 elite_features.json 로딩 중 오류 발생: {e}")
            self.models = []

    def predict(self, game_history):
        if not self.models or not self.elite_features:
            self.analysis_text = "모델이 로드되지 않아 예측할 수 없습니다."
            return None, 0.5, []

        pb_history = [r for r in game_history if r in 'PB']
        all_features = self.feature_extractor.get_features(pb_history)
        
        if all_features is None:
            self.analysis_text = "데이터가 부족하여 예측할 수 없습니다."
            return None, 0.5, []

        try:
            feature_df = pd.DataFrame([all_features])[self.elite_features]
        except KeyError as e:
            self.analysis_text = f"특성 키 오류: {e}"
            return None, 0.5, []
        
        individual_predictions = []
        probas = []
        for model in self.models:
            pred_result = model.predict(feature_df)[0]
            p_or_b = 'P' if pred_result == 1 else 'B'
            individual_predictions.append(p_or_b)
            pred_proba = model.predict_proba(feature_df)[0]
            confidence = pred_proba[1] if p_or_b == 'P' else pred_proba[0]
            probas.append(confidence)

        if not individual_predictions: return None, 0.5, []
        
        vote_counts = Counter(individual_predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        
        confidence = np.mean([p for pred, p in zip(individual_predictions, probas) if pred == final_prediction])
        
        vote_counts_text = f"P {individual_predictions.count('P')} : B {individual_predictions.count('B')}"
        self.analysis_text = f"AI 예측: {final_prediction} (신뢰도: {confidence:.2%}, 투표: {vote_counts_text})"
        
        return final_prediction, confidence, individual_predictions
