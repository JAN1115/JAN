import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import SGDClassifier
import numpy as np

# --- 헬퍼 클래스 및 함수 정의 (기존과 동일) ---

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
        
        # 가중치 투표 결과가 동률일 경우, LGBM > Flow > Online 우선순위로 결정
        if votes['P'] == votes['B']:
            if 'LGBM' in preds: return preds['LGBM']
            if 'Flow' in preds: return preds['Flow']
            if 'Online' in preds: return preds['Online']
            return None # 모든 예측이 없는 경우
        return 'P' if votes['P'] > votes['B'] else 'B'
    def update(self, preds, actual, hit):
        for strat, p in preds.items():
            if p in ('P', 'B') and p != actual: self.weights[strat] *= self.beta
        total = sum(self.weights.values())
        if total > 1e-9:
            for strat in self.weights: self.weights[strat] /= total

# --- [업그레이드] 통계적 흐름 분석 함수 ---
def analyze_statistical_flow(shoe_pb_history):
    """
    슈 전체의 통계적 흐름을 분석하여 P/B를 예측합니다.
    """
    if len(shoe_pb_history) < 20: # 최소 20게임 데이터가 쌓여야 분석 의미가 있음
        return None, "[흐름분석: 데이터 축적 중]"

    p_count = shoe_pb_history.count('P')
    b_count = shoe_pb_history.count('B')
    
    # 1. P-B 격차 계산
    gap = p_count - b_count
    
    # 2. 격차 모멘텀 계산 (최근 15게임)
    window = shoe_pb_history[-15:]
    momentum_gap = window.count('P') - window.count('B')

    # 3. 국면 진단 및 예측 로직
    phase = ""
    prediction = None

    if abs(gap) <= 3:
        phase = "팽팽한 접전"
        # 직전 결과의 반대에 베팅 (균형 회귀 전략)
        prediction = 'B' if shoe_pb_history[-1] == 'P' else 'P'
            
    elif gap > 3 and momentum_gap > 1:
        phase = "플레이어 강세"
        # 흐름을 따라 P에 베팅
        prediction = 'P'
        
    elif gap < -3 and momentum_gap < -1:
        phase = "뱅커 강세"
        # 흐름을 따라 B에 베팅
        prediction = 'B'

    elif (gap > 3 and momentum_gap < 0) or (gap < -3 and momentum_gap > 0):
        phase = "흐름 전환기"
        # 새로운 모멘텀을 따라 베팅
        prediction = 'P' if momentum_gap > 0 else 'B'
    
    else:
        phase = "흐름 탐색 중"
        prediction = None # 불확실한 경우 베팅하지 않음

    analysis_text = f"[흐름분석: {phase} / 전체격차:{gap} / 모멘텀:{momentum_gap}]"
    return prediction, analysis_text

# ---------- 통합 AI 클래스 (업그레이드 버전) ----------
class MLConcordanceAI:
    def __init__(self, lgbm_model_path='baccarat_lgbm_model.joblib',
                 chaos_model_path='baccarat_chaos_model.joblib'):
        self.lgbm_model = joblib.load(lgbm_model_path)
        self.chaos_model = joblib.load(chaos_model_path)
        self.online_model = SGDClassifier(loss='log_loss', random_state=42)
        
        # [업그레이드] 'Flow' 전략을 Combiner에 추가
        self.combiner = WeightedMajorityCombiner(strategies=['LGBM', 'Online', 'Flow'], initial_beta=0.95)
        
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
        
        all_features = create_features(self.game_history, self.shoe_history)
        self.last_features = all_features
        if not all_features: return

        features_for_pretrained = {k: all_features[k] for k in self.original_feature_names}
        df_for_pretrained = pd.DataFrame([features_for_pretrained])
        
        # [업그레이드] 통계 흐름 분석 로직 호출
        shoe_pb_history = [r for r in self.shoe_history if r in 'PB']
        flow_prediction, flow_analysis_text = analyze_statistical_flow(shoe_pb_history)

        mode = "혼돈" if chaos_index >= self.chaos_threshold else "안정"
        
        strategist_override = False
        if len(self.mode_performance_history) >= 3:
            last_3 = self.mode_performance_history[-3:]
            if last_3 == ['안정-X']*3 and mode == "안정": mode = "혼돈"; strategist_override = True
            elif last_3 == ['혼돈-X']*3 and mode == "혼돈": mode = "안정"; strategist_override = True

        final_prediction = None
        if mode == "혼돈":
            final_prediction = self.chaos_model.predict(df_for_pretrained)[0]
        elif mode == "안정":
            preds = {}
            preds['LGBM'] = self.lgbm_model.predict(df_for_pretrained)[0]
            if self.online_model_ready:
                preds['Online'] = self.online_model.predict(pd.DataFrame([all_features]))[0]
            
            # [업그레이드] '흐름 분석' 전략의 예측값을 preds에 추가
            if flow_prediction:
                preds['Flow'] = flow_prediction
            
            if preds: final_prediction = self.combiner.predict(preds)
            self.last_preds = preds
        
        self.next_prediction = final_prediction
        self.should_bet_now = (final_prediction is not None)
        self.last_mode_for_bet = mode
        
        override_note = " (전략 AI 개입)" if strategist_override else ""
        
        # [업그레이드] 분석 텍스트에 흐름 분석 결과 추가
        base_analysis = (f"AI 분석: [국면: {mode}{override_note} / "
                         f"혼돈지수: {chaos_index:.0%} / 임계값: {self.chaos_threshold:.0%}]")
        final_pred_text = f"➡️ 최종 예측: {self.next_prediction}" if self.next_prediction else "➡️ 최종 예측: 관망"
        
        self.analysis_text = f"{base_analysis}\n{flow_analysis_text}\n{final_pred_text}"


    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return {"총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
                "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}


# --- 이 아래 부분은 AI를 실행하기 위한 예시 코드입니다. ---
# 터미널에서 직접 실행할 때 사용하세요.

if __name__ == '__main__':
    # AI 모델 파일이 있는지 확인하세요.
    # 예: baccarat_lgbm_model.joblib, baccarat_chaos_model.joblib
    # 파일이 없다면 이 코드는 실행되지 않습니다.
    try:
        ai = MLConcordanceAI()
        print("="*50)
        print("바카라 AI 예측 프로그램 (통계 흐름 분석 업그레이드 버전)")
        print("결과를 입력하세요 (P: 플레이어, B: 뱅커, T: 타이). '종료' 입력 시 종료.")
        print("="*50)
        
        while True:
            print("\n" + ai.analysis_text)
            print(f"통계: {ai.get_stats()}")
            
            user_input = input("지난 게임 결과 입력: ").upper()
            
            if user_input in ['P', 'B', 'T']:
                ai.handle_input(user_input)
            elif user_input == '종료':
                print("프로그램을 종료합니다.")
                break
            else:
                print("잘못된 입력입니다. P, B, T 중 하나를 입력하세요.")

    except FileNotFoundError:
        print("오류: AI 모델 파일(.joblib)을 찾을 수 없습니다.")
        print("현재 폴더에 'baccarat_lgbm_model.joblib'와 'baccarat_chaos_model.joblib' 파일이 있는지 확인하세요.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")
