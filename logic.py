import numpy as np
from collections import deque

# --- [핵심 로직] 상시 위기 대응 알고리즘 ---
def predict_always_in_chaos(pb_history, current_chaos_strategy):
    """'안정' 모드 없이, 항상 '자동 스위칭' 위기 대응 로직만으로 예측합니다."""
    if len(pb_history) < 2:
        return None

    if pb_history[-1] == pb_history[-2]:
        trend = pb_history[-1]
        if current_chaos_strategy == 'counter':
            return 'B' if trend == 'P' else 'P'
        else:
            return trend
    else:
        return 'B' if pb_history[-1] == 'P' else 'P'

# ---------- AI 클래스 (최종 진화 버전) ----------
class MLConcordanceAI:
    # --- 설정값 ---
    MOMENTUM_LOOKBACK = 10 # 격차 모멘텀을 분석할 과거 데이터의 길이
    WIN_RATE_LOOKBACK = 20 # 동적 우선순위를 위한 최근 승률 분석 길이

    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.history = []
        self.game_history = []
        self.hit_record = []
        self.bet_count = 0
        self.correct = 0
        self.max_win = 0
        self.max_loss = 0
        self.current_win = 0
        self.current_loss = 0
        self.analysis_text = "AI 초기화 중..."
        self.next_prediction = None
        self.should_bet_now = False
        self.chaos_strategy = 'counter'
        self.prediction_mode = 'NORMAL'
        
        self.p_count = 0
        self.b_count = 0
        self.gap_history = []
        
        # [추가] 동적 우선순위를 위한 변수
        self.primary_outcomes = deque(maxlen=self.WIN_RATE_LOOKBACK)
        self.momentum_outcomes = deque(maxlen=self.WIN_RATE_LOOKBACK)
        self.last_primary_pred = None
        self.last_momentum_pred = None
        
        print("--- 새로운 게임(슈)을 시작합니다. (동적 우선순위 최종 버전) ---")

    def handle_input(self, result):
        # 통계 및 격차 기록
        if result == 'P': self.p_count += 1
        elif result == 'B': self.b_count += 1
        if result in 'PB': self.gap_history.append(self.p_count - self.b_count)

        # [추가] 지난 예측들의 성공 여부를 기록하여 각 로직의 승률을 계산
        if self.last_primary_pred is not None:
            self.primary_outcomes.append(self.last_primary_pred == result)
        if self.last_momentum_pred is not None:
            self.momentum_outcomes.append(self.last_momentum_pred == result)

        if result == 'T':
            self.history.append('T')
            self.hit_record.append(None)
        else:
            hit = None
            if self.should_bet_now:
                hit = (self.next_prediction == result)
                self.bet_count += 1
                
                if hit:
                    self.correct += 1
                    self.current_win += 1
                    self.current_loss = 0 # [수정 완료] 오직 성공 시에만 연패가 초기화됨
                    if self.prediction_mode != 'NORMAL':
                        self.prediction_mode = 'NORMAL'
                else:
                    self.current_win = 0
                    self.current_loss += 1
                    
                    if self.prediction_mode == 'NORMAL':
                        self.chaos_strategy = 'trend' if self.chaos_strategy == 'counter' else 'counter'

                    # 혼합 방어 사이클 로직
                    if self.current_loss == 2: self.prediction_mode = 'FAST'
                    elif self.current_loss == 3: self.prediction_mode = 'DELAYED'
                    elif self.current_loss == 4: self.prediction_mode = 'FAST'
                    elif self.current_loss == 5: self.prediction_mode = 'DELAYED'
                    elif self.current_loss >= 6:
                        self.prediction_mode = 'NORMAL' # [수정 완료] 사이클만 리셋, 연패는 유지

                self.max_win = max(self.max_win, self.current_win)
                self.max_loss = max(self.max_loss, self.current_loss)
            
            self.hit_record.append('O' if hit else ('X' if hit is not None else None))
            self.history.append(result)
            self.game_history.append(result)
        
        self.predict_next()

    def _analyze_gap_momentum(self):
        if len(self.gap_history) < self.MOMENTUM_LOOKBACK: return None, "데이터 부족"
        recent_gaps = self.gap_history[-self.MOMENTUM_LOOKBACK:]
        widening_count, narrowing_count = 0, 0
        for i in range(1, len(recent_gaps)):
            if abs(recent_gaps[i]) > abs(recent_gaps[i-1]): widening_count += 1
            elif abs(recent_gaps[i]) < abs(recent_gaps[i-1]): narrowing_count += 1
        
        if widening_count > narrowing_count:
            return ('P' if self.p_count >= self.b_count else 'B'), "확대"
        elif narrowing_count > widening_count:
            return ('B' if self.p_count >= self.b_count else 'P'), "축소"
        else:
            return None, "중립"

    def predict_next(self):
        pb_history = [r for r in self.game_history if r in 'PB']
        primary_prediction, mode_text = None, ""

        # 1. 혼합 방어 사이클에 따른 1차 예측 생성
        if self.prediction_mode == 'FAST':
            mode_text = "빠른"
            if len(pb_history) >= 1: primary_prediction = pb_history[-1]
        elif self.prediction_mode == 'DELAYED':
            mode_text = "지연"
            if len(pb_history) >= 2: primary_prediction = pb_history[-2]
        else: # NORMAL 모드
            mode_text = "일반"
            prediction, _ = self.check_for_2_2_pattern(pb_history)
            if prediction: primary_prediction = prediction
            elif len(pb_history) >= 2: primary_prediction = predict_always_in_chaos(pb_history, self.chaos_strategy)

        # 2. 격차 모멘텀 분석을 통한 2차 예측
        momentum_prediction, momentum_trend = self._analyze_gap_momentum()

        # [추가] 다음 턴의 승률 계산을 위해 예측값을 미리 저장
        self.last_primary_pred = primary_prediction
        self.last_momentum_pred = momentum_prediction
        
        self.next_prediction = primary_prediction
        self.should_bet_now = (primary_prediction is not None)
        final_decision_text = f"➡️ 최종 예측: {self.next_prediction}" if self.should_bet_now else "➡️ 최종 예측: 관망"

        # 3. 최종 결정: 예측 충돌 시, 최근 승률이 높은 쪽을 선택
        if self.should_bet_now and momentum_prediction and (primary_prediction != momentum_prediction):
            p_rate = sum(self.primary_outcomes) / len(self.primary_outcomes) if self.primary_outcomes else 0
            m_rate = sum(self.momentum_outcomes) / len(self.momentum_outcomes) if self.momentum_outcomes else 0
            
            if m_rate > p_rate:
                self.next_prediction = momentum_prediction # 2차 예측으로 최종 결정 변경
                final_decision_text = f"➡️ 예측 충돌! 승률에 따라 2차 예측({self.next_prediction})을 우선합니다."
            else:
                final_decision_text = f"➡️ 예측 충돌! 승률에 따라 1차 예측({self.next_prediction})을 우선합니다."

        # 분석 텍스트 생성
        p_win_rate = (sum(self.primary_outcomes) / len(self.primary_outcomes) * 100) if self.primary_outcomes else 0
        m_win_rate = (sum(self.momentum_outcomes) / len(self.momentum_outcomes) * 100) if self.momentum_outcomes else 0
        primary_text = f"1차({mode_text} / {p_win_rate:.1f}%): {primary_prediction}"
        momentum_text = f"2차({momentum_trend} / {m_win_rate:.1f}%): {momentum_prediction}"
        self.analysis_text = f"{primary_text} | {momentum_text}\n{final_decision_text}"
        
    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return { "총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
            "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss,
            "최대연패": self.max_loss, "현재모드": self.prediction_mode }
    
    def check_for_2_2_pattern(self, pb_history):
        if len(pb_history) < 4: return None, None
        recent_4 = pb_history[-4:]
        if recent_4 == ['P', 'P', 'B', 'B']: return 'P', "[PPBB]"
        if recent_4 == ['B', 'B', 'P', 'P']: return 'B', "[BBPP]"
        return None, None
