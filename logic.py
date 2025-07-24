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

# ---------- AI 클래스 (다수결 필터 최종 버전) ----------
class MLConcordanceAI:
    # --- 설정값 ---
    MOMENTUM_LOOKBACK = 10 # 격차 모멘텀을 분석할 과거 데이터의 길이

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
        
        print("--- 새로운 게임(슈)을 시작합니다. (다수결 신뢰도 필터) ---")

    def handle_input(self, result):
        if result == 'P': self.p_count += 1
        elif result == 'B': self.b_count += 1
        if result in 'PB': self.gap_history.append(self.p_count - self.b_count)

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
                    self.current_loss = 0
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
                        self.prediction_mode = 'NORMAL'

                self.max_win = max(self.max_win, self.current_win)
                self.max_loss = max(self.max_loss, self.current_loss)
            
            self.hit_record.append('O' if hit else ('X' if hit is not None else None))
            self.history.append(result)
            self.game_history.append(result)
        
        self.predict_next()

    def _analyze_gap_momentum(self):
        """P/B 격차의 역사를 분석하여 장기적인 추세를 예측합니다."""
        if len(self.gap_history) < self.MOMENTUM_LOOKBACK: return None
        recent_gaps = self.gap_history[-self.MOMENTUM_LOOKBACK:]
        widening_count, narrowing_count = 0, 0
        for i in range(1, len(recent_gaps)):
            if abs(recent_gaps[i]) > abs(recent_gaps[i-1]): widening_count += 1
            elif abs(recent_gaps[i]) < abs(recent_gaps[i-1]): narrowing_count += 1
        
        if widening_count > narrowing_count:
            return 'P' if self.p_count >= self.b_count else 'B'
        elif narrowing_count > widening_count:
            return 'B' if self.p_count >= self.b_count else 'P'
        else:
            return None

    def predict_next(self):
        pb_history = [r for r in self.game_history if r in 'PB']
        
        # 1. AI 내부의 모든 예측 로직을 독립적으로 실행하여 투표 준비
        pred_normal = predict_always_in_chaos(pb_history, self.chaos_strategy)
        pred_fast = pb_history[-1] if len(pb_history) >= 1 else None
        pred_delayed = pb_history[-2] if len(pb_history) >= 2 else None
        pred_momentum = self._analyze_gap_momentum()
        
        # 2. 투표 집계
        votes = [p for p in [pred_normal, pred_fast, pred_delayed, pred_momentum] if p is not None]
        p_votes = votes.count('P')
        b_votes = votes.count('B')
        
        # 3. 다수결 또는 동률 처리 규칙에 따라 최종 예측 결정
        final_prediction = None
        decision_reason = ""

        if p_votes > b_votes:
            final_prediction = 'P'
            decision_reason = f"다수결({p_votes}:{b_votes})에 따라 'P'를 선택합니다."
        elif b_votes > p_votes:
            final_prediction = 'B'
            decision_reason = f"다수결({p_votes}:{b_votes})에 따라 'B'를 선택합니다."
        else: # 동률일 경우, 현재 '혼합 방어 사이클' 모드의 예측을 우선
            current_mode_pred = None
            if self.prediction_mode == 'FAST': current_mode_pred = pred_fast
            elif self.prediction_mode == 'DELAYED': current_mode_pred = pred_delayed
            else: current_mode_pred = pred_normal
            
            final_prediction = current_mode_pred
            decision_reason = f"동률({p_votes}:{b_votes})! 현재 모드({self.prediction_mode}) 예측 '{final_prediction}'을 우선합니다."

        self.next_prediction = final_prediction
        self.should_bet_now = (self.next_prediction is not None)

        # 분석 텍스트 생성
        vote_text = f"투표-> 일반:{pred_normal}, 빠른:{pred_fast}, 지연:{pred_delayed}, 모멘텀:{pred_momentum}"
        self.analysis_text = f"{vote_text}\n➡️ {decision_reason}"
        
    def get_stats(self):
        accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        return { "총입력": len(self.history), "베팅횟수": self.bet_count, "적중률(%)": f"{accuracy:.2f}",
            "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss,
            "최대연패": self.max_loss, "현재모드": self.prediction_mode }
