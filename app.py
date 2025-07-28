import streamlit as st
from collections import Counter
from logic import BaccaratAI_Ensemble
from ui import render_css, render_history_table, render_stats_panel, render_ai_analysis, render_controls

st.set_page_config(layout="wide", page_title="Baccarat AI Ensemble", page_icon="🧠")
render_css()

# --- 세션 상태 초기화 ---
if 'ai' not in st.session_state:
    st.session_state.ai = BaccaratAI_Ensemble()
    if not st.session_state.ai.models:
        st.error("앙상블 모델(baccarat_model_*.joblib)을 찾을 수 없습니다. create_model.py를 먼저 실행해주세요.")
        st.stop()
if 'game_history' not in st.session_state: st.session_state.game_history = []
if 'predictions' not in st.session_state: st.session_state.predictions = []
if 'individual_predictions' not in st.session_state: st.session_state.individual_predictions = []
if 'hit_record' not in st.session_state: st.session_state.hit_record = []
if 'stats' not in st.session_state:
    st.session_state.stats = { 'bet_count': 0, 'correct': 0, 'current_win': 0, 'max_win': 0, 'current_loss': 0, 'max_loss': 0 }
if 'analysis_text' not in st.session_state: st.session_state.analysis_text = "AI 대기 중..."
if 'last_recommendation' not in st.session_state: st.session_state.last_recommendation = None
if 'system_mode' not in st.session_state: st.session_state.system_mode = "AI"
# --- 로직 수정: 2개의 주기 인덱스 초기화 ---
if 'cycle_index_1' not in st.session_state: st.session_state.cycle_index_1 = 0 # 1차 주기 (15턴)
if 'cycle_index_2' not in st.session_state: st.session_state.cycle_index_2 = 0 # 2차 주기 (7턴)
if 'internal_ai_loss_streak' not in st.session_state: st.session_state.internal_ai_loss_streak = 0
if 'internal_ai_loss_streak_history' not in st.session_state: st.session_state.internal_ai_loss_streak_history = []
if 'show_stats' not in st.session_state: st.session_state.show_stats = True
if 'show_ai_analysis' not in st.session_state: st.session_state.show_ai_analysis = True
if 'show_overlay_setting' not in st.session_state: st.session_state.show_overlay_setting = True
if 'count_tie_in_cycle' not in st.session_state: st.session_state.count_tie_in_cycle = True


# --- 핸들러 함수 정의 ---
def handle_click(result):
    stats = st.session_state.stats
    recommendation_for_this_round = st.session_state.last_recommendation
    original_ai_prediction = st.session_state.predictions[-1] if st.session_state.predictions else None

    hit = None
    is_betting_round = (recommendation_for_this_round is not None and result != 'T')
    if is_betting_round:
        stats['bet_count'] += 1
        if recommendation_for_this_round == result:
            hit = True; stats['correct'] += 1; stats['current_win'] += 1; stats['current_loss'] = 0
        else:
            hit = False; stats['current_win'] = 0; stats['current_loss'] += 1
        stats['max_win'] = max(stats['max_win'], stats['current_win'])
        stats['max_loss'] = max(stats['max_loss'], stats['current_loss'])

    is_ai_round = (original_ai_prediction is not None and result != 'T')
    if is_ai_round:
        if original_ai_prediction == result: st.session_state.internal_ai_loss_streak = 0
        else: st.session_state.internal_ai_loss_streak += 1

    st.session_state.hit_record.append('O' if hit else 'X') if is_betting_round else st.session_state.hit_record.append(None)
    st.session_state.internal_ai_loss_streak_history.append(st.session_state.internal_ai_loss_streak)
    st.session_state.game_history.append(result)

    # --- 로직 수정: 주기 인덱스 업데이트 ---
    if st.session_state.get('count_tie_in_cycle', True):
        st.session_state.cycle_index_1 = (st.session_state.cycle_index_1 + 1) % 15
        st.session_state.cycle_index_2 = (st.session_state.cycle_index_2 + 1) % 7
    else:
        if result in ['P', 'B']:
            st.session_state.cycle_index_1 = (st.session_state.cycle_index_1 + 1) % 15
            st.session_state.cycle_index_2 = (st.session_state.cycle_index_2 + 1) % 7

    with st.spinner("AI가 다음 수를 분석 중입니다..."):
        ai = st.session_state.ai
        next_prediction, _, individual_preds = ai.predict(st.session_state.game_history)
        st.session_state.predictions.append(next_prediction)
        st.session_state.individual_predictions.append(individual_preds)
        st.session_state.analysis_text = ai.analysis_text

    if hit is not None and st.session_state.get('show_overlay_setting', True):
        st.toast('적중!' if hit else '미적중!', icon='🎉' if hit else '💥')

def handle_reset():
    st.session_state.clear(); st.rerun()

def handle_undo():
    if not st.session_state.game_history:
        st.toast("더 이상 되돌릴 수 없습니다.", icon="⚠️"); return

    last_result_to_undo = st.session_state.game_history[-1]

    st.session_state.game_history.pop(); st.session_state.hit_record.pop()
    st.session_state.predictions.pop(); st.session_state.individual_predictions.pop()
    st.session_state.internal_ai_loss_streak_history.pop()

    if st.session_state.internal_ai_loss_streak_history:
        st.session_state.internal_ai_loss_streak = st.session_state.internal_ai_loss_streak_history[-1]
    else: st.session_state.internal_ai_loss_streak = 0

    # --- 로직 수정: 주기 인덱스 감소 ---
    if st.session_state.get('count_tie_in_cycle', True):
        st.session_state.cycle_index_1 = (st.session_state.get('cycle_index_1', 0) - 1 + 15) % 15
        st.session_state.cycle_index_2 = (st.session_state.get('cycle_index_2', 0) - 1 + 7) % 7
    else:
        if last_result_to_undo in ['P', 'B']:
            st.session_state.cycle_index_1 = (st.session_state.get('cycle_index_1', 0) - 1 + 15) % 15
            st.session_state.cycle_index_2 = (st.session_state.get('cycle_index_2', 0) - 1 + 7) % 7

    stats = st.session_state.stats
    temp_hit_record = [h for h in st.session_state.hit_record if h is not None]
    stats['bet_count'] = len(temp_hit_record); stats['correct'] = temp_hit_record.count('O')
    cw, cl, mw, ml = 0, 0, 0, 0
    for hit_val in st.session_state.hit_record:
        if hit_val == 'O': cw += 1; cl = 0
        elif hit_val == 'X': cl += 1; cw = 0
        mw = max(mw, cw); ml = max(ml, cl)
    stats['current_win'], stats['current_loss'] = cw, cl; stats['max_win'], stats['max_loss'] = mw, ml

    ai = st.session_state.ai
    next_prediction, _, individual_preds = ai.predict(st.session_state.game_history)
    st.session_state.predictions.append(next_prediction); st.session_state.individual_predictions.append(individual_preds)
    st.session_state.analysis_text = ai.analysis_text
    st.toast("↩️ 이전 상태로 되돌렸습니다.")

# --- 최종 로직 통합 컨테이너 ---
class UIDataContainer:
    def __init__(self):
        stats = st.session_state.stats
        self.history = st.session_state.game_history
        self.hit_record = st.session_state.hit_record
        self.next_prediction = st.session_state.predictions[-1] if st.session_state.predictions else None

        self.bet_count = stats['bet_count']; self.correct = stats['correct']
        self.accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0

        if self.bet_count < 10: AI_FAILURE_THRESHOLD = 3
        elif self.accuracy > 60.0: AI_FAILURE_THRESHOLD = 4
        elif self.accuracy < 45.0: AI_FAILURE_THRESHOLD = 2
        else: AI_FAILURE_THRESHOLD = 3

        internal_loss_streak = st.session_state.get('internal_ai_loss_streak', 0)
        is_safeguard_active = internal_loss_streak >= AI_FAILURE_THRESHOLD

        raw_prediction = self.next_prediction

        if is_safeguard_active:
            pb_history = [r for r in self.history if r in 'PB']
            last_pb_result = pb_history[-1] if pb_history else None
            if last_pb_result:
                raw_prediction = last_pb_result
        
        # --- 1차 주기 (기존 3-2-2-3-1-4) 적용 ---
        bet_layer_1 = raw_prediction
        cycle_index_1 = st.session_state.get('cycle_index_1', 0)
        is_inverted_1 = (3 <= cycle_index_1 <= 4) or \
                      (7 <= cycle_index_1 <= 9) or \
                      (11 <= cycle_index_1 <= 14)
        if is_inverted_1 and bet_layer_1 in ['P', 'B']:
            bet_layer_1 = 'B' if bet_layer_1 == 'P' else 'P'

        # --- 2차 주기 ("3-4" 패턴) 적용 ---
        self.final_recommendation = bet_layer_1  # 1차 결과를 2차 주기의 입력값으로 사용
        cycle_index_2 = st.session_state.get('cycle_index_2', 0)
        # 3턴 정벳(0,1,2), 4턴 역벳(3,4,5,6)
        is_inverted_2 = (cycle_index_2 >= 3)
        if is_inverted_2 and self.final_recommendation in ['P', 'B']:
            self.final_recommendation = 'B' if self.final_recommendation == 'P' else 'P'
            
        # --- UI 텍스트 업데이트 ---
        current_analysis_text = st.session_state.analysis_text
        # 최종 분석 상태 결정 (1차 또는 2차에서 하나라도 전환되었으면 '전환분석')
        status_text = "전환분석" if (raw_prediction != self.final_recommendation) else "일반분석"
        
        if st.session_state.get('count_tie_in_cycle', True):
            total_turns = len(self.history)
            turn1 = ((total_turns - 1) % 15) + 1 if total_turns > 0 else 1
            turn2 = ((total_turns - 1) % 7) + 1 if total_turns > 0 else 1
        else:
            pb_count = sum(1 for r in self.history if r in 'PB')
            turn1 = ((pb_count - 1) % 15) + 1 if pb_count > 0 else 1
            turn2 = ((pb_count - 1) % 7) + 1 if pb_count > 0 else 1
        
        turn_info = f"(1차 {turn1}/15 | 2차 {turn2}/7: 최종 {status_text})"
        self.analysis_text = f"{current_analysis_text} {turn_info}"
        # --------------------------------

        st.session_state.system_mode = "AI"
        self.system_mode = "AI"

        st.session_state.last_recommendation = self.final_recommendation
        self.should_bet_now = (self.final_recommendation is not None)

        self.current_win = stats['current_win']; self.max_win = stats['max_win']
        self.current_loss = stats['current_loss']; self.max_loss = stats['max_loss']

    def get_stats(self):
        return {"총입력": len(self.history), "적중률(%)": f"{self.accuracy:.2f}", "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}

# --- UI 렌더링 ---
ui_data = UIDataContainer()
render_history_table(ui_data)
render_stats_panel(ui_data)
render_ai_analysis(ui_data)
render_controls(handle_click, handle_undo, handle_reset)
