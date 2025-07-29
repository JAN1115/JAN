import streamlit as st
from collections import Counter
from logic import BaccaratAI
from ui import render_css, render_history_table, render_stats_panel, render_ai_analysis, render_controls

st.set_page_config(layout="wide", page_title="Baccarat AI Ensemble", page_icon="🧠")
render_css()

# --- 세션 상태 초기화 ---
if 'ai' not in st.session_state:
    st.session_state.ai = BaccaratAI()
    if not st.session_state.ai.models:
        st.error("모델 파일을 찾을 수 없습니다. create_model.py를 먼저 실행해주세요.")
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
if 'show_stats' not in st.session_state: st.session_state.show_stats = True
if 'show_ai_analysis' not in st.session_state: st.session_state.show_ai_analysis = True
if 'show_overlay_setting' not in st.session_state: st.session_state.show_overlay_setting = True
if 'count_tie_in_cycle' not in st.session_state: st.session_state.count_tie_in_cycle = True

def handle_click(result):
    stats = st.session_state.stats
    recommendation_for_this_round = st.session_state.last_recommendation
    
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
        
    st.session_state.hit_record.append('O' if hit else 'X') if is_betting_round else st.session_state.hit_record.append(None)
    st.session_state.game_history.append(result)

    with st.spinner("AI가 다음 수를 분석 중입니다..."):
        ai = st.session_state.ai
        next_prediction, _, individual_preds, mode = ai.predict(st.session_state.game_history)
        st.session_state.predictions.append(next_prediction)
        st.session_state.individual_predictions.append(individual_preds)
        st.session_state.analysis_text = ai.analysis_text
        st.session_state.system_mode = mode

    if hit is not None and st.session_state.get('show_overlay_setting', True):
        st.toast('적중!' if hit else '미적중!', icon='🎉' if hit else '💥')

def handle_reset():
    st.session_state.clear(); st.rerun()

def handle_undo():
    if not st.session_state.game_history:
        st.toast("더 이상 되돌릴 수 없습니다.", icon="⚠️"); return

    st.session_state.game_history.pop()
    st.session_state.hit_record.pop()
    st.session_state.predictions.pop()
    st.session_state.individual_predictions.pop()
    
    stats = st.session_state.stats
    temp_hit_record = [h for h in st.session_state.hit_record if h is not None]
    stats['bet_count'] = len(temp_hit_record)
    stats['correct'] = temp_hit_record.count('O')
    
    cw, cl, mw, ml = 0, 0, 0, 0
    for hit_val in st.session_state.hit_record:
        if hit_val == 'O': cw += 1; cl = 0
        elif hit_val == 'X': cl += 1; cw = 0
        mw = max(mw, cw); ml = max(ml, cl)
    stats['current_win'], stats['current_loss'] = cw, cl; stats['max_win'], stats['max_loss'] = mw, ml

    ai = st.session_state.ai
    next_prediction, _, individual_preds, mode = ai.predict(st.session_state.game_history)
    st.session_state.predictions.append(next_prediction); st.session_state.individual_predictions.append(individual_preds)
    st.session_state.analysis_text = ai.analysis_text
    st.session_state.system_mode = mode
    st.toast("↩️ 이전 상태로 되돌렸습니다.")

class UIDataContainer:
    def __init__(self):
        stats = st.session_state.stats
        self.history = st.session_state.game_history
        self.hit_record = st.session_state.hit_record
        
        # --- ▼ [수정됨] 겉값(Overlay) 로직 제거 ---
        # AI의 순수 예측값을 그대로 사용합니다.
        raw_prediction = st.session_state.predictions[-1] if st.session_state.predictions else None
        
        self.final_recommendation = raw_prediction
        self.analysis_text = st.session_state.analysis_text
        # --- ▲ [수정됨] 여기까지 ---

        self.system_mode = st.session_state.system_mode
        self.bet_count = stats['bet_count']; self.correct = stats['correct']
        self.accuracy = (self.correct / self.bet_count * 100) if self.bet_count > 0 else 0
        
        st.session_state.last_recommendation = self.final_recommendation
        self.should_bet_now = (self.final_recommendation is not None)

        self.current_win = stats['current_win']; self.max_win = stats['max_win']
        self.current_loss = stats['current_loss']; self.max_loss = stats['max_loss']

    def get_stats(self):
        return {"총입력": len(self.history), "적중률(%)": f"{self.accuracy:.2f}", "현재연승": self.current_win, "최대연승": self.max_win, "현재연패": self.current_loss, "최대연패": self.max_loss}

ui_data = UIDataContainer()
render_history_table(ui_data)
render_stats_panel(ui_data)
render_ai_analysis(ui_data)
render_controls(handle_click, handle_undo, handle_reset)
