import streamlit as st
import os
import copy
from logic import MLConcordanceAI
from ui import render_css, render_history_table, render_stats_panel, render_ai_analysis, render_controls

st.set_page_config(layout="wide", page_title="ML Baccarat AI", page_icon="🤖")
render_css()

LGBM_MODEL_FILE = 'baccarat_lgbm_model.joblib'
CHAOS_MODEL_FILE = 'baccarat_chaos_model.joblib'

model_files = [LGBM_MODEL_FILE, CHAOS_MODEL_FILE]
missing_files = [file for file in model_files if not os.path.exists(file)]

if missing_files:
    st.error(f"오류: 다음 모델 파일을 찾을 수 없습니다: {', '.join(missing_files)}. 훈련 스크립트를 실행해주세요.")
else:
    if 'pred' not in st.session_state:
        st.session_state.pred = MLConcordanceAI(
            lgbm_model_path=LGBM_MODEL_FILE,
            chaos_model_path=CHAOS_MODEL_FILE
        )
    if 'stack' not in st.session_state:
        st.session_state.stack = []
    if 'prev_stats' not in st.session_state:
        st.session_state.prev_stats = {}
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = True
    if 'show_ai_analysis' not in st.session_state:
        st.session_state.show_ai_analysis = True
    if 'show_overlay_setting' not in st.session_state:
        st.session_state.show_overlay_setting = True

    def handle_click(result):
        st.session_state.stack.append(copy.deepcopy(st.session_state.pred))
        pred = st.session_state.pred
        with st.spinner("AI가 다음 수를 분석 중입니다..."):
            pred.handle_input(result)
        if hasattr(pred, 'last_bet_result') and pred.last_bet_result is not None and st.session_state.get('show_overlay_setting', True):
            st.toast('적중!' if pred.last_bet_result else '미적중!', icon='🎉' if pred.last_bet_result else '💥')

    def handle_undo():
        if st.session_state.stack:
            st.session_state.pred = st.session_state.stack.pop()

    def handle_reset():
        st.session_state.pred = MLConcordanceAI(
            lgbm_model_path=LGBM_MODEL_FILE,
            chaos_model_path=CHAOS_MODEL_FILE
        )
        st.session_state.stack = []
        st.session_state.prev_stats = {}

    pred = st.session_state.pred
    render_history_table(pred)
    render_stats_panel(pred)
    render_ai_analysis(pred)
    render_controls(handle_click, handle_undo, handle_reset)
