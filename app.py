import streamlit as st
import os
import copy
from logic import MLConcordanceAI # logic.py에 최종 AI 로직이 있다고 가정
from ui import render_css, render_history_table, render_stats_panel, render_ai_analysis, render_controls

# --- 페이지 및 CSS 설정 ---
st.set_page_config(layout="wide", page_title="ML Baccarat AI", page_icon="🤖")
render_css()

# --- 모델 파일 관련 코드 삭제 ---
# [삭제] 최종 로직은 더 이상 외부 모델 파일을 사용하지 않으므로 모든 관련 코드를 제거합니다.

# --- 세션 상태 초기화 ---
if 'pred' not in st.session_state:
    # [수정] MLConcordanceAI를 인자 없이 호출합니다.
    st.session_state.pred = MLConcordanceAI()
    
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

# --- 핸들러 함수 정의 ---
def handle_click(result):
    st.session_state.stack.append(copy.deepcopy(st.session_state.pred))
    pred = st.session_state.pred
    with st.spinner("AI가 다음 수를 분석 중입니다..."):
        pred.handle_input(result)
    # last_bet_result 와 같은 속성이 logic.py에 정의되어 있다는 가정 하에 동작합니다.
    if hasattr(pred, 'last_bet_result') and pred.last_bet_result is not None and st.session_state.get('show_overlay_setting', True):
        st.toast('적중!' if pred.last_bet_result else '미적중!', icon='🎉' if pred.last_bet_result else '💥')

def handle_undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()

def handle_reset():
    # [수정] MLConcordanceAI를 인자 없이 호출합니다.
    st.session_state.pred = MLConcordanceAI()
    st.session_state.stack = []
    st.session_state.prev_stats = {}

# --- UI 렌더링 ---
pred = st.session_state.pred
render_history_table(pred)
render_stats_panel(pred)
render_ai_analysis(pred)
render_controls(handle_click, handle_undo, handle_reset)
