# main.py

import streamlit as st
import os
import copy
from logic import MLConcordanceAI
from ui import render_css, render_history_table, render_stats_panel, render_ai_analysis, render_controls

# --- 페이지 기본 설정 ---
st.set_page_config(layout="wide", page_title="ML Baccarat AI", page_icon="🤖")

# --- CSS 렌더링 ---
render_css()

# --- 모델 파일 경로 정의 및 존재 여부 확인 (수정된 부분) ---
MODEL_FILE = 'baccarat_model.joblib'
LGBM_MODEL_FILE = 'baccarat_lgbm_model.joblib' # LGBM 모델 파일 경로 추가

if not os.path.exists(MODEL_FILE) or not os.path.exists(LGBM_MODEL_FILE):
    st.error(f"오류: 모델 파일('{MODEL_FILE}', '{LGBM_MODEL_FILE}')을 찾을 수 없습니다. 훈련 스크립트를 먼저 실행해주세요.")
else:
    # --- 세션 상태 초기화 ---
    if 'pred' not in st.session_state:
        # MLConcordanceAI 초기화 시 두 모델 경로 모두 전달 (수정된 부분)
        st.session_state.pred = MLConcordanceAI(model_path=MODEL_FILE, lgbm_model_path=LGBM_MODEL_FILE)
        st.session_state.pred.predict_next()
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

    # --- 버튼 클릭 핸들러 ---
    def handle_click(result):
        """P, B, T 버튼 클릭을 처리합니다."""
        if len(st.session_state.stack) > 100:
            st.session_state.stack.pop(0)
        st.session_state.stack.append(copy.deepcopy(st.session_state.pred))
        
        pred = st.session_state.pred
        pred.handle_input(result)
        
        if st.session_state.show_overlay_setting and pred.last_bet_result is not None:
            st.toast('적중!' if pred.last_bet_result else '미적중!', icon='🎉' if pred.last_bet_result else '💥')

    def handle_undo():
        """되돌리기 버튼 클릭을 처리합니다."""
        if st.session_state.stack:
            st.session_state.pred = st.session_state.stack.pop()

    def handle_reset():
        """초기화 버튼 클릭을 처리합니다."""
        # 초기화 시에도 두 모델 경로 모두 전달 (수정된 부분)
        st.session_state.pred = MLConcordanceAI(model_path=MODEL_FILE, lgbm_model_path=LGBM_MODEL_FILE)
        st.session_state.stack = []
        st.session_state.prev_stats = {} # 통계도 초기화

    # --- 메인 UI 렌더링 ---
    pred = st.session_state.pred
    
    # 1. 기록 테이블
    render_history_table(pred)
    
    # 2. 통계 패널
    render_stats_panel(pred)
    
    # 3. AI 분석 및 예측
    render_ai_analysis(pred)
    
    # 4. 컨트롤 버튼
    render_controls(handle_click, handle_undo, handle_reset)
