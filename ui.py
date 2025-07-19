# ui.py

import streamlit as st
import numpy as np

def render_css():
    """앱 전체에 적용될 CSS 스타일을 렌더링합니다."""
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #0c111b !important; color: #e0fcff !important; }
    .stButton>button { border: none; border-radius: 12px; padding: 12px 24px; color: white; font-size: 1.1em; font-weight: bold; transition: all .3s ease-out; }
    .stButton>button:hover { transform: translateY(-3px); filter: brightness(1.3); }
    div[data-testid="stHorizontalBlock"]>div:nth-child(1) .stButton>button { background: #0c3483; box-shadow: 0 0 8px #3b82f6, 0 0 12px #3b82f6; }
    div[data-testid="stHorizontalBlock"]>div:nth-child(2) .stButton>button { background: #880e4f; box-shadow: 0 0 8px #f06292, 0 0 12px #f06292; }
    div[data-testid="stHorizontalBlock"]>div:nth-child(3) .stButton>button { background: #1b5e20; box-shadow: 0 0 8px #4caf50, 0 0 12px #4caf50; }
    .top-stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 15px -10px 15px -10px; }
    .stat-item { background: rgba(30, 45, 60, .7); border-radius: 10px; padding: 8px; text-align: center; border: 1px solid #2a3b4e; transition: all 0.4s ease-out; }
    .stat-label { font-size: 0.9em; color: #a0c8d0; margin-bottom: 4px; display: block; }
    .stat-value { font-size: 1.25em; font-weight: bold; color: #e0fcff; }
    .desktop-only { display: inline; }
    .mobile-only { display: none; }
    .stat-changed-neon { animation: neon-flash 1s ease-in-out; }
    @keyframes neon-flash { 50% { box-shadow: 0 0 5px #fff, 0 0 10px #00fffa, 0 0 15px #00fffa; border-color: #00fffa; transform: scale(1.03); } }
    .fire-animation { display: inline-block; animation: fire-burn 1.2s infinite ease-in-out; text-shadow: 0 0 5px #ff5722, 0 0 10px #ff5722, 0 0 15px #ff9800; }
    @keyframes fire-burn { 0%, 100% { transform: scale(1.0) rotate(-1deg); } 50% { transform: scale(1.15) rotate(1deg); } }
    .skull-animation { display: inline-block; animation: skull-shake 0.4s infinite linear; text-shadow: 0 0 5px #f44336, 0 0 10px #f44336; }
    @keyframes skull-shake { 0%, 100% { transform: translateY(0) rotate(0); } 25% { transform: translateY(1px) rotate(-3deg); } 75% { transform: translateY(-1px) rotate(3deg); } }
    .ai-waiting-bar { background: linear-gradient(90deg, rgba(43,41,0,0.6) 0%, rgba(80,70,0,0.9) 50%, rgba(43,41,0,0.6) 100%); border-radius: 10px; padding: 12px; margin: 10px 0 15px 0; color: #ffd400; font-size: 1.2em; font-weight: 900; text-align: center; width: 100%; }
    .blinking-text { animation: blink-animation 2s linear infinite; }
    @keyframes blink-animation { 50% { opacity: 0.4; } }
    .rotating-hourglass { display: inline-block; animation: rotate 2s linear infinite; }
    @keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
    .latest-result-pop { animation: pop-in 0.6s ease-out; }
    @keyframes pop-in { 0% { transform: scale(0.5); } 50% { transform: scale(1.4); } 100% { transform: scale(1.0); } }
    @keyframes pulse-glow { 0%, 100% { transform: scale(1); text-shadow: 0 0 4px rgba(255,255,255,0.7), 0 0 8px #00e6e6, 0 0 12px #00e6e6; } 50% { transform: scale(1.1); text-shadow: 0 0 8px rgba(255,255,255,1), 0 0 16px #00ffff, 0 0 25px #00ffff; } }
    .history-container { margin-bottom: 10px; }
    .history-table { border-spacing: 2px 1px; width: 100%; }
    .history-cell { padding: 0; text-align: center; height: 26px; width: 26px; }
    .history-symbol {
        border-radius: 50%; font-weight: bold; font-size: 12px;
        display: inline-flex; align-items: center; justify-content: center;
        width: 22px; height: 22px; line-height: 1;
    }
    .sixgrid-fluo { color:#d4ffb3; background:rgba(100,255,110,.25); }
    .sixgrid-miss { color:#ffb3b3; background:rgba(255,100,110,.25); }
    @media (max-width: 768px) {
        .stButton>button { padding: 10px 18px; font-size: 1.0em; }
        .top-stats-grid { gap: 4px; margin-left: 0; margin-right: 0; }
        .stat-item { padding: 6px 4px; }
        .stat-label { font-size: 0.75em; margin-bottom: 2px; }
        .stat-value { font-size: 1.0em; }
        .desktop-only { display: none; }
        .mobile-only { display: inline; }
    }
    @media (max-width: 400px) {
        .history-cell { height: 22px; width: 22px; }
        .history-symbol { font-size: 10px; width: 18px; height: 18px; }
    }
    </style>
    """, unsafe_allow_html=True)

def render_history_table(pred):
    """결과 기록 테이블을 HTML로 렌더링합니다."""
    st.markdown("---")
    history, hitrec = pred.history, pred.hit_record
    max_row = 6
    ncols = max(10, (len(history) + max_row - 1) // max_row)
    six_html = '<div class="history-container"><table class="history-table"><tbody>'
    for r in range(max_row):
        six_html += "<tr>"
        for c in range(ncols):
            idx = c * max_row + r
            cell_content = "&nbsp;"
            if idx < len(history):
                ICONS = {'P': '🔵', 'B': '🔴', 'T': '🟢'}
                val = ICONS.get(history[idx], "")
                color_class = ""
                if idx < len(hitrec) and hitrec[idx] is not None:
                    color_class = "sixgrid-fluo" if hitrec[idx] == "O" else "sixgrid-miss"
                anim_class = "latest-result-pop" if idx == len(history) - 1 else ""
                cell_content = f'<span class="history-symbol {color_class} {anim_class}">{val}</span>'
            six_html += f'<td class="history-cell">{cell_content}</td>'
        six_html += "</tr>"
    six_html += '</tbody></table></div>'
    st.markdown(six_html, unsafe_allow_html=True)
    st.markdown("---")

def render_stats_panel(pred):
    """통계 정보 패널을 렌더링합니다."""
    if st.session_state.show_stats:
        s = pred.get_stats()
        prev_s = st.session_state.get('prev_stats', {})
        win_anim_class = "stat-changed-neon" if s['현재연승'] > 0 and s['현재연승'] != prev_s.get('현재연승', 0) else ""
        loss_anim_class = "stat-changed-neon" if s['현재연패'] > 0 and s['현재연패'] != prev_s.get('현재연패', 0) else ""
        win_icon = f"<span class='fire-animation'>🔥</span>" if s['현재연승'] > 0 else "⚪"
        loss_icon = f"<span class='skull-animation'>💀</span>" if s['현재연패'] > 0 else "⚪"

        st.markdown(f"""
        <div class="top-stats-grid">
            <div class="stat-item"><span class="stat-label">🎯 적중률</span><span class="stat-value">{s['적중률(%)']}%</span></div>
            <div class="stat-item"><span class="stat-label">📊 총 베팅</span><span class="stat-value">{pred.bet_count}/{s['총입력']}</span></div>
            <div class="stat-item {win_anim_class}"><span class="stat-label">{win_icon} 연승</span><span class="stat-value"><span class="desktop-only">{s['현재연승']} (최대 {s['최대연승']})</span><span class="mobile-only">{s['현재연승']}/{s['최대연승']}</span></span></div>
            <div class="stat-item {loss_anim_class}"><span class="stat-label">{loss_icon} 연패</span><span class="stat-value"><span class="desktop-only">{s['현재연패']} (최대 {s['최대연패']})</span><span class="mobile-only">{s['현재연패']}/{s['최대연패']}</span></span></div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.prev_stats = s.copy()

def render_ai_analysis(pred):
    """AI 분석 및 예측 결과를 렌더링합니다."""
    if st.session_state.show_ai_analysis:
        st.markdown(f"🧠 **AI 분석**: {pred.analysis_text}", unsafe_allow_html=True)
        npred, should_bet = pred.next_prediction, pred.should_bet_now
        if should_bet and npred:
            turn_count = pred.get_stats()['총입력']
            entrance_animation_name = f"pop-in-effect-{turn_count}"
            st.markdown(f"""
            <style>
            @keyframes {entrance_animation_name} {{ 0% {{ transform: scale(0.5); opacity: 0; }} 70% {{ transform: scale(1.2); opacity: 1; }} 100% {{ transform: scale(1.0); opacity: 1; }} }}
            .animated-prediction-icon {{ display: inline-block; font-size: 2.5em; animation: {entrance_animation_name} 0.8s cubic-bezier(0.250, 0.460, 0.450, 0.940) both, pulse-glow 2.5s ease-in-out infinite 1s; }}
            </style>""", unsafe_allow_html=True)
            ICONS = {'P': '🔵', 'B': '🔴'}
            st.markdown(f'''<div style="text-align:center; margin:5px 0 15px 0; height: 60px; display:flex; align-items:center; justify-content:center;"><span class="animated-prediction-icon">{ICONS.get(npred, npred)}</span></div>''', unsafe_allow_html=True)
        else:
            display_text = pred.analysis_text.split(":")[-1].strip()
            is_collecting = "데이터 수집 중" in display_text
            anim_class = "blinking-text" if is_collecting else ""
            st.markdown(f'<div class="ai-waiting-bar {anim_class}"><span class="rotating-hourglass">⏳</span> {display_text}...</div>', unsafe_allow_html=True)

def render_controls(handle_click, handle_undo, handle_reset):
    """입력 버튼, 되돌리기, 초기화 등 컨트롤 버튼을 렌더링합니다."""
    button_cols = st.columns([1, 1, 1, 0.5, 0.5, 1.2])
    button_cols[0].button("플레이어 (P)", use_container_width=True, on_click=handle_click, args=("P",))
    button_cols[1].button("뱅커 (B)", use_container_width=True, on_click=handle_click, args=("B",))
    button_cols[2].button("타이 (T)", use_container_width=True, on_click=handle_click, args=("T",))

    if button_cols[3].button("↩️", help="이전 상태로 되돌립니다.", on_click=handle_undo):
        st.rerun()

    if button_cols[4].button("🗑️", help="모든 기록을 초기화합니다.", on_click=handle_reset):
        st.rerun()

    with button_cols[5].expander("⚙️ 설정", expanded=False):
        st.toggle("통계 표시", key="show_stats", help="적중률, 총 베팅, 연승/연패 통계를 표시하거나 숨깁니다.")
        st.toggle("AI 분석 표시", key="show_ai_analysis", help="AI의 분석 및 다음 예측을 표시하거나 숨깁니다.")
        st.toggle("적중/미적중 알림 표시", key="show_overlay_setting", help="베팅 결과에 따라 화면 우측 상단에 알림을 표시합니다.")
