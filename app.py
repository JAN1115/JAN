import streamlit as st
import random
import copy
from collections import Counter

# ----------------------- UltraSafetyAI -----------------------
class UltraSafetyAI:
    def __init__(self, min_history=4, warmup_turns=3):
        self.min_history = min_history
        self.warmup_turns = warmup_turns
        self.history = []
        self.pb_history = []
        self.correct = 0
        self.incorrect = 0
        self.current_win = 0
        self.current_loss = 0
        self.max_win = 0
        self.max_loss = 0

        self.last_result = None
        self.prev_prediction = None
        self.next_prediction = None
        self.hit_history = []

    def reverse_predict(self):
        pure = [x for x in self.pb_history if x in ('P','B')]
        if not pure:
            return random.choice(['P', 'B'])
        return 'B' if pure[-1] == 'P' else 'P'

    def trend_predict(self, window=2):
        pure = [x for x in self.pb_history if x in ('P','B')]
        if len(pure) < window:
            return None
        cnt = Counter(pure[-window:])
        if cnt['P'] > cnt['B']:
            return 'P'
        elif cnt['B'] > cnt['P']:
            return 'B'
        return None

    def ngram_predict(self, maxn=3):
        pure = [x for x in self.pb_history if x in ('P','B')]
        for n in range(maxn, 1, -1):
            if len(pure) < n: continue
            key = tuple(pure[-n:])
            c = Counter()
            for i in range(len(pure)-n):
                if tuple(pure[i:i+n]) == key:
                    c[pure[i+n]] += 1
            if c:
                return c.most_common(1)[0][0]
        return None

    def ultra_safety_predict(self):
        L = self.current_loss
        if L >= 6:
            return random.choice(['P', 'B'])
        elif L == 5:
            return 'B' if self.prev_prediction == 'P' else 'P'
        elif L == 4:
            choices = [self.reverse_predict(), random.choice(['P','B']), self.reverse_predict()]
            return random.choice(choices)
        elif L == 3:
            last = self.pb_history[-1] if len(self.pb_history) > 0 else random.choice(['P', 'B'])
            return 'B' if last == 'P' else 'P'
        else:
            choices = [
                self.trend_predict(window=3),
                self.ngram_predict(maxn=3),
                self.reverse_predict()
            ]
            filtered = [v for v in choices if v in ('P','B')]
            return Counter(filtered).most_common(1)[0][0] if filtered else random.choice(['P','B'])

    def handle_input(self, r):
        if r == 'T':
            self.history.append('T')
            self.last_result = None
            self.prev_prediction = self.next_prediction
            self.prepare_next_prediction()
            return
        self.pb_history.append(r)
        self.history.append(r)
        self.prev_prediction = self.next_prediction
        if len(self.pb_history) <= self.warmup_turns:
            self.last_result = None
            self.prepare_next_prediction()
            return
        prev_pred = self.prev_prediction
        if prev_pred in ('P', 'B') and r in ('P', 'B'):
            hit = (prev_pred == r)
            self.last_result = hit
            if hit:
                self.correct += 1
                self.current_win += 1
                self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
            else:
                self.incorrect += 1
                self.current_win = 0
                self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
            self.hit_history.append(hit)
        else:
            self.last_result = None
        self.prepare_next_prediction()

    def prepare_next_prediction(self):
        self.next_prediction = self.ultra_safety_predict()

    def stats(self):
        total_pb = len(self.pb_history)
        hitrate = round(self.correct / total_pb * 100, 2) if total_pb else 0
        return {
            '총입력':       len(self.history),
            '총예측(P/B)':  total_pb,
            '적중':         self.correct,
            '미적중':       self.incorrect,
            '적중률(%)':    hitrate,
            '현재연승':      self.current_win,
            '현재연패':      self.current_loss,
            '최대연승':      self.max_win,
            '최대연패':      self.max_loss
        }

# ------------------- Streamlit UI -------------------
st.set_page_config(layout="wide", page_title="UltraSafetyAI 연패방지 바카라", page_icon="🦾")
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {background: #181c21 !important; color: #eee;}
    .stButton>button {
        background: #23222a !important; color: #fff !important;
        font-size: 1.0em !important; border: 1.5px solid #39f3fa66 !important;
        border-radius: 10px !important; padding: 0.23em 0.1em !important;
        min-width: 32px !important; min-height: 32px !important;
        margin: 1px 1px 1px 1px !important; cursor: pointer !important;
        transition: 0.2s !important; box-shadow: 0 0 7px #39f3fa22 !important;
    }
    .stButton>button:hover {
        background: #2a3c44 !important; color: #39f3fa !important;
        border: 1.5px solid #39f3fa !important;
    }
    .neon { color: #39f3fa; text-shadow:0 0 6px #39f3fa,0 0 10px #28b3ff; font-weight: bold;}
    .neon-pink { color: #ff6fd8; text-shadow:0 0 6px #ff6fd8,0 0 10px #ff44b3; font-weight: bold;}
    .sixgrid { font-size: 1em; line-height: 1.11; letter-spacing: 2px; font-family: 'Segoe UI', 'Pretendard', 'Malgun Gothic', sans-serif;}
    </style>
""", unsafe_allow_html=True)

if 'stack' not in st.session_state:
    st.session_state.stack = []
if 'pred' not in st.session_state:
    st.session_state.pred = UltraSafetyAI()
pred = st.session_state.pred

def push_state():
    st.session_state.stack.append(copy.deepcopy(pred))
def undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
def full_reset():
    st.session_state.pred = UltraSafetyAI()
    st.session_state.stack.clear()

ICONS = {'P':'🔵','B':'🔴','T':'🟢'}

btn_cols = st.columns(5, gap="small")
with btn_cols[0]:
    if st.button("🔵P", key="btnP1", use_container_width=True):
        push_state(); pred.handle_input("P")
with btn_cols[1]:
    if st.button("🔴B", key="btnB1", use_container_width=True):
        push_state(); pred.handle_input("B")
with btn_cols[2]:
    if st.button("🟢T", key="btnT1", use_container_width=True):
        push_state(); pred.handle_input("T")
with btn_cols[3]:
    if st.button("↩️", key="btnU1", use_container_width=True):
        undo()
with btn_cols[4]:
    if st.button("🗑️", key="btnR1", use_container_width=True):
        full_reset()

if len(pred.pb_history) < pred.warmup_turns:
    st.markdown('<div class="neon" style="font-size:1.15em;">데이터 쌓는 중...</div>', unsafe_allow_html=True)
elif len(pred.pb_history) == pred.warmup_turns:
    if pred.next_prediction:
        st.markdown(f'<div class="neon" style="font-size:1.7em;">🎯 4번째 턴 예측 → {ICONS[pred.next_prediction]}</div>', unsafe_allow_html=True)
    st.info(f"{pred.warmup_turns}턴 데이터 쌓기 끝! 이제 4번째 턴부터 예측/적중/통계 집계 시작.")
else:
    # NEW: 연패구간 표시!
    if pred.current_loss >= 3:
        st.markdown(f"<div class='neon-pink' style='font-size:1.12em'>⚡연패구간: {pred.current_loss}회</div>", unsafe_allow_html=True)
    if pred.next_prediction:
        st.markdown(
            f'<div class="neon" style="font-size:2.1em;">🎯 예측 → {ICONS[pred.next_prediction]}</div>',
            unsafe_allow_html=True
        )
    if hasattr(pred, 'last_result'):
        if pred.last_result is True:
            st.success("✅ 적중")
        elif pred.last_result is False:
            st.error("❌ 미적중")

# 6매 기록
st.markdown('<div class="neon" style="font-size:1em;">6매 기록</div>', unsafe_allow_html=True)
history = pred.history
max_row = 6
cols = (len(history) + max_row - 1) // max_row
six_grid = []
for r in range(max_row):
    row = []
    for c in range(cols):
        idx = c * max_row + r
        if idx < len(history):
            row.append(ICONS[history[idx]])
        else:
            row.append("　")
    six_grid.append(row)
six_html = '<div class="sixgrid" style="margin-bottom:8px;letter-spacing:1.5px;">'
for row in six_grid:
    six_html += '<span style="display:inline-block;min-width:1.2em;">' + " ".join(row) + '</span><br>'
six_html += '</div>'
st.markdown(six_html, unsafe_allow_html=True)
st.markdown("---")

# 통계
s = pred.stats()
stat_cols = st.columns([1,1,1,1])
stat_cols[0].markdown(f"<div class='neon'>현재 연승<br><span style='font-size:1.08em'>{s['현재연승']}</span></div>", unsafe_allow_html=True)
stat_cols[1].markdown(f"<div class='neon-pink'>현재 연패<br><span style='font-size:1.08em'>{s['현재연패']}</span></div>", unsafe_allow_html=True)
stat_cols[2].markdown(f"<div style='color:#39f3fa'>적중률<br><span style='font-size:1.07em;font-weight:bold'>{s['적중률(%)']}%</span></div>", unsafe_allow_html=True)
stat_cols[3].markdown(f"<div style='color:#fa5252'>최대연패<br><span style='font-size:1.08em'>{s['최대연패']}</span></div>", unsafe_allow_html=True)
with st.expander("📊 전체 통계 자세히 (터치/클릭)", expanded=False):
    st.json(s)

st.markdown('<div class="neon" style="font-size:1.18em; margin-top:0.7em;">UltraSafetyAI 연패방지 AI 바카라</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-pink" style="font-size:0.95em;">실전형 연패자동회피 · 초민감 전략전환</div>', unsafe_allow_html=True)

if pred.next_prediction is None:
    pred.prepare_next_prediction()
