import streamlit as st
import random
import copy
from collections import Counter
import numpy as np
from itertools import groupby

class MetaAdaptivePredictor:
    def __init__(self, warmup_turns=3):
        self.history = []
        self.correct = 0
        self.incorrect = 0
        self.current_win = 0
        self.current_loss = 0
        self.max_win = 0
        self.max_loss = 0
        self.warmup_turns = warmup_turns
        self.last_result = None
        self.next_prediction = None
        self.prev_prediction = None
        self.mode = "NORMAL"   # 현재 모드: NORMAL/REVERSE/TREND/CYCLE/RANDOM
        self.mode_score = Counter()
        self.hit_history = []

    # ========== 패턴 구간 감지 ==========
    def detect_reverse(self, window=8):
        # 최근 window 중 예측이 60% 이상 반대로 나올 때
        if len(self.hit_history) < window: return False
        rev_cnt = sum(1 for x in self.hit_history[-window:] if x is False)
        return rev_cnt / window > 0.6

    def detect_trend(self, window=7):
        if len(self.history) < window: return None
        cnt = Counter(self.history[-window:])
        if max(cnt.values()) >= window - 1:
            return cnt.most_common(1)[0][0]
        return None

    def detect_cycle(self, maxlen=4):
        for n in range(maxlen, 1, -1):
            if len(self.history) < n * 2: continue
            last = self.history[-n:]
            prev = self.history[-n*2:-n]
            if last == prev:
                return last
        return None

    def detect_random(self, window=14):
        if len(self.history) < window: return False
        cnt = Counter(self.history[-window:])
        diff = abs(cnt['P'] - cnt['B'])
        return diff <= 3  # 비슷하게 나올 때

    # ========== 예측 전략 (각 상황별) ==========
    def ngram_predict(self, maxn=5):
        # 기본 n-그램 예측 (최근 패턴 기반)
        best = []
        for n in range(maxn, 1, -1):
            if len(self.history) < n: continue
            key = tuple(self.history[-n:])
            c = Counter()
            for i in range(len(self.history)-n):
                if tuple(self.history[i:i+n]) == key:
                    c[self.history[i+n]] += 1
            if c:
                best.append(c.most_common(1)[0][0])
        if best:
            return Counter(best).most_common(1)[0][0]
        return None

    def trend_predict(self):
        trend = self.detect_trend(window=7)
        return trend if trend else None

    def cycle_predict(self):
        cycle = self.detect_cycle(maxlen=4)
        if cycle:
            idx = (len(self.history)) % len(cycle)
            return cycle[idx]
        return None

    def reverse_predict(self):
        # 가장 최근 예측의 반대로
        if self.prev_prediction in ('P','B'):
            return 'B' if self.prev_prediction == 'P' else 'P'
        return None

    def random_predict(self):
        return random.choice(['P', 'B'])

    def meta_predict(self):
        votes = []
        # ngram, trend, cycle, reverse, random 각 전략으로 예측
        ngram = self.ngram_predict()
        if ngram: votes.append(('ngram', ngram))
        trend = self.trend_predict()
        if trend: votes.append(('trend', trend))
        cycle = self.cycle_predict()
        if cycle: votes.append(('cycle', cycle))
        if self.mode == "REVERSE":
            rev = self.reverse_predict()
            if rev: votes.extend([('reverse', rev)]*2)  # 역베팅은 2표
        if self.mode == "RANDOM":
            votes.append(('random', self.random_predict()))

        # 투표 가중치: 모드에 따라 해당 예측 2표
        if self.mode == "TREND" and trend: votes.append(('trend', trend))
        if self.mode == "CYCLE" and cycle: votes.append(('cycle', cycle))
        if self.mode == "NORMAL" and ngram: votes.append(('ngram', ngram))
        # 1표도 없으면 랜덤
        if not votes:
            pred = self.random_predict()
        else:
            by_val = Counter([v for k,v in votes])
            pred = by_val.most_common(1)[0][0]
        self.prev_prediction = pred
        self.next_prediction = pred
        return pred

    # ========== 전략 선택/적응 ==========
    def update_mode(self):
        # 최근 결과/패턴 기반으로 현재 전략 모드 전환
        if self.detect_reverse(window=8):
            self.mode = "REVERSE"
        elif self.detect_trend(window=7):
            self.mode = "TREND"
        elif self.detect_cycle(maxlen=4):
            self.mode = "CYCLE"
        elif self.detect_random(window=14):
            self.mode = "RANDOM"
        else:
            self.mode = "NORMAL"
        self.mode_score[self.mode] += 1

    def handle_input(self, r):
        self.update_mode()
        pred = self.next_prediction
        self.history.append(r)
        if len([x for x in self.history if x in ('P','B')]) <= self.warmup_turns:
            self.last_result = None
            self.next_prediction = self.meta_predict()
            return
        if pred in ('P', 'B') and r in ('P', 'B'):
            hit = (pred == r)
            self.last_result = hit
            if hit:
                self.correct += 1
                self.current_win += 1
                self.max_win = max(self.max_win, self.current_win)
                self.current_loss = 0
            else:
                self.incorrect += 1
                self.current_win = 0
                self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
            self.hit_history.append(hit)
        else:
            self.last_result = None
        self.next_prediction = self.meta_predict()

    def stats(self):
        total_pb = len([x for x in self.history if x in ('P','B')])
        hitrate = round(self.correct / total_pb * 100, 2) if total_pb else 0
        # 연승/연패 집계
        win_list = []
        loss_list = []
        if self.hit_history:
            for k, g in groupby(self.hit_history):
                length = sum(1 for _ in g)
                if k:
                    win_list.append(length)
                else:
                    loss_list.append(length)
        return {
            '총입력':       len(self.history),
            '총예측(P/B)':  total_pb,
            '적중':         self.correct,
            '미적중':       self.incorrect,
            '적중률(%)':    hitrate,
            '현재연승':      self.current_win,
            '현재연패':      self.current_loss,
            '최대연승':      self.max_win,
            '최대연패':      self.max_loss,
            '평균연승':      round(np.mean(win_list),2) if win_list else 0,
            '평균연패':      round(np.mean(loss_list),2) if loss_list else 0,
        }

# ====== Streamlit UI 이하 구조 동일 (pred만 교체) ======

st.set_page_config(layout="wide", page_title="실전 AI 적응형 예측기", page_icon="🦾")
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #181c21 !important; color: #eee;
    }
    .stButton>button {
        background: #23222a !important; color: #fff !important;
        font-size: 1.0em !important;
        border: 1.5px solid #39f3fa66 !important;
        border-radius: 10px !important;
        padding: 0.23em 0.1em !important;
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
    st.session_state.pred = MetaAdaptivePredictor()
pred = st.session_state.pred

def push_state():
    st.session_state.stack.append(copy.deepcopy(pred))
def undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
def full_reset():
    st.session_state.pred = MetaAdaptivePredictor()
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

# 2. 예측 결과
if len([x for x in pred.history if x in ('P','B')]) < pred.warmup_turns:
    st.markdown('<div class="neon" style="font-size:1.15em;">데이터 쌓는 중...</div>', unsafe_allow_html=True)
elif len([x for x in pred.history if x in ('P','B')]) == pred.warmup_turns:
    if pred.next_prediction:
        st.markdown(f'<div class="neon" style="font-size:1.7em;">🎯 4번째 턴 예측 → {ICONS[pred.next_prediction]}</div>', unsafe_allow_html=True)
    st.info(f"{pred.warmup_turns}턴 데이터 쌓기 끝! 이제 4번째 턴부터 예측/적중/통계 집계 시작.")
else:
    if pred.next_prediction:
        st.markdown(
            f'<div class="neon" style="font-size:2.1em;">🎯 예측 → {ICONS[pred.next_prediction]} <span style="font-size:0.6em;">[{pred.mode}]</span></div>',
            unsafe_allow_html=True
        )
    if hasattr(pred, 'last_result'):
        if pred.last_result is True:
            st.success("✅ 적중")
        elif pred.last_result is False:
            st.error("❌ 미적중")

# 3. 6매 기록
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
            row.append(ICONS.get(history[idx], "　"))
        else:
            row.append("　")
    six_grid.append(row)
six_html = '<div class="sixgrid" style="margin-bottom:8px;letter-spacing:1.5px;">'
for row in six_grid:
    six_html += '<span style="display:inline-block;min-width:1.2em;">' + " ".join(row) + '</span><br>'
six_html += '</div>'
st.markdown(six_html, unsafe_allow_html=True)
st.markdown("---")

# 4. 통계
s = pred.stats()
stat_cols = st.columns([1,1,1,1])
stat_cols[0].markdown(f"<div class='neon'>현재 연승<br><span style='font-size:1.08em'>{s['현재연승']}</span></div>", unsafe_allow_html=True)
stat_cols[1].markdown(f"<div class='neon-pink'>현재 연패<br><span style='font-size:1.08em'>{s['현재연패']}</span></div>", unsafe_allow_html=True)
stat_cols[2].markdown(f"<div style='color:#39f3fa'>적중률<br><span style='font-size:1.07em;font-weight:bold'>{s['적중률(%)']}%</span></div>", unsafe_allow_html=True)
stat_cols[3].markdown(f"<div style='color:#fa5252'>최대연패<br><span style='font-size:1.08em'>{s['최대연패']}</span></div>", unsafe_allow_html=True)
with st.expander("📊 전체 통계 자세히 (터치/클릭)", expanded=False):
    st.json(s)

st.markdown('<div class="neon" style="font-size:1.18em; margin-top:0.7em;">AI 실전 적응형 메타 러너 예측기</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-pink" style="font-size:0.95em;">복합 패턴 자동 감지/전략 전환/실전 적중률 강화</div>', unsafe_allow_html=True)

if pred.next_prediction is None:
    pred.next_prediction = pred.meta_predict()
