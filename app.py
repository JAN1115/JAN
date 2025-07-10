import streamlit as st
import random
import copy
from collections import defaultdict, Counter
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from gensim import corpora, models

# ============= UltraStrongPredictor 클래스 =============
class UltraStrongPredictor:
    def __init__(self, min_history=4, decay=0.88, zigzag_n=4, longrun_n=4, warmup_turns=3):
        self.min_history = min_history
        self.decay = decay
        self.zigzag_n = zigzag_n
        self.longrun_n = longrun_n
        self.warmup_turns = warmup_turns

        self.history = []
        self.pb_history = []
        self.correct = 0
        self.incorrect = 0
        self.current_win = 0
        self.current_loss = 0
        self.max_win = 0
        self.max_loss = 0

        self.ngram = defaultdict(lambda: defaultdict(int))
        self.last_result = None
        self.prev_prediction = None
        self.next_prediction = None

        self.q_table = defaultdict(lambda: {'P': 0.0, 'B': 0.0})
        self.last_state = None
        self.last_action = None
        self.alpha = 0.2
        self.gamma = 0.9

        self.markov = defaultdict(Counter)
        self.hmm_states = [0, 1]
        self.hmm_emission = {(0, 'P'): 0.55, (0, 'B'): 0.45, (1, 'P'): 0.45, (1, 'B'): 0.55}
        self.hmm_trans = {(0,0):0.9, (0,1):0.1, (1,0):0.1, (1,1):0.9}
        self.hmm_last = 0

        self.hit_history = []

    def _pure(self):
        return [x for x in self.pb_history if x in ('P','B')]

    def predict_loss_safe(self):
        if self.current_loss >= 6:
            return random.choice(['P','B'])
        elif self.current_loss == 5:
            m1 = self.predict_markov()
            m2 = self.predict_hmm()
            vals = [m1, m2, self.predict_reverse(), random.choice(['P','B'])]
            votes = [v for v in vals if v in ('P','B')]
            return Counter(votes).most_common(1)[0][0] if votes else random.choice(['P','B'])
        elif self.current_loss == 4:
            return self.predict_forget_bad_pattern()
        elif self.current_loss == 3:
            return self.predict_multilayer_voting()
        elif self.current_loss == 2:
            return self.predict_reverse()
        return None

    def predict_ngram_ensemble(self, max_n=6, ignore_pattern=None):
        pure = self._pure()
        votes = []
        for n in range(2, max_n+1):
            if len(pure) < n:
                continue
            key = tuple(pure[-n:])
            if ignore_pattern and key == ignore_pattern:
                continue
            cnt = self.ngram.get(key, {})
            if cnt:
                votes.append(max(cnt, key=cnt.get))
        if votes:
            return Counter(votes).most_common(1)[0][0]
        return None

    def predict_china_bigroad(self):
        pure = self._pure()
        if len(pure) < 6:
            return None
        color = pure[-1]
        streak = 1
        for i in range(len(pure)-2, -1, -1):
            if pure[i] == color:
                streak += 1
            else:
                break
        return color if streak >= 2 else None

    def predict_china_side(self):
        pure = self._pure()
        if len(pure) < 8:
            return None
        if pure[-2] == pure[-1]:
            return pure[-1]
        return None

    def predict_zigzag(self, n=4):
        pure = self._pure()
        if len(pure) < n: return None
        last_n = pure[-n:]
        if all(last_n[i] != last_n[i-1] for i in range(1, n)):
            return 'B' if last_n[-1]=='P' else 'P'
        return None

    def predict_cycle(self, maxlen=6):
        pure = self._pure()
        for l in range(maxlen, 1, -1):
            if len(pure) < l*2: continue
            last = pure[-l:]
            for i in range(len(pure)-l*2+1):
                if pure[i:i+l] == last:
                    idx = i+l
                    if idx < len(pure):
                        return pure[idx]
        return None

    def predict_most_trend(self):
        pure = self._pure()
        if len(pure) < 8:
            return None
        cnt = Counter(pure[-8:])
        if cnt:
            return cnt.most_common(1)[0][0]
        return None

    def predict_zero_crossing(self, window=10):
        pure = self._pure()
        if len(pure) < window:
            return None
        mapping = {'P':1, 'B':-1}
        arr = np.cumsum([mapping[x] for x in pure[-window:]])
        zero_cross = np.where(np.diff(np.sign(arr)))[0]
        if len(zero_cross) > 0 and zero_cross[-1] >= window-3:
            return 'P' if arr[-1] > 0 else 'B'
        return None

    def predict_multilayer_voting(self):
        group1 = [self.predict_ngram_ensemble(), self.predict_cycle(), self.predict_zigzag()]
        group2 = [self.predict_most_trend(), self.predict_noise_cancelling(), self.predict_multi_avg()]
        group3 = [self.predict_china_bigroad(), self.predict_china_side()]
        reps = []
        for idx, group in enumerate([group1, group2, group3]):
            votes = [g for g in group if g in ('P','B')]
            if votes:
                reps.append(Counter(votes).most_common(1)[0][0])
        if reps:
            return Counter(reps).most_common(1)[0][0]
        return None

    def predict_noise_cancelling(self, window=5):
        pure = self._pure()
        if len(pure) < window:
            return None
        mapping = {'P':1, 'B':-1}
        arr = np.array([mapping[x] for x in pure[-window:]])
        mean = np.mean(arr)
        if mean > 0.2:
            return 'P'
        elif mean < -0.2:
            return 'B'
        else:
            return None

    def predict_multi_avg(self, window=6):
        pure = self._pure()
        if len(pure) < window:
            return None
        mapping = {'P':1, 'B':-1}
        arr = np.array([mapping[x] for x in pure[-window:]])
        sma = np.mean(arr)
        ema = np.mean(arr * np.linspace(0.5, 1.5, num=window)) / np.mean(np.linspace(0.5, 1.5, num=window))
        wma = np.average(arr, weights=np.arange(1,window+1))
        preds = []
        for val in [sma, ema, wma]:
            if val > 0.1:
                preds.append('P')
            elif val < -0.1:
                preds.append('B')
        if preds:
            return Counter(preds).most_common(1)[0][0]
        return None

    def predict_markov(self):
        pure = self._pure()
        if len(pure) < 3:
            return None
        key = tuple(pure[-2:])
        if key in self.markov:
            return max(self.markov[key], key=self.markov[key].get)
        return None

    def predict_hmm(self):
        pure = self._pure()
        if len(pure) < 4:
            return None
        last_obs = pure[-1]
        s = self.hmm_last
        pP = self.hmm_trans[s,0] * self.hmm_emission[0,last_obs]
        pB = self.hmm_trans[s,1] * self.hmm_emission[1,last_obs]
        self.hmm_last = 0 if pP > pB else 1
        return 'P' if self.hmm_last == 0 else 'B'

    def predict_forget_bad_pattern(self):
        pure = self._pure()
        if self.current_loss >= 3 and len(pure) > 6:
            last_pattern = tuple(pure[-3:])
            val = self.predict_ngram_ensemble(ignore_pattern=last_pattern)
            if val in ('P','B'):
                return val
        return None

    def predict_heuristic_switch(self):
        if self.current_loss >= 3:
            return self.predict_reverse()
        elif self.current_win >= 3:
            return self.predict_most_trend()
        return None

    def predict_reverse(self):
        pure = self._pure()
        if not pure: return random.choice(['P','B'])
        return 'B' if pure[-1] == 'P' else 'P'

    def predict(self):
        safe = self.predict_loss_safe()
        if safe: return safe
        meta = self.predict_multilayer_voting()
        if meta in ('P','B'): return meta
        forget = self.predict_forget_bad_pattern()
        if forget in ('P','B'): return forget
        zc = self.predict_zero_crossing()
        if zc in ('P','B'): return zc
        for func in [
            self.predict_heuristic_switch,
            self.predict_ngram_ensemble,
            self.predict_china_bigroad,
            self.predict_china_side,
            self.predict_zigzag,
            self.predict_cycle,
            self.predict_most_trend,
            self.predict_noise_cancelling,
            self.predict_multi_avg,
            self.predict_markov,
            self.predict_hmm,
        ]:
            res = func()
            if res in ('P','B'): return res
        return self.predict_reverse()

    def update_ngram(self, r):
        pure = self._pure()
        for d in range(2, min(len(pure), 6)+1):
            key = tuple(pure[-d:])
            self.ngram[key][r] += 1

    def update_markov(self, r):
        pure = self._pure()
        if len(pure) < 3:
            return
        key = tuple(pure[-3:-1])
        self.markov[key][r] += 1

    def prepare_next_prediction(self):
        self.next_prediction = self.predict()
        pure = self._pure()
        if len(pure) >= 6:
            state = tuple(pure[-4:])
            action = self.next_prediction
            self.last_state = state
            self.last_action = action

    def handle_input(self, r):
        if r == 'T':
            self.history.append('T')
            self.last_result = None
            self.prev_prediction = self.next_prediction
            self.prepare_next_prediction()
            return

        self.pb_history.append(r)
        self.update_ngram(r)
        self.update_markov(r)
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

# ============ Streamlit UI 코드 ==============
st.set_page_config(layout="wide", page_title="AI 바카라 예측기", page_icon="🎲")

# --- 네온 다크 스타일 + 버튼 가로정렬 (모바일 강제) ---
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #15181c !important;
        color: #eee;
    }
    .stButton>button {
        background: #23222a !important;
        color: #fff !important;
        font-size: 1.0em !important;
        border: 1.5px solid #39f3fa66 !important;
        border-radius: 10px !important;
        padding: 0.23em 0.1em !important;
        min-width: 32px !important;
        min-height: 32px !important;
        margin: 1px 1px 1px 1px !important;
        cursor: pointer !important;
        transition: 0.2s !important;
        box-shadow: 0 0 7px #39f3fa22 !important;
    }
    .stButton>button:hover {
        background: #2a3c44 !important;
        color: #39f3fa !important;
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
    st.session_state.pred = UltraStrongPredictor()
pred = st.session_state.pred

def push_state():
    st.session_state.stack.append(copy.deepcopy(pred))
def undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
def full_reset():
    st.session_state.pred = UltraStrongPredictor()
    st.session_state.stack.clear()

ICONS = {'P':'🔵','B':'🔴','T':'🟢'}

# 1. 버튼 한줄(가로정렬, 고유 key 부여, 작게)
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
if len(pred.pb_history) < pred.warmup_turns:
    st.markdown('<div class="neon" style="font-size:1.15em;">데이터 쌓는 중...</div>', unsafe_allow_html=True)
elif len(pred.pb_history) == pred.warmup_turns:
    if pred.next_prediction:
        st.markdown(f'<div class="neon" style="font-size:1.7em;">🎯 4번째 턴 예측 → {ICONS[pred.next_prediction]}</div>', unsafe_allow_html=True)
    st.info(f"{pred.warmup_turns}턴 데이터 쌓기 끝! 이제 4번째 턴부터 예측/적중/통계 집계 시작.")
else:
    if pred.next_prediction:
        st.markdown(f'<div class="neon" style="font-size:2.1em;">🎯 예측 → {ICONS[pred.next_prediction]}</div>', unsafe_allow_html=True)
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

# 4. 통계
s = pred.stats()
stat_cols = st.columns([1,1,1,1])
stat_cols[0].markdown(f"<div class='neon'>현재 연승<br><span style='font-size:1.08em'>{s['현재연승']}</span></div>", unsafe_allow_html=True)
stat_cols[1].markdown(f"<div class='neon-pink'>현재 연패<br><span style='font-size:1.08em'>{s['현재연패']}</span></div>", unsafe_allow_html=True)
stat_cols[2].markdown(f"<div style='color:#39f3fa'>적중률<br><span style='font-size:1.07em;font-weight:bold'>{s['적중률(%)']}%</span></div>", unsafe_allow_html=True)
stat_cols[3].markdown(f"<div style='color:#fa5252'>최대연패<br><span style='font-size:1.08em'>{s['최대연패']}</span></div>", unsafe_allow_html=True)
with st.expander("📊 전체 통계 자세히 (터치/클릭)", expanded=False):
    st.json(s)

# 타이틀
st.markdown('<div class="neon" style="font-size:1.18em; margin-top:0.7em;">AI 바카라 예측기 V7</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-pink" style="font-size:0.95em;">풀밸런스 메타+다계층+네온UI+모바일한줄버튼</div>', unsafe_allow_html=True)

if pred.next_prediction is None:
    pred.prepare_next_prediction()
