import streamlit as st
import random
import copy
from collections import defaultdict, Counter
import numpy as np

class UltraStrongPredictor:
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

        self.break_points = {
            "zigzag": 0, "몰빵": 0, "사이클": 0, "트렌드": 0,
            "2-1-2": 0, "2-2-1": 0, "1-2-1": 0, "롱런": 0,
            "박스": 0, "유사패턴": 0,
        }
        self.break_events = []
        self.hit_history = []
        self.reverse_mode = False
        self.reverse_mode_count = 0
        self.cycle_mode = False
        self.cycle_pattern = None
        self.cycle_detect_idx = None

    def _pure(self, use_pattern=None):
        idx = 0
        if use_pattern and use_pattern in self.break_points:
            idx = self.break_points[use_pattern]
        return [x for x in self.pb_history[idx:] if x in ('P','B')]

    # ------ 복잡/확장 패턴 깨짐 감지 ------
    def check_zigzag_break(self):
        pb = self._pure()
        N = len(pb)
        if N < 3: return None
        a, b, c = pb[-3:]
        if a != b and b == c:
            return len(self.pb_history)-1
        return None
    def check_molbbang_break(self):
        pb = self._pure()
        N = len(pb)
        if N < 4: return None
        if pb[-4] != pb[-3] == pb[-2] == pb[-1]:
            return len(self.pb_history)-1
        return None
    def check_cycle_break(self, cycle_len=2):
        pb = self._pure()
        N = len(pb)
        if N < cycle_len*2: return None
        last = pb[-cycle_len:]
        prev = pb[-cycle_len*2:-cycle_len]
        if last == prev and N > cycle_len*2:
            if pb[-cycle_len-1] != last[0]:
                return len(self.pb_history)-1
        return None
    def check_trend_break(self, window=6):
        pb = self._pure()
        N = len(pb)
        if N < window: return None
        sub = pb[-window:]
        cnt = Counter(sub)
        if max(cnt.values()) >= window-1:
            if N > window and pb[-window-1] != sub[0]:
                return len(self.pb_history)-1
        return None
    def check_212_break(self):
        pb = self._pure()
        if len(pb) < 5: return None
        pat = pb[-5:]
        if pat[0] == pat[1] and pat[2] != pat[0] and pat[3] == pat[4] and pat[3] == pat[0]:
            if len(pb) >= 6 and pb[-6] != pat[0]:
                return len(self.pb_history)-1
        return None
    def check_221_break(self):
        pb = self._pure()
        if len(pb) < 5: return None
        pat = pb[-5:]
        if pat[0] == pat[1] and pat[2] == pat[3] and pat[0] != pat[2] and pat[4] != pat[2]:
            if len(pb) >= 6 and pb[-6] == pat[2]:
                return len(self.pb_history)-1
        return None
    def check_121_break(self):
        pb = self._pure()
        if len(pb) < 5: return None
        pat = pb[-5:]
        if pat[0] != pat[1] and pat[1] == pat[2] and pat[2] != pat[3] and pat[3] == pat[4] and pat[1] != pat[3]:
            return len(self.pb_history)-1
        return None
    def check_longrun_break(self, run_len=6):
        pb = self._pure()
        N = len(pb)
        if N < run_len+1: return None
        last = pb[-run_len-1:-1]
        if all(x == last[0] for x in last):
            if pb[-1] != last[0]:
                return len(self.pb_history)-1
        return None
    def check_box_break(self, box_size=2):
        pb = self._pure()
        N = len(pb)
        if N < box_size*3+1: return None
        last = pb[-box_size*3-1:]
        box_pat = [last[i:i+box_size] for i in range(0, box_size*3, box_size)]
        if (box_pat[0][0]==box_pat[0][1] and
            box_pat[1][0]==box_pat[1][1] and
            box_pat[2][0]==box_pat[2][1] and
            len(set([box_pat[0][0],box_pat[1][0],box_pat[2][0]]))==2):
            if pb[-1] != box_pat[2][0]:
                return len(self.pb_history)-1
        return None
    def check_similar_pattern_break(self, window=6):
        pb = self._pure()
        N = len(pb)
        if N < window*2: return None
        last = pb[-window:]
        prev = pb[-window*2:-window]
        sim = sum([a==b for a,b in zip(last,prev)])/window
        if sim >= 0.8 and last[-1] != prev[-1]:
            return len(self.pb_history)-1
        return None

    def detect_and_update_breaks(self):
        patterns = [
            ("zigzag", self.check_zigzag_break),
            ("몰빵", self.check_molbbang_break),
            ("사이클", lambda: self.check_cycle_break(2)),
            ("트렌드", lambda: self.check_trend_break(6)),
            ("2-1-2", self.check_212_break),
            ("2-2-1", self.check_221_break),
            ("1-2-1", self.check_121_break),
            ("롱런", lambda: self.check_longrun_break(6)),
            ("박스", lambda: self.check_box_break(2)),
            ("유사패턴", lambda: self.check_similar_pattern_break(6)),
        ]
        updated = False
        for name, func in patterns:
            idx = func()
            if idx is not None and idx+1 > self.break_points[name]:
                self.break_points[name] = idx + 1
                self.break_events.append((name+" 깨짐", idx + 1))
                updated = True
        return updated

    def get_new_pattern_labels(self):
        pb_len = len(self.pb_history)
        labels = [k for k,v in self.break_points.items() if v == pb_len]
        if self.reverse_mode:
            labels = ["조작/역패턴 감지"] + labels
        if self.cycle_mode and self.cycle_pattern:
            labels = ["사이클 감지(" + "".join(self.cycle_pattern) + ")"] + labels
        return labels

    # =============== 조작/역패턴/사이클 감지 ================
    def detect_reverse_pattern(self, window=10, threshold=0.7):
        # 예측-실제 hit_history를 기반으로 '반대로만 나올 때' 감지
        if len(self.hit_history) < window: return False
        reverse_count = sum([1 for i in range(-window, 0) if self.hit_history[i] is False])
        rate = reverse_count / window
        return rate >= threshold

    def detect_cycle_pattern(self, max_cycle=6):
        pure = [x for x in self.pb_history if x in ('P','B')]
        for n in range(2, max_cycle+1):
            if len(pure) < n*2: continue
            cycle = pure[-n:]
            prev = pure[-n*2:-n]
            if cycle == prev:
                return cycle
        return None

    # =============== 예측 ===================
    def smart_predict(self):
        # 1. 역패턴(조작) 감지 시, 역베팅 모드 진입
        if self.detect_reverse_pattern(window=10, threshold=0.7):
            self.reverse_mode = True
            self.reverse_mode_count += 1
        else:
            self.reverse_mode = False
            self.reverse_mode_count = 0
        # 2. 사이클(주기 패턴) 감지 시, cycle_mode 진입
        cycle = self.detect_cycle_pattern(max_cycle=6)
        if cycle:
            self.cycle_mode = True
            self.cycle_pattern = cycle
            self.cycle_detect_idx = len(self.pb_history)
        else:
            self.cycle_mode = False
            self.cycle_pattern = None
            self.cycle_detect_idx = None

        # 우선순위: 역패턴 > 사이클 > 연패 세이프티 > 기존 앙상블
        # 1) 역패턴(조작) 감지 : 항상 내 예측의 반대로 베팅
        if self.reverse_mode and self.prev_prediction in ('P','B'):
            return 'B' if self.prev_prediction == 'P' else 'P'
        # 2) 사이클 패턴 감지
        if self.cycle_mode and self.cycle_pattern:
            idx = (len(self.pb_history)) % len(self.cycle_pattern)
            return self.cycle_pattern[idx]
        # 3) 연패 세이프티 (연패 3회 이상이면, 역베팅+랜덤+트렌드+희소 등 다수결)
        if self.current_loss >= 3:
            votes = []
            votes.append(self.predict_reverse())
            votes.append(random.choice(['P','B']))
            pure = [x for x in self.pb_history if x in ('P','B')]
            if pure:
                cnt = Counter(pure[-10:])
                rare = 'P' if cnt['P'] < cnt['B'] else 'B'
                votes.append(rare)
            if pure and len(pure)>=5:
                cnt5 = Counter(pure[-5:])
                trendrev = 'P' if cnt5['P'] < cnt5['B'] else 'B'
                votes.append(trendrev)
            if hasattr(self, 'prev_prediction') and self.prev_prediction in ('P','B'):
                votes.append('B' if self.prev_prediction == 'P' else 'P')
            best = Counter(votes).most_common()
            max_vote = best[0][1]
            candidates = [k for k,v in best if v==max_vote]
            return random.choice(candidates)
        # 4) 기존 앙상블 패턴
        preds = []
        for func in [
            self.predict_zigzag, self.predict_molbbang, self.predict_cycle,
            self.predict_trend, self.predict_212, self.predict_221, self.predict_121,
            self.predict_longrun, self.predict_box, self.predict_similar_pattern,
        ]:
            v = func()
            if v in ('P','B'):
                preds.append(v)
        if preds:
            pred = Counter(preds).most_common(1)[0][0]
            if hasattr(self, 'prev_prediction') and pred == self.prev_prediction:
                pred = 'B' if pred == 'P' else 'P'
            return pred
        return random.choice(['P','B'])

    # -------- 패턴 기반 예측 --------
    def predict_zigzag(self):
        pure = self._pure("zigzag")
        n = 4
        if len(pure) < n: return None
        last_n = pure[-n:]
        if all(last_n[i] != last_n[i-1] for i in range(1, n)):
            return 'B' if last_n[-1]=='P' else 'P'
        return None
    def predict_molbbang(self):
        pure = self._pure("몰빵")
        if len(pure) < 4: return None
        if pure[-1] == pure[-2] == pure[-3]:
            return pure[-1]
        return None
    def predict_cycle(self):
        pure = self._pure("사이클")
        maxlen = 4
        for l in range(maxlen, 1, -1):
            if len(pure) < l*2: continue
            last = pure[-l:]
            for i in range(len(pure)-l*2+1):
                if pure[i:i+l] == last:
                    idx = i+l
                    if idx < len(pure):
                        return pure[idx]
        return None
    def predict_trend(self):
        pure = self._pure("트렌드")
        if len(pure) < 6:
            return None
        cnt = Counter(pure[-6:])
        if cnt:
            return cnt.most_common(1)[0][0]
        return None
    def predict_212(self):
        pure = self._pure("2-1-2")
        if len(pure) < 5: return None
        pat = pure[-5:]
        if pat[0]==pat[1] and pat[2]!=pat[0] and pat[3]==pat[4] and pat[3]==pat[0]:
            return pat[2]
        return None
    def predict_221(self):
        pure = self._pure("2-2-1")
        if len(pure) < 5: return None
        pat = pure[-5:]
        if pat[0]==pat[1] and pat[2]==pat[3] and pat[0]!=pat[2] and pat[4]!=pat[2]:
            return pat[0]
        return None
    def predict_121(self):
        pure = self._pure("1-2-1")
        if len(pure) < 5: return None
        pat = pure[-5:]
        if pat[0]!=pat[1] and pat[1]==pat[2] and pat[2]!=pat[3] and pat[3]==pat[4] and pat[1]!=pat[3]:
            return pat[2]
        return None
    def predict_longrun(self):
        pure = self._pure("롱런")
        if len(pure) < 7: return None
        if all(x == pure[-7] for x in pure[-7:-1]) and pure[-1] != pure[-7]:
            return pure[-7]
        return None
    def predict_box(self):
        pure = self._pure("박스")
        if len(pure) < 7: return None
        last = pure[-7:]
        box_pat = [last[i:i+2] for i in range(0, 6, 2)]
        if all(p[0]==p[1] for p in box_pat):
            if box_pat[0][0]!=box_pat[1][0] and box_pat[1][0]!=box_pat[2][0]:
                return box_pat[1][0]
        return None
    def predict_similar_pattern(self):
        pure = self._pure("유사패턴")
        if len(pure) < 12: return None
        last = pure[-6:]
        prev = pure[-12:-6]
        sim = sum([a==b for a,b in zip(last,prev)])/6
        if sim >= 0.8:
            return last[-1]
        return None
    def predict_reverse(self):
        pure = [x for x in self.pb_history if x in ('P','B')]
        if not pure: return random.choice(['P','B'])
        return 'B' if pure[-1] == 'P' else 'P'

    def predict(self):
        return self.smart_predict()

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
        self.detect_and_update_breaks()
        self.prepare_next_prediction()

    def prepare_next_prediction(self):
        self.next_prediction = self.predict()

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
st.set_page_config(layout="wide", page_title="AI 역패턴/사이클 탐지 예측기", page_icon="🦾")
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

# 2. 예측 결과 (NEW 패턴 라벨 표시)
if len(pred.pb_history) < pred.warmup_turns:
    st.markdown('<div class="neon" style="font-size:1.15em;">데이터 쌓는 중...</div>', unsafe_allow_html=True)
elif len(pred.pb_history) == pred.warmup_turns:
    if pred.next_prediction:
        st.markdown(f'<div class="neon" style="font-size:1.7em;">🎯 4번째 턴 예측 → {ICONS[pred.next_prediction]}</div>', unsafe_allow_html=True)
    st.info(f"{pred.warmup_turns}턴 데이터 쌓기 끝! 이제 4번째 턴부터 예측/적중/통계 집계 시작.")
else:
    new_labels = pred.get_new_pattern_labels()
    label_html = ""
    if new_labels:
        label_html = "<span style='color:#ffb347; font-size:1em; font-weight:bold;'> [NEW: " + " | ".join(new_labels) + "]</span>"
    if pred.next_prediction:
        st.markdown(
            f'<div class="neon" style="font-size:2.1em;">🎯 예측 → {ICONS[pred.next_prediction]} {label_html}</div>',
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

# 5. 패턴별 깨짐 경고 및 이력
if pred.break_events:
    st.markdown('<div class="neon-pink" style="font-size:1.04em;">🚨 패턴 깨짐 경고/슬라이싱!</div>', unsafe_allow_html=True)
    last_breaks = pred.break_events[-3:]
    for name, idx in last_breaks:
        st.warning(f"🔔 {name} 발생 - 이 시점부터 해당 패턴은 새로 분석! (#{idx}턴 이후)")
    with st.expander("전체 깨짐 이력 (터치/클릭)", expanded=False):
        for name, idx in pred.break_events:
            st.write(f"{name} | 위치: {idx}번째 입력 이후 새로 분석")

st.markdown('<div class="neon" style="font-size:1.18em; margin-top:0.7em;">AI 역패턴·사이클·세이프티 ALL-IN 예측기</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-pink" style="font-size:0.95em;">조작·역패턴·사이클·복잡패턴 자동감지/슬라이싱/연패방지</div>', unsafe_allow_html=True)

if pred.next_prediction is None:
    pred.prepare_next_prediction()
