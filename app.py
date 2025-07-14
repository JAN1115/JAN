import streamlit as st
import random, copy, time
from collections import Counter, defaultdict

# --- 알고리즘 함수 ---
def expert_ngram(pb_history, n=3):
    pure = [x for x in pb_history if x in 'PB']
    if len(pure) >= n:
        key = tuple(pure[-n:])
        c = Counter()
        for i in range(len(pure)-n):
            if tuple(pure[i:i+n]) == key:
                c[pure[i+n]] += 1
        if c:
            return c.most_common(1)[0][0]
    return random.choice(['P', 'B'])

def expert_longrun(pb_history, length=4):
    pure = [x for x in pb_history if x in 'PB']
    if len(pure) < length: return random.choice(['P', 'B'])
    seq = pure[-length:]
    if all(x == seq[0] for x in seq):
        return seq[0]
    return random.choice(['P', 'B'])

def expert_china(pb_history):
    def get_4_china_scores(pb_history):
        def calc_big_road(pb_history):
            grid = defaultdict(lambda: defaultdict(str))
            row = col = 0
            prev = None
            drop_col = 0
            for r in pb_history:
                if r not in 'PB': continue
                if prev is None or r == prev:
                    if prev is not None: row += 1
                    grid[row][col] = r
                else:
                    drop_col += 1
                    col = drop_col
                    row = 0
                    grid[row][col] = r
                prev = r
            return grid
        def extract_col(grid, col_idx):
            return [grid[row].get(col_idx,'') for row in range(50)]
        def calc_china_roads(bigroad, mode):
            score = 0
            col_offset = {1:1, 2:2, 3:3}[mode]
            for col in range(col_offset, 100):
                cur_col = extract_col(bigroad, col)
                ref_col = extract_col(bigroad, col-col_offset)
                if not cur_col[0]: break
                if len([x for x in cur_col if x]) == len([x for x in ref_col if x]):
                    score += 1
                else:
                    score -= 1
            return score
        bigroad = calc_big_road(pb_history)
        scores = {}
        streaks = 0
        last = None
        for r in pb_history:
            if r == last: streaks += 1
            last = r
        scores['빅로드'] = streaks
        scores['빅아이'] = calc_china_roads(bigroad, 1)
        scores['스몰로드'] = calc_china_roads(bigroad, 2)
        scores['바퀴'] = calc_china_roads(bigroad, 3)
        return scores
    scores = get_4_china_scores(pb_history)
    preds = []
    for name, score in scores.items():
        preds.append('P' if score % 2 == 0 else 'B')
    return Counter(preds).most_common(1)[0][0] if preds else random.choice(['P', 'B'])

def expert_side_6(pb_history):
    pure = [x for x in pb_history if x in 'PB']
    # 6매 옆줄 비교: 6턴전 값과 같으면 그 값
    if len(pure) >= 6:
        return pure[-6]
    return random.choice(['P', 'B'])

# --- 메인 보조/순환형 예측기 ---
class HybridCyclePredictor:
    def __init__(self):
        self.history = []
        self.pb_history = []
        self.correct = 0
        self.incorrect = 0
        self.current_win = 0
        self.current_loss = 0
        self.max_win = 0
        self.max_loss = 0
        self.prev_prediction = None
        self.next_prediction = None
        self.fail_count = 0
        self.last_alg = '-'
        self.last_subs = []
        self.all_losses = []
        self.all_wins = []
        self.china_scores = {'빅로드':0,'빅아이':0,'스몰로드':0,'바퀴':0}
        self.round = 0

    def get_next_prediction(self):
        fc = self.fail_count
        idx = fc % 6  # 0~2(ngram), 3~5(longrun), 6→0
        subs = []
        if idx < 3:
            main = expert_ngram(self.pb_history, 3)
            subs = [expert_china(self.pb_history), expert_side_6(self.pb_history)]
            label = "ngram3"
        else:
            main = expert_longrun(self.pb_history, 4)
            subs = [expert_china(self.pb_history), expert_side_6(self.pb_history)]
            label = "longrun4"
        # 투표 (3개 중 다수, 없으면 main)
        votes = [main] + subs
        pred = Counter(votes).most_common(1)[0][0] if Counter(votes).most_common(1)[0][1] >=2 else main
        self.last_alg = label
        self.last_subs = subs
        return pred

    def handle_input(self, r):
        self.history.append(r)
        self.round += 1
        if r == 'T':  # 타이는 무효
            self.prev_prediction = self.next_prediction
            self.prepare_next_prediction()
            return
        self.pb_history.append(r)
        self.prev_prediction = self.next_prediction
        if len(self.pb_history) > 6 and self.prev_prediction in 'PB' and r in 'PB':
            hit = (self.prev_prediction == r)
            if hit:
                self.correct += 1
                self.all_losses.append(self.current_loss)
                self.all_wins.append(self.current_win)
                self.current_win += 1
                self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
                self.fail_count = 0
            else:
                self.incorrect += 1
                self.all_wins.append(self.current_win)
                self.current_win = 0
                self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
                self.fail_count += 1
        self.prepare_next_prediction()

    def prepare_next_prediction(self):
        self.next_prediction = self.get_next_prediction()
        self.china_scores = expert_china_scores(self.pb_history)

    def stats(self):
        total_pb = len([x for x in self.pb_history if x in 'PB']
                      ) if len(self.pb_history) > 6 else 0
        hitrate = round(self.correct / total_pb * 100, 2) if total_pb else 0
        avg_loss = round(sum(self.all_losses)/len(self.all_losses),2) if self.all_losses else 0
        avg_win = round(sum(self.all_wins)/len(self.all_wins),2) if self.all_wins else 0
        return {
            '총입력': len(self.history),
            '적중률(%)': hitrate,
            '현재연승': self.current_win,
            '현재연패': self.current_loss,
            '최대연승': self.max_win,
            '최대연패': self.max_loss,
            '평균연패': avg_loss,
            '평균연승': avg_win,
            '알고리즘': self.last_alg,
            '보조1(중국점)': self.last_subs[0] if self.last_subs else '-',
            '보조2(6매옆)': self.last_subs[1] if self.last_subs else '-'
        }

def expert_china_scores(pb_history):
    # 4중국점 점수
    def get_4_china_scores(pb_history):
        def calc_big_road(pb_history):
            grid = defaultdict(lambda: defaultdict(str))
            row = col = 0
            prev = None
            drop_col = 0
            for r in pb_history:
                if r not in 'PB': continue
                if prev is None or r == prev:
                    if prev is not None: row += 1
                    grid[row][col] = r
                else:
                    drop_col += 1
                    col = drop_col
                    row = 0
                    grid[row][col] = r
                prev = r
            return grid
        def extract_col(grid, col_idx):
            return [grid[row].get(col_idx,'') for row in range(50)]
        def calc_china_roads(bigroad, mode):
            score = 0
            col_offset = {1:1, 2:2, 3:3}[mode]
            for col in range(col_offset, 100):
                cur_col = extract_col(bigroad, col)
                ref_col = extract_col(bigroad, col-col_offset)
                if not cur_col[0]: break
                if len([x for x in cur_col if x]) == len([x for x in ref_col if x]):
                    score += 1
                else:
                    score -= 1
            return score
        bigroad = calc_big_road(pb_history)
        scores = {}
        streaks = 0
        last = None
        for r in pb_history:
            if r == last: streaks += 1
            last = r
        scores['빅로드'] = streaks
        scores['빅아이'] = calc_china_roads(bigroad, 1)
        scores['스몰로드'] = calc_china_roads(bigroad, 2)
        scores['바퀴'] = calc_china_roads(bigroad, 3)
        return scores
    return get_4_china_scores(pb_history)

# ========== UI / 네온/6매/통계/애니메이션 ==========

st.set_page_config(layout="wide", page_title="UltraSafetyAI Hybrid+Cycle", page_icon="🧠")
st.markdown("""
<style>
html, body { background: #090d14 !important; color: #f9f9fa !important;}
[data-testid="stAppViewContainer"] {background: #090d14;}
.stButton > button {
    border-radius: 13px !important; border: 2.2px solid #39f3fa !important;
    font-weight: bold; font-size: 1.11em;
    margin: 5px 6px 5px 0; padding:10px 0 9px 0; width: 100%; min-width:120px;
    box-shadow:0 0 12px #39f3fa5a,0 0 8px #28b3ff42;
    transition: transform 0.10s;
}
.stButton > button:active { transform: scale(1.08);}
.stButton > button.pbtn {background:linear-gradient(90deg,#0d233b,#174aaa 95%); color:#52e3ff !important; border-color:#28f6fc;}
.stButton > button.bbtn {background:linear-gradient(90deg,#350a13,#d3244c 80%); color:#ff637a !important; border-color:#fd3358;}
.stButton > button.tbtn {background:linear-gradient(90deg,#123a13,#24b364 90%); color:#95ffb4 !important; border-color:#1be48d;}
.stButton > button:hover {filter:brightness(1.12); border:2.5px solid #fff;}
.neon {color: #00ffe7; text-shadow:0 0 12px #39f3fa,0 0 14px #28b3ff; font-weight: bold;}
.sixgrid {font-size: 1.16em; letter-spacing: 2.05px; line-height: 1.19em; display:inline-block;}
.china-scores-wrap {display:flex; gap:15px; flex-wrap:wrap; margin-bottom:3px; margin-top:7px;}
.china-label {
  font-size:1.19em; color:#fff; font-weight:800; padding:2px 15px 2px 15px;
  border-radius:11px 11px 5px 5px; margin-right:6px;
  background:linear-gradient(90deg,#292759 40%,#1ad1fc 96%);
  letter-spacing:0.13em; text-shadow:0 0 10px #7cf8fc,0 0 5px #fff;
  border-bottom:4px solid #00f6f9; box-shadow:0 0 11px #25f0e399, 0 0 0 #fff0; display:inline-block;
}
.china-score {
  font-size:1.13em; font-weight:800; padding:3px 11px; margin-right:5px;
  background:linear-gradient(90deg,#1d232a,#1cd1e8 88%);
  border-radius:9px 9px 14px 14px; color:#00ffe7;
  border:2px solid #3ee1fd; box-shadow:0 0 7px #47f9ffb9; display:inline-block;
}
.stats-grid {display:grid;grid-template-columns: repeat(2,1fr);gap:9px;}
@keyframes glow {
  0% {box-shadow:0 0 11px #5dfcff,0 0 0 #fff0;}
  40% {box-shadow:0 0 28px #ff459f,0 0 15px #fff0;}
  100% {box-shadow:0 0 11px #5dfcff,0 0 0 #fff0;}
}
.glow-anim {animation:glow 1.25s infinite;}
@keyframes bounceIn {
  0% {transform: scale(0.85);}
  50% {transform: scale(1.17);}
  100% {transform: scale(1);}
}
.bounce-anim {animation:bounceIn 0.46s;}
</style>
""", unsafe_allow_html=True)

if 'stack' not in st.session_state: st.session_state.stack = []
if 'pred' not in st.session_state: st.session_state.pred = HybridCyclePredictor()
if 'clicked' not in st.session_state: st.session_state.clicked = 0
pred = st.session_state.pred

def push_state():
    st.session_state.stack.append(copy.deepcopy(pred))
    st.session_state.clicked = int(time.time()*1000)
def undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
        st.session_state.clicked = int(time.time()*1000)
def full_reset():
    st.session_state.pred = HybridCyclePredictor()
    st.session_state.stack.clear()
    st.session_state.clicked = int(time.time()*1000)

ICONS = {'P':'🔵','B':'🔴','T':'🟢'}

btncols = st.columns([1.1,1.1,1,0.9,0.9])
with btncols[0]:
    if st.button("🔵 플레이어(P)", key='pbtn', use_container_width=True):
        push_state(); pred.handle_input("P")
with btncols[1]:
    if st.button("🔴 뱅커(B)", key='bbtn', use_container_width=True):
        push_state(); pred.handle_input("B")
with btncols[2]:
    if st.button("🟢 타이(T)", key='tbtn', use_container_width=True):
        push_state(); pred.handle_input("T")
with btncols[3]:
    if st.button("↩️ 되돌리기"):
        undo()
with btncols[4]:
    if st.button("🗑️ 전체 초기화"):
        full_reset()

# 예측 결과/애니메이션/6턴 이하 안내
if len(pred.pb_history) < 6:
    st.markdown(
        '<div class="neon bounce-anim" style="font-size:1.20em;">🔎 데이터 수집 중입니다...<br><span style="font-size:0.95em;">(6턴까지 예측·통계 미노출)</span></div>',
        unsafe_allow_html=True)
elif pred.next_prediction:
    st.markdown(
        f'<div class="neon glow-anim" style="font-size:1.64em;margin-top:2px;">🎯 <span style="font-size:1.17em;margin-left:10px;">다음 예측 → {ICONS.get(pred.next_prediction,"-")}</span></div>',
        unsafe_allow_html=True
    )

# 6매 시각화 + 애니메이션
st.markdown('<div class="neon" style="font-size:1.07em;margin-top:10px;">6매 기록</div>', unsafe_allow_html=True)
history = pred.history
max_row = 6
ncols = (len(history) + max_row - 1) // max_row
six_grid = []
last_idx = len(history)-1
six_html = '<div class="sixgrid">'
for r in range(max_row):
    six_html += "<div>"
    for c in range(ncols):
        idx = c * max_row + r
        cell_class = "sixcell"
        if idx == last_idx and len(history) >= 6:
            cell_class += " bounce-anim"
        if idx < len(history):
            six_html += f"<span class='{cell_class}'>{ICONS.get(history[idx], '　')}</span>"
        else:
            six_html += "<span class='sixcell'>　</span>"
    six_html += "</div>"
six_html += '</div>'
st.markdown(six_html, unsafe_allow_html=True)

# 통계/중국점/알고리즘 정보
if len(pred.pb_history) >= 6:
    s = pred.stats()
    stats_html = f"""
    <div class="stats-grid" style="margin-bottom:2px;">
        <div class='neon'>총입력<br><span style='font-size:1.14em'>{s['총입력']}</span></div>
        <div class='neon'>적중률<br><span style='font-size:1.16em;'>{s['적중률(%)']}%</span></div>
        <div class='neon'>현재 연승<br><span style='font-size:1.12em;'>{s['현재연승']}</span></div>
        <div class='neon'>현재 연패<br><span style='font-size:1.12em;'>{s['현재연패']}</span></div>
        <div class='neon'>최대 연승<br><span style='font-size:1.12em;'>{s['최대연승']}</span></div>
        <div class='neon'>최대 연패<br><span style='font-size:1.12em;'>{s['최대연패']}</span></div>
        <div class='neon'>평균 연패<br><span style='font-size:1.12em;'>{s['평균연패']}</span></div>
        <div class='neon'>평균 연승<br><span style='font-size:1.12em;'>{s['평균연승']}</span></div>
        <div class='neon'>AI알고리즘<br><span style='font-size:0.98em;'>{s['알고리즘']}</span></div>
        <div class='neon'>보조1(중국점)<br><span style='font-size:1.08em;'>{s['보조1(중국점)']}</span></div>
        <div class='neon'>보조2(6매옆)<br><span style='font-size:1.08em;'>{s['보조2(6매옆)']}</span></div>
    </div>
    <div style="display:flex;gap:13px;flex-wrap:wrap;margin:11px 0 8px 0;">
      <div class="china-label">빅로드</div>
      <div class="china-score">{pred.china_scores['빅로드']}</div>
      <div class="china-label">빅아이</div>
      <div class="china-score">{pred.china_scores['빅아이']}</div>
      <div class="china-label">스몰</div>
      <div class="china-score">{pred.china_scores['스몰로드']}</div>
      <div class="china-label">바퀴</div>
      <div class="china-score">{pred.china_scores['바퀴']}</div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

if st.button("🐞 버그 리포트 복사"):
    st.code(f"입력 기록: {pred.history}\n예측값: {pred.next_prediction}\n중국점: {pred.china_scores}\n통계: {pred.stats()}")

st.markdown(
    '<div style="margin-top:13px;font-size:0.98em;color:#ff64b3;font-weight:bold;text-align:right;">'
    'UltraSafetyAI™ | Hybrid+Cycle+보조 | by ChatGPT v202507</div>',
    unsafe_allow_html=True
)
