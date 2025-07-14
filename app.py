import streamlit as st
import random, copy, time
from collections import Counter, defaultdict

# ---------- 알고리즘 & 3턴 예측 & 스마트 진입 타이밍 ----------

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

def meta_ensemble_predict(pb_history, loglist=None):
    pure = [x for x in pb_history if x in 'PB']
    pred_ngram, pred_trend, pred_china, pred_simul, pred_combo = None, None, None, None, None
    ngram_type = None
    # n-gram(3)
    if len(pure) >= 3:
        key = tuple(pure[-3:])
        c = Counter()
        for i in range(len(pure)-3):
            if tuple(pure[i:i+3]) == key:
                c[pure[i+3]] += 1
        if c:
            pred_ngram = c.most_common(1)[0][0]
            ngram_type = "ngram(3)"
    if not pred_ngram:
        pred_ngram = random.choice(['P','B'])
        ngram_type = "random(ngram)"
    # trend
    if len(pure) < 4:
        pred_trend = random.choice(['P','B'])
        trend_type = "random(trend)"
    else:
        cnt = Counter(pure[-4:])
        if cnt['P'] > cnt['B']:
            pred_trend = 'P'
        elif cnt['B'] > cnt['P']:
            pred_trend = 'B'
        else:
            pred_trend = random.choice(['P','B'])
        trend_type = "trend"
    # 중국점
    scores = get_4_china_scores(pb_history)
    preds = []
    for score in scores.values():
        preds.append('P' if score % 2 == 0 else 'B')
    pred_china = Counter(preds).most_common(1)[0][0] if preds else random.choice(['P','B'])
    # 샘플링
    base = pure[-4:] if len(pure) >= 4 else pure
    nextvals = []
    for _ in range(90):
        seq = list(base)
        for _ in range(5):
            if random.random() < 0.21:
                val = random.choice(['P','B'])
                run = random.randint(3, 6)
                seq.extend([val]*run)
            else:
                seq.append(random.choice(['P','B']))
        if len(seq) > len(base):
            nextvals.append(seq[len(base)])
    if not nextvals:
        pred_simul = random.choice(['P','B'])
    else:
        pred_simul = Counter(nextvals).most_common(1)[0][0]
    # 복합 패턴 (간단 예시)
    if len(pure) >= 6:
        pattern = "".join(pure[-6:])
        if pattern in ["PPPPPP","BBBBBB"]:
            pred_combo = pure[-1]
        elif pattern in ["PBPBPB","BPBPBP"]:
            pred_combo = pure[-2]
        else:
            pred_combo = random.choice(['P','B'])
    else:
        pred_combo = random.choice(['P','B'])
    # 투표
    preds_list = [pred_ngram, pred_trend, pred_china, pred_simul, pred_combo]
    counter = Counter(preds_list)
    result = counter.most_common(1)[0][0]
    algolog = (f"[n-gram:{pred_ngram}({ngram_type}), trend:{pred_trend}, "
               f"중국점:{pred_china}, 샘플링:{pred_simul}, 복합:{pred_combo}] → 최종:{result}")
    if loglist is not None:
        loglist.append(algolog)
    return result

def predict_3turn(pb_history, loglist=None):
    l = loglist if loglist is not None else []
    pred1 = meta_ensemble_predict(pb_history, l)
    ph2 = pb_history + [pred1]
    pred2 = meta_ensemble_predict(ph2, l)
    ph3 = pb_history + [pred1, pred2]
    pred3 = meta_ensemble_predict(ph3, l)
    return [pred1, pred2, pred3]

# ---- 스마트 진입 타이밍 AI (진입대기 중에도 실시간 트래킹)
def ai_should_enter_now(pb_history, analysis_trail):
    pure = [x for x in pb_history if x in 'PB']
    if len(pure) < 8:
        analysis_trail.append("AI: 데이터 부족, 진입")
        return True
    last6 = pure[-6:]
    scores = get_4_china_scores(pb_history)
    if last6 in (['P','B']*3, ['B','P']*3):
        analysis_trail.append("AI: 최근 지그재그, 빠른 진입")
        return True
    if last6.count('P') >= 5 or last6.count('B') >= 5:
        analysis_trail.append("AI: 최근 롱런/트렌드 감지, 빠른 진입")
        return True
    diff = abs(scores['빅로드']) + abs(scores['빅아이']) + abs(scores['스몰로드']) + abs(scores['바퀴'])
    if diff > 18:
        analysis_trail.append("AI: 중국점 점수 급변, 조기 진입")
        return True
    cnt8 = Counter(pure[-8:])
    if abs(cnt8['P'] - cnt8['B']) <= 1:
        analysis_trail.append("AI: 최근 8턴 균형, 진입 지연")
        return False
    analysis_trail.append("AI: 기본 진입")
    return True

class MetaChinaSimulAI_3T_SmartEntry:
    def __init__(self):
        self.history = []
        self.pb_history = []
        self.bet_count = 0
        self.correct = 0
        self.incorrect = 0
        self.current_win = 0
        self.current_loss = 0
        self.max_win = 0
        self.max_loss = 0
        self.popup_trigger = False
        self.popup_type = None
        self.popup_time = 0
        self.ongoing_3turn = False
        self.pred_3turn = None
        self.idx_3turn = 0
        self.skip_count = 0
        self.algo_logs = []
        self.analysis_trail = []
        self.last_hit_indices = []
        self.last_feedback = ""      # 알림 메시지용
        self.feedback_time = 0       # 알림 발생 시간
        self.hit_record = []         # 적중 O, 미적중 X 기록

    def handle_input(self, r):
        now = time.time()
        self.last_feedback = ""
        if r == 'T':
            self.history.append('T')
            self.pb_history.append('T')
            self.popup_trigger = False
            self.last_hit_indices = []
            self.hit_record.append(None)
            return

        self.history.append(r)
        self.pb_history.append(r)
        self.popup_trigger = False
        self.last_hit_indices = []

        # 스킵 카운트(3턴 미적중→3턴 스킵) 처리
        if self.skip_count > 0:
            self.skip_count -= 1
            self.ongoing_3turn = False
            self.pred_3turn = None
            self.idx_3turn = 0
            self.hit_record.append(None)
            self.analysis_trail.append("AI: 3턴 스킵(미적중)")
            return

        # 진입대기 중에도 실시간 패턴 감지/분석 트래킹 (진입조건 AI 판단)
        if not self.ongoing_3turn:
            pb_count = len([x for x in self.pb_history if x in 'PB'])
            if pb_count < 6:
                self.pred_3turn = None
                self.hit_record.append(None)
                self.analysis_trail.append("AI: 데이터 부족")
                return
            should_enter = ai_should_enter_now(self.pb_history, self.analysis_trail)
            if should_enter:
                self.pred_3turn = predict_3turn(self.pb_history, self.algo_logs)
                self.idx_3turn = 0
                self.ongoing_3turn = True
                self.hit_record.append(None)
                return
            else:
                self.pred_3turn = None
                self.hit_record.append(None)
                return

        if self.ongoing_3turn and self.pred_3turn:
            pred = self.pred_3turn[self.idx_3turn]
            self.bet_count += 1
            idx = len(self.history)-1
            if pred == r:
                self.correct += 1
                self.current_win += 1
                self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
                self.popup_trigger = True
                self.popup_type = "hit"
                self.popup_time = time.time()
                self.skip_count = 2
                self.ongoing_3turn = False
                self.pred_3turn = None
                self.idx_3turn = 0
                self.last_hit_indices = [idx]
                self.hit_record.append("O")
                self.analysis_trail.append("AI: 적중 → 3턴 스킵")
                return
            else:
                self.incorrect += 1
                self.current_win = 0
                self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
                self.popup_trigger = True
                self.popup_type = "miss"
                self.popup_time = time.time()
                self.idx_3turn += 1
                self.last_hit_indices = []
                self.hit_record.append("X")
                if self.idx_3turn == 3:
                    self.ongoing_3turn = False
                    self.pred_3turn = None
                    self.idx_3turn = 0
                    self.skip_count = 3
                    self.analysis_trail.append("AI: 3턴 미적중 → 3턴 스킵")
        else:
            self.popup_trigger = False
            self.last_hit_indices = []
            self.hit_record.append(None)

    def stats(self):
        hitrate = round(self.correct / self.bet_count * 100, 2) if self.bet_count else 0
        return {
            '총입력': len(self.history),
            '적중률(%)': hitrate,
            '현재연승': self.current_win,
            '현재연패': self.current_loss,
            '최대연승': self.max_win,
            '최대연패': self.max_loss
        }

    def get_next_pred(self):
        if self.ongoing_3turn and self.pred_3turn:
            idx = self.idx_3turn
            if idx < 3:
                p = self.pred_3turn[idx]
                return p, idx+1
        return None, None

    def set_feedback(self, msg):
        self.last_feedback = msg
        self.feedback_time = time.time()

    def get_feedback(self):
        if self.last_feedback and (time.time() - self.feedback_time) < 2.5:
            return self.last_feedback
        else:
            self.last_feedback = ""
            return ""

# ========== UI 파트 ==========

if 'stack' not in st.session_state: st.session_state.stack = []
if 'pred' not in st.session_state: st.session_state.pred = MetaChinaSimulAI_3T_SmartEntry()
pred = st.session_state.pred

if 'show_all_trail' not in st.session_state: st.session_state.show_all_trail = False

# ----------- 맨 위 해설 한줄 + 전체 해설 버튼 ----------
one_line = ""
if hasattr(pred, "analysis_trail") and pred.analysis_trail:
    one_line = pred.analysis_trail[-1]
if not one_line:
    one_line = "진입판단 해설 준비중..."

trail_cols = st.columns([7, 1.7])
with trail_cols[0]:
    st.markdown(
        f"<div style='font-size:0.93em;color:#b2ffed;background:#10232d;padding:6px 12px 3px 12px;margin-bottom:4px;border-radius:9px;display:inline-block;text-shadow:0 0 5px #4affc7;letter-spacing:0.01em;max-width:94vw;overflow-x:auto;'>"
        f"🧠 <b>진입 AI 해설</b>: {one_line}</div>", unsafe_allow_html=True)
with trail_cols[1]:
    if st.button("전체 해설 보기", key="trail_toggle"):
        st.session_state.show_all_trail = not st.session_state.show_all_trail

# ... (아래는 UI/스타일/버튼/예측 결과 등 기존 코드 그대로)

# ---------- 네온/다크/펄스/애니 스타일 ----------
st.set_page_config(layout="wide", page_title="짜라라잔", page_icon="⚡️")
st.markdown("""
<style>
html, body { background: #0c111b !important; color: #e0fcff !important;}
[data-testid="stAppViewContainer"] {background: #0c111b !important;}
.hourglass-ani {
  display:inline-block;
  animation: hourglass-spin 1.25s linear infinite;
  font-size:1.5em;
  margin-right: 7px;
}
@keyframes hourglass-spin {
  0% {transform:rotate(0deg);}
  100% {transform:rotate(360deg);}
}
@keyframes neon-glow {
  0%, 100% { filter: brightness(1.35) drop-shadow(0 0 18px #80ffd7);}
  50% { filter: brightness(2.0) drop-shadow(0 0 40px #3fffbb);}
}
.next-pred-ani {
  animation: neon-glow 1.07s ease-in-out;
  font-size:2.05em;
  font-weight:bold;
  background:linear-gradient(120deg,#132227 50%,#183940 100%);
  color:#00fffa !important;
  border-radius:27px;
  padding: 15px 40px;
  text-shadow:0 0 22px #53ffcd, 0 0 7px #fff;
  letter-spacing:0.10em;
  display:inline-block;
  margin:0 auto;
  border:2.5px solid #7afcd2;
  box-shadow:0 0 38px #42ffd1;
}
.sixgrid-fluo {
  color:#c8ff7d !important;
  text-shadow:0 0 21px #c8ff7d,0 0 24px #f7ffb3,0 0 8px #fff;
  background:rgba(150,255,110,0.16);
  border-radius:50%;
  font-weight:bold;
  padding:1.5px 7px;
  box-shadow:0 0 13px #e2ffa0, 0 0 20px #cffda1 inset;
}
.sixgrid-miss {
  color:#ff5e7a !important;
  text-shadow:0 0 13px #ff5e7a,0 0 13px #ffd7d7,0 0 7px #fff;
  background:rgba(255,70,110,0.17);
  border-radius:50%;
  font-weight:bold;
  padding:1.5px 7px;
  box-shadow:0 0 11px #ff8db1, 0 0 13px #ffb8b8 inset;
}
.sixgrid-title {
    font-size:1.13em;
    font-weight:bold;
    color:#b2fd4b;
    letter-spacing:0.20em;
    text-shadow:0 0 11px #e9fe95,0 0 6px #fff;
    margin-bottom:2px;
    display:flex;align-items:center;gap:6px;
}
.sixgrid { font-size:1.05em; letter-spacing: 1.18px;}
.stats-box {
    font-size:1.09em;
    font-weight:bold;
    padding:11px 0 7px 0;
    margin-bottom:3px;
    border-radius:15px;
    box-shadow:0 0 19px #52fff7a2;
    background:linear-gradient(120deg,#0a1823 70%,#223346 100%);
    text-align:center;
    letter-spacing:1.03px;
    border:2px solid #37ffea;
    color: #fff !important;
    text-shadow:0 0 8px #00f7ff60, 0 0 11px #fff5, 0 0 5px #fff;
}
.stats-title {
    font-size:1.13em;
    letter-spacing:0.04em;
    color:#3ffe8d !important;
    font-weight:900;
    text-shadow:0 0 8px #1cffea, 0 0 3px #fff;
}
.toast-mini {
    position: fixed; bottom: 28px; right: 24px; z-index: 1500;
    background: #111a33;
    color: #fffbe0;
    border-radius: 13px;
    box-shadow:0 0 19px #d0f1ff60;
    padding: 15px 27px 12px 17px;
    font-size: 1.16em;
    font-weight: 700;
    opacity: 0.97;
    animation: neon-glow 1.1s infinite alternate;
    border:2.5px solid #40f4b1;
}
@media (max-width:600px) {
  .popup-center, .popup-center-miss {font-size:2em;}
  .sixgrid-title {font-size:0.92em;}
  .sixgrid-ani {font-size:0.79em;}
  .stats-box {font-size:1.03em;}
  .toast-mini {font-size:0.91em;padding:9px 13px 7px 13px;right:7px;bottom:10px;}
}
</style>
""", unsafe_allow_html=True)

def push_state(): st.session_state.stack.append(copy.deepcopy(pred))
def undo():
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
        st.session_state.pred.set_feedback("되돌리기 완료!")
def full_reset():
    st.session_state.pred = MetaChinaSimulAI_3T_SmartEntry()
    st.session_state.stack.clear()
    st.session_state.pred.set_feedback("전체 초기화 완료!")

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

def dots_anim():
    t = int(time.time()*1.9) % 3 + 1
    return "."*t + " "*(3-t)

npred, nidx = pred.get_next_pred()
if pred.ongoing_3turn and npred is not None:
    picon = ICONS[npred]
    st.markdown(
        f'<div style="display:flex;justify-content:center;margin:15px 0 7px 0;">'
        f'<span class="next-pred-ani">🔮 NEXT {picon} <span style="font-size:0.52em;opacity:0.62;">({nidx}/3)</span></span>'
        f'</div>',
        unsafe_allow_html=True
    )
elif not pred.ongoing_3turn and len([x for x in pred.pb_history if x in "PB"]) < 6:
    st.markdown(
        f"""<div style="display:flex;justify-content:center;align-items:center;">
        <div style="background:#2b2900;border-radius:16px;border:2.2px solid #ffd400;padding:10px 32px;margin-top:30px;box-shadow:0 0 22px #fffbb4,0 0 7px #ffd400;">
            <span class="hourglass-ani">⏳</span>
            <span style='font-size:1.22em;font-weight:900;color:#ffd400;text-shadow:0 0 13px #fffbb4,0 0 7px #111,0 0 8px #fff;letter-spacing:0.08em;'>
            예측 준비중{dots_anim()}</span>
        </div></div>""",
        unsafe_allow_html=True)
else:
    st.markdown(
        f"""<div style="display:flex;justify-content:center;align-items:center;">
        <div style="background:#2b2900;border-radius:16px;border:2.2px solid #ffd400;padding:10px 32px;margin-top:30px;box-shadow:0 0 22px #fffbb4,0 0 7px #ffd400;">
            <span class="hourglass-ani">⏳</span>
            <span style='font-size:1.22em;font-weight:900;color:#ffd400;text-shadow:0 0 13px #fffbb4,0 0 7px #111,0 0 8px #fff;letter-spacing:0.08em;'>
            진입 대기{dots_anim()}</span>
        </div></div>""",
        unsafe_allow_html=True)

# ---- 적중/미적중 팝업(토스트, 오른쪽 아래)
if pred.popup_trigger and pred.popup_type:
    if 'popup_time' not in st.session_state: st.session_state['popup_time'] = 0
    if st.session_state['popup_time'] < time.time():
        st.session_state['popup_time'] = time.time() + 1.0
    if pred.popup_type == "hit":
        st.markdown('<div class="toast-mini" style="background:#0e2b17;color:#d6ffcf;font-size:1.18em;border:2.7px solid #7dffad;font-weight:800;text-shadow:0 0 8px #77ffb1,0 0 3px #fff;">🎉 적중!</div>', unsafe_allow_html=True)
    elif pred.popup_type == "miss":
        st.markdown('<div class="toast-mini" style="background:#3b1823;color:#ffe6ea;font-size:1.18em;border:2.7px solid #ff5c6e;font-weight:800;text-shadow:0 0 9px #ff99ba,0 0 4px #fff;">💥 미적중!</div>', unsafe_allow_html=True)
    if time.time() > st.session_state['popup_time']:
        pred.popup_trigger = False
        pred.popup_type = None

# ---- 6매 기록: 일직선, 형광/빨강, 크기 조정, 정렬 개선
st.markdown('<div class="sixgrid-title">🪧 <b>6매 기록</b></div>', unsafe_allow_html=True)
history = pred.history
hitrec = pred.hit_record
max_row = 6
ncols = (len(history) + max_row - 1) // max_row
six_html = '<div class="sixgrid" style="display:inline-block;"><table style="border-spacing:4px 2px;"><tbody>'
for r in range(max_row):
    six_html += "<tr>"
    for c in range(ncols):
        idx = c * max_row + r
        if idx < len(history):
            hit = (hitrec[idx]=="O")
            miss = (hitrec[idx]=="X")
            val = {'P':'🔵','B':'🔴','T':'🟢'}.get(history[idx],"　")
            if hit:
                cell = f"<span class='sixgrid-fluo'>{val}</span>"
            elif miss:
                cell = f"<span class='sixgrid-miss'>{val}</span>"
            else:
                cell = f"<span style='opacity:0.88'>{val}</span>"
        else:
            cell = "　"
        six_html += f"<td style='padding:1px 6px;text-align:center'>{cell}</td>"
    six_html += "</tr>"
six_html += '</tbody></table></div>'
st.markdown(six_html, unsafe_allow_html=True)

# ---- 미니 토스트 알림 (되돌리기/초기화)
fb = pred.get_feedback()
if fb:
    st.markdown(f'<div class="toast-mini">{fb}</div>', unsafe_allow_html=True)

# ---- 통계 (네온/가독성/더 진한 컬러)
s = pred.stats()
stats_html = f"""
<div style="display:grid;grid-template-columns: repeat(3,1fr);gap:17px;margin-bottom:2px;">
  <div class='stats-box'>
    <span class='stats-title'>🧮 총입력</span><br>
    <span style='font-size:1.28em;color:#fff;font-weight:800;text-shadow:0 0 14px #67fff1;'>{s['총입력']}</span>
  </div>
  <div class='stats-box'>
    <span class='stats-title'>🎯 적중률</span><br>
    <span style='font-size:1.35em;color:#f4ff52;font-weight:900;text-shadow:0 0 12px #fff6a1,0 0 7px #fff;'>{s['적중률(%)']}%</span>
  </div>
  <div class='stats-box'>
    <span class='stats-title'>💡 현재 연승</span><br>
    <span style='font-size:1.19em;color:#9bfde6;font-weight:900;'>{s['현재연승']}</span>
  </div>
  <div class='stats-box'>
    <span class='stats-title'>🔥 현재 연패</span><br>
    <span style='font-size:1.19em;color:#ff456a;font-weight:900;'>{s['현재연패']}</span>
  </div>
  <div class='stats-box'>
    <span class='stats-title'>🏆 최대 연승</span><br>
    <span style='font-size:1.16em;color:#b6f8ff;font-weight:900;'>{s['최대연승']}</span>
  </div>
  <div class='stats-box'>
    <span class='stats-title'>💣 최대 연패</span><br>
    <span style='font-size:1.19em;color:#ff456a;font-weight:900;'>{s['최대연패']}</span>
  </div>
</div>
"""
st.markdown(stats_html, unsafe_allow_html=True)

if st.button("🐞 버그 리포트 복사"):
    st.code(
        f"입력 기록: {pred.history}\n통계: {s}\n알고리즘 기록:\n" +
        "\n".join(pred.algo_logs[-8:])
    )

st.markdown(
    '<div style="margin-top:15px;font-size:1.02em;color:#44ffde;text-align:right;">'
    'MetaRunner™ AI | v1.15 |  | by 짜라라잔</div>',
    unsafe_allow_html=True
)

# ---- 진입 타이밍 AI 해설 (analysis_trail) 추가 ----
if st.session_state.show_all_trail and hasattr(pred, "analysis_trail") and pred.analysis_trail:
    expl_html = "<div style='background:#10232d;padding:11px 18px 7px 13px;margin:3px 0 17px 0;border-radius:13px;font-size:0.98em;color:#d2fefa;text-shadow:0 0 5px #6bffea90;'>"
    expl_html += "<b>진입판단 전체 해설 이력 (최대 20개)</b><hr style='margin:6px 0 8px 0;border:0;border-top:1.1px solid #1ff9f5;'/>"
    for line in pred.analysis_trail[-20:]:
        expl_html += f"<div style='margin-bottom:2px;'>• {line}</div>"
    expl_html += "</div>"
    st.markdown(expl_html, unsafe_allow_html=True)
