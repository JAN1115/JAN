import streamlit as st
import random, copy, time
from collections import Counter, defaultdict

# ---------- 알고리즘 (ConcordanceAI 클래스 내부 메소드로 사용) ----------
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

class ConcordanceAI:
    def __init__(self):
        self.history, self.pb_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.incorrect = 0, 0, 0
        self.current_win, self.current_loss, self.max_win, self.max_loss = 0, 0, 0, 0
        
        self.predictor_stats = {
            'ngram': {'hits': 0, 'bets': 0}, 'trend': {'hits': 0, 'bets': 0},
            'china': {'hits': 0, 'bets': 0}, 'combo': {'hits': 0, 'bets': 0}
        }
        self.cooldown_turns = 0
        self.prediction_plan = []
        self.phase_turn = 0

        self.analysis_text = "AI: 초기 데이터 수집 중..."
        self.popup_trigger, self.popup_type = False, None
        self.next_prediction, self.should_bet_now = None, False

    def _get_all_predictions(self, pb_history):
        pure = [x for x in pb_history if x in 'PB']
        if len(pure) < 6: return {k: random.choice(['P','B']) for k in self.predictor_stats.keys()}
        preds = {}
        key = tuple(pure[-3:])
        c = Counter(p for i in range(len(pure)-3) if tuple(pure[i:i+3])==key and (p:=pure[i+3]))
        preds['ngram'] = c.most_common(1)[0][0] if c else random.choice(['P', 'B'])
        cnt = Counter(pure[-4:])
        preds['trend'] = 'P' if cnt['P'] > cnt['B'] else ('B' if cnt['B'] > cnt['P'] else random.choice(['P','B']))
        scores = get_4_china_scores(pb_history)
        china_preds = ['P' if score % 2 == 0 else 'B' for score in scores.values()]
        preds['china'] = Counter(china_preds).most_common(1)[0][0]
        pattern = "".join(pure[-6:])
        if pattern in ["PPPPPP", "BBBBBB"]: preds['combo'] = pure[-1]
        elif pattern in ["PBPBPB", "BPBPBP"]: preds['combo'] = pure[-2]
        else: preds['combo'] = random.choice(['P', 'B'])
        return preds

    def _analyze_meta(self, pb_history):
        pure = [x for x in pb_history if x in 'PB']
        if len(pure) < 6: return "혼돈"
        
        last6 = pure[-6:]
        if last6.count('P') >= 5 or last6.count('B') >= 5: return "장줄"
        
        last4 = pure[-4:]
        if last4 in (['P','B','P','B'], ['B','P','B','P']): return "퐁당퐁당"

        # 선제적 진입 조건
        if last4.count('P') == 3 or last4.count('B') == 3: return "혼돈 (장줄 조짐)"
        last5 = pure[-5:]
        if last5 in (['P','B','P','B','P'], ['B','P','B','P','B']): return "혼돈 (퐁당퐁당 조짐)"
        
        return "혼돈"

    def process_next_turn(self):
        self.should_bet_now = False
        self.next_prediction = None
        self.popup_trigger = False

        if len(self.pb_history) < 6:
            self.analysis_text = f"AI: 데이터 수집 중... ({len(self.pb_history)}/6)"
            return

        if self.cooldown_turns > 0:
            self.analysis_text = f"AI: 베팅 후 휴식 ({self.cooldown_turns}턴 남음)"
            self.cooldown_turns -= 1
            return
        
        if not self.prediction_plan:
            meta = self._analyze_meta(self.pb_history)
            all_preds = self._get_all_predictions(self.pb_history)
            
            plan_keys = []
            if "장줄" in meta:
                plan_keys = ['trend', 'combo', 'china']
            elif "퐁당퐁당" in meta:
                plan_keys = ['ngram', 'combo', 'china']
            
            if plan_keys: # 장줄 또는 퐁당퐁당 패턴 감지 시 계획 수립
                self.prediction_plan = [all_preds[key] for key in plan_keys]
                self.phase_turn = 0
                self.analysis_text = f"AI: '{meta}' 패턴 감지, 계획 실행"
            else: # 혼돈 상태일 때는 대기
                self.analysis_text = "AI: 혼돈 상태, 패턴 분석 중..."
                return
        
        if self.prediction_plan:
            self.next_prediction = self.prediction_plan[self.phase_turn]
            self.should_bet_now = True
            self.analysis_text = f"AI: 계획 {self.phase_turn + 1}/3 실행 중"

    def handle_input(self, r):
        if r == 'T':
            self.history.append(r)
            self.hit_record.append(None)
            self.analysis_text = "AI: 타이 발생, 베팅 무효"
            self.process_next_turn()
            return

        all_preds = self._get_all_predictions(self.pb_history)
        for key, pred_val in all_preds.items():
            self.predictor_stats[key]['bets'] += 1
            if pred_val == r: self.predictor_stats[key]['hits'] += 1
        
        if self.should_bet_now:
            self.bet_count += 1
            is_hit = (self.next_prediction == r)
            
            if is_hit:
                self.correct += 1; self.current_win += 1; self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
                self.popup_type = "hit"; self.hit_record.append("O")
            else:
                self.incorrect += 1; self.current_win = 0; self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
                self.popup_type = "miss"; self.hit_record.append("X")

            self.popup_trigger = True
            self.phase_turn += 1

            # 계획 종료 조건: 적중했거나, 3턴이 모두 끝났을 때
            if is_hit or self.phase_turn >= len(self.prediction_plan):
                self.prediction_plan = []
                self.phase_turn = 0
                self.cooldown_turns = 3
        else:
            self.hit_record.append(None)
        
        self.history.append(r)
        if r in 'PB': self.pb_history.append(r)
        self.process_next_turn()

    def get_stats(self):
        return {'총입력':len(self.history),'적중률(%)':round(self.correct/self.bet_count*100,2) if self.bet_count else 0,'현재연승':self.current_win,'최대연승':self.max_win,'현재연패':self.current_loss,'최대연패':self.max_loss}

# ========== UI 파트 ==========
if 'pred' not in st.session_state: 
    st.session_state.pred = ConcordanceAI()
    st.session_state.pred.process_next_turn()
if 'stack' not in st.session_state: st.session_state.stack = []
if 'prev_stats' not in st.session_state: st.session_state.prev_stats = {}
pred = st.session_state.pred

st.set_page_config(layout="wide", page_title="MetaRunner AI v6.0", page_icon="🧠")
st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#0c111b!important;color:#e0fcff!important;}
.stButton>button{border:none;border-radius:12px;padding:12px 24px;color:white;font-size:1.1em;font-weight:bold;transition:all .3s ease-out;}.stButton>button:hover{transform:translateY(-3px);filter:brightness(1.3);}
div[data-testid="stHorizontalBlock"]>div:nth-child(1) .stButton>button{background:#0c3483;box-shadow:0 0 8px #3b82f6,0 0 12px #3b82f6;}
div[data-testid="stHorizontalBlock"]>div:nth-child(2) .stButton>button{background:#880e4f;box-shadow:0 0 8px #f06292,0 0 12px #f06292;}
div[data-testid="stHorizontalBlock"]>div:nth-child(3) .stButton>button{background:#1b5e20;box-shadow:0 0 8px #4caf50,0 0 12px #4caf50;}
.sixgrid-symbol{border-radius:50%;font-weight:bold;padding:1.5px 7px;display:inline-block;}.sixgrid-fluo{color:#d4ffb3;background:rgba(100,255,110,.25);}.sixgrid-miss{color:#ffb3b3;background:rgba(255,100,110,.25);}
.stat-changed{animation:flash 1s ease-out;} @keyframes flash{50%{background:rgba(40,167,69,.3);transform:scale(1.05);}}
.rotating-hourglass{display:inline-block;animation:rotate 2s linear infinite;} @keyframes rotate{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.next-prediction-box {font-size:1.8em;font-weight:bold;color:#00fffa!important;animation:prediction-pop-in .5s ease-out;}
@keyframes prediction-pop-in {0%{transform:scale(.5);opacity:0}100%{transform:scale(1);opacity:1}}
.top-notification {text-align:center;font-weight:bold;color:white;padding:8px;margin:-10px -10px 10px -10px;border-radius:8px;animation:slide-in-out 2.5s ease-in-out forwards;}
.top-notification.hit { background: linear-gradient(90deg, #28a745, #1f8336); }
.top-notification.miss { background: linear-gradient(90deg, #dc3545, #b32a38); }
@keyframes slide-in-out {0%{transform:translateY(-100%);opacity:0} 15%{transform:translateY(0);opacity:1} 85%{transform:translateY(0);opacity:1} 100%{transform:translateY(-100%);opacity:0}}
@media (max-width: 768px) {
    .stButton>button { padding: 10px 18px; font-size: 1.0em; }
    .next-prediction-box { font-size: 1.5em; }
    .stats-container { grid-template-columns: 1fr 1fr !important; }
    .stExpander { margin-bottom: 10px; }
}
</style>""", unsafe_allow_html=True)

if pred.popup_trigger:
    result_class = "hit" if pred.popup_type == "hit" else "miss"
    result_text = "🎉 적중!" if pred.popup_type == "hit" else "💥 미적중!"
    st.markdown(f'<div class="top-notification {result_class}">{result_text}</div>', unsafe_allow_html=True)

st.markdown(f"🧠 **AI 분석**: {pred.analysis_text}", unsafe_allow_html=True)

npred, should_bet = pred.next_prediction, pred.should_bet_now
if should_bet and npred:
    ICONS = {'P':'🔵','B':'🔴'}
    st.markdown(f'<div style="text-align:center;margin:5px 0 15px 0;"><span class="next-prediction-box">NEXT {ICONS[npred]}</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div style="text-align:center;"><div style="background:#2b2900;border-radius:16px;padding:10px 32px;margin:10px auto;color:#ffd400;font-size:1.22em;font-weight:900;display:inline-block;"><span class="rotating-hourglass">⏳</span> AI 대기중...</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    with st.expander("📈 알고리즘 성과 보기"):
        predictor_labels = list(pred.predictor_stats.keys())
        sub_cols = st.columns(len(predictor_labels))
        for i, key in enumerate(predictor_labels):
            stats = pred.predictor_stats[key]
            rate = round(stats['hits'] / stats['bets'] * 100, 1) if stats['bets'] > 0 else 0
            with sub_cols[i]:
                st.metric(label=f"{key.upper()}", value=f"{rate}%")

with col2:
    s, prev_s = pred.get_stats(), st.session_state.prev_stats
    win_changed = "stat-changed" if s['현재연승'] > 0 and s['현재연승'] != prev_s.get('현재연승', 0) else ""
    loss_changed = "stat-changed" if s['현재연패'] > 0 and s['현재연패'] != prev_s.get('현재연패', 0) else ""
    st.markdown(f"""
    <div class="stats-container" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div style="background:rgba(30,45,60,.5);border-radius:10px;padding:5px;text-align:center;">🎯적중률<br><span style="color:#fff;font-size:1.2em;">{s['적중률(%)']}%</span></div>
        <div style="background:rgba(30,45,60,.5);border-radius:10px;padding:5px;text-align:center;">📊총 베팅<br><span style="color:#fff;font-size:1.2em;">{pred.bet_count}/{s['총입력']}</span></div>
        <div class="{win_changed}" style="background:rgba(30,45,60,.5);border-radius:10px;padding:5px;text-align:center;">💡연승<br><span style="color:#fff;font-size:1.2em;">{s['현재연승']} (최대 {s['최대연승']})</span></div>
        <div class="{loss_changed}" style="background:rgba(30,45,60,.5);border-radius:10px;padding:5px;text-align:center;">🔥연패<br><span style="color:#fff;font-size:1.2em;">{s['현재연패']} (최대 {s['최대연패']})</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.prev_stats = s.copy()

st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)

def handle_click(result):
    if 'stack' not in st.session_state: st.session_state.stack = []
    st.session_state.stack.append(copy.deepcopy(st.session_state.pred))
    st.session_state.pred.handle_input(result)
    st.rerun()

button_cols = st.columns([1,1,1,0.5,0.5])
button_cols[0].button("플레이어 (P)", use_container_width=True, on_click=handle_click, args=("P",))
button_cols[1].button("뱅커 (B)", use_container_width=True, on_click=handle_click, args=("B",))
button_cols[2].button("타이 (T)", use_container_width=True, on_click=handle_click, args=("T",))

if button_cols[3].button("↩️", help="이전 상태로 되돌립니다."):
    if st.session_state.stack:
        st.session_state.pred = st.session_state.stack.pop()
        st.rerun()

if button_cols[4].button("🗑️", help="모든 기록을 초기화합니다."):
    st.session_state.pred = ConcordanceAI()
    st.session_state.stack = []
    st.rerun()

st.markdown('<hr style="border:1px solid #222; margin: 15px 0;">', unsafe_allow_html=True)
st.markdown("🪧 <b>6매 기록</b>", unsafe_allow_html=True)
history, hitrec = pred.history, pred.hit_record
max_row = 6
ncols = (len(history) + max_row - 1) // max_row if len(history) > 0 else 0
six_html = '<table style="border-spacing:4px 2px;width:100%;"><tbody>'
for r in range(max_row):
    six_html += "<tr>"
    for c in range(ncols):
        idx, cell_content = c * max_row + r, "&nbsp;"
        if idx < len(history):
            ICONS = {'P':'🔵','B':'🔴','T':'🟢'}
            val = ICONS.get(history[idx],"")
            color_class = ""
            if idx < len(hitrec) and hitrec[idx] is not None:
                if hitrec[idx] == "O": color_class = "sixgrid-fluo"
                elif hitrec[idx] == "X": color_class = "sixgrid-miss"
            cell_content = f'<span class="sixgrid-symbol {color_class}">{val}</span>'
        six_html += f"<td style='padding:1px;text-align:center'>{cell_content}</td>"
    six_html += "</tr>"
six_html += '</tbody></table>'
st.markdown(six_html, unsafe_allow_html=True)
