import streamlit as st
import random, copy, time
from collections import Counter, defaultdict

# ---------- 알고리즘 유틸리티 함수 (클래스 외부) ----------
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
    pure_pb = [r for r in pb_history if r in 'PB']
    if pure_pb:
        last = pure_pb[-1]
        for r in reversed(pure_pb):
            if r == last:
                streaks += 1
            else:
                break
    scores['빅로드'] = streaks
    scores['빅아이'] = calc_china_roads(bigroad, 1)
    scores['스몰로드'] = calc_china_roads(bigroad, 2)
    scores['바퀴'] = calc_china_roads(bigroad, 3)
    return scores

# ---------- '3+3 하이브리드' AI 클래스 ----------
class ConcordanceAI:
    def __init__(self):
        self.history, self.pb_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.incorrect = 0, 0, 0
        self.current_win, self.current_loss, self.max_win, self.max_loss = 0, 0, 0, 0
        
        self.predictor_stats = {
            'ngram': {'hits': 0, 'bets': 0}, 'trend': {'hits': 0, 'bets': 0},
            'china': {'hits': 0, 'bets': 0}, 'combo': {'hits': 0, 'bets': 0}
        }
        self.contextual_stats = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'bets': 0}))

        self.cooldown_turns = 0
        
        self.plan_step = 0
        self.plan_expert = None

        self.analysis_text = "AI: 초기 데이터 수집 중..."
        self.popup_trigger, self.popup_type = False, None
        self.next_prediction, self.should_bet_now = None, False

    def _get_current_context(self, history):
        pure_hist = [x for x in history if x in 'PB']
        if len(pure_hist) < 6: return ('초기', 0)
        meta, _ = self._analyze_meta_with_scores(pure_hist)
        streak = 0
        if pure_hist:
            last_val = pure_hist[-1]
            for val in reversed(pure_hist):
                if val == last_val: streak += 1
                else: break
        return (meta, streak)

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

    def _analyze_meta_with_scores(self, pb_history):
        pure = [x for x in pb_history if x in 'PB']
        if len(pure) < 6: return "혼돈", 0
        last6 = pure[-6:]
        streak_score = (max(last6.count('P'), last6.count('B')) - 3) * 30
        zigzag_score = 0
        for i in range(len(last6) - 1):
            if last6[i] != last6[i+1]: zigzag_score += 20
        CONFIDENCE_THRESHOLD = 75
        if streak_score > zigzag_score and streak_score >= CONFIDENCE_THRESHOLD: return "장줄", streak_score
        elif zigzag_score > streak_score and zigzag_score >= CONFIDENCE_THRESHOLD: return "퐁당퐁당", zigzag_score
        else: return "혼돈", 0
        
    def _get_best_current_expert(self):
        current_context = self._get_current_context(self.pb_history)
        context_specific_stats = self.contextual_stats[current_context]

        def get_contextual_hit_rate(predictor_key):
            stats = context_specific_stats.get(predictor_key)
            if stats and stats['bets'] > 3: return stats['hits'] / stats['bets']
            return -1

        sorted_predictors = sorted(
            self.predictor_stats.keys(),
            key=lambda k: (get_contextual_hit_rate(k), self.predictor_stats[k]['hits'] / self.predictor_stats[k]['bets'] if self.predictor_stats[k]['bets'] > 0 else 0),
            reverse=True
        )
        return sorted_predictors[0]

    def process_next_turn(self):
        self.should_bet_now = False
        self.next_prediction = None
        self.popup_trigger = False

        if len(self.pb_history) < 10:
            self.analysis_text = f"AI: 상황 학습 데이터 수집 중... ({len(self.pb_history)}/10)"
            return

        if self.cooldown_turns > 0:
            self.analysis_text = f"AI: 분석중 ({self.cooldown_turns}턴 남음)." # <-- 문구 수정
            self.cooldown_turns -= 1
            return
        
        if self.plan_step == 0:
            self.analysis_text = "AI: 새로운 작전 계획 수립 대기 중..."
            self.plan_step = 1
            self.plan_expert = self._get_best_current_expert()
        
        all_preds = self._get_all_predictions(self.pb_history)
        
        if 1 <= self.plan_step <= 3:
            predictor_to_use = self.plan_expert
            self.analysis_text = f"AI: [일관성 단계 {self.plan_step}/3] 전문가 '{predictor_to_use.upper()}'"
        elif 4 <= self.plan_step <= 6:
            predictor_to_use = self._get_best_current_expert()
            self.analysis_text = f"AI: [적응 단계 {self.plan_step-3}/3] 전문가 '{predictor_to_use.upper()}'"
        else:
            return

        self.next_prediction = all_preds[predictor_to_use]
        self.should_bet_now = True
        self.analysis_text += f" ({self.next_prediction} 예측)"


    def handle_input(self, r):
        if len(self.pb_history) >= 6:
            context_before_result = self._get_current_context(self.pb_history)
            all_preds = self._get_all_predictions(self.pb_history)
            for key, pred_val in all_preds.items():
                self.predictor_stats[key]['bets'] += 1
                self.contextual_stats[context_before_result][key]['bets'] += 1
                if pred_val == r:
                    self.predictor_stats[key]['hits'] += 1
                    self.contextual_stats[context_before_result][key]['hits'] += 1

        if r == 'T':
            self.history.append(r)
            self.hit_record.append(None)
            self.process_next_turn() 
            return

        if self.should_bet_now:
            self.bet_count += 1
            is_hit = (self.next_prediction == r)
            self.hit_record.append("O" if is_hit else "X")
            
            if is_hit:
                self.correct += 1; self.current_win += 1; self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
                self.popup_type = "hit"
                self.cooldown_turns = 2
                self.plan_step = 0
                self.plan_expert = None
            else:
                self.incorrect += 1; self.current_win = 0; self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
                self.popup_type = "miss"
                self.plan_step += 1

                if self.plan_step > 6:
                    self.cooldown_turns = 4
                    self.plan_step = 0
                    self.plan_expert = None

            self.popup_trigger = True
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

st.set_page_config(layout="wide", page_title="JAN Hybrid AI 1.15v", page_icon="🧬")

# --- UI Customization ---
st.markdown("""
<style>
/* --- 기본 배경 및 폰트 --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #0c111b !important;
    color: #e0fcff !important;
}
/* --- 버튼 스타일 --- */
.stButton>button {
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    color: white;
    font-size: 1.1em;
    font-weight: bold;
    transition: all .3s ease-out;
}
.stButton>button:hover {
    transform: translateY(-3px);
    filter: brightness(1.3);
}
div[data-testid="stHorizontalBlock"]>div:nth-child(1) .stButton>button { background: #0c3483; box-shadow: 0 0 8px #3b82f6, 0 0 12px #3b82f6; }
div[data-testid="stHorizontalBlock"]>div:nth-child(2) .stButton>button { background: #880e4f; box-shadow: 0 0 8px #f06292, 0 0 12px #f06292; }
div[data-testid="stHorizontalBlock"]>div:nth-child(3) .stButton>button { background: #1b5e20; box-shadow: 0 0 8px #4caf50, 0 0 12px #4caf50; }

/* --- 상단 고정 스탯 바 --- */
.top-stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: -10px -10px 15px -10px;
}
.stat-item {
    background: rgba(30, 45, 60, .7);
    border-radius: 10px;
    padding: 8px;
    text-align: center;
    border: 1px solid #2a3b4e;
    transition: all 0.4s ease-out;
}
.stat-label {
    font-size: 0.9em;
    color: #a0c8d0;
    margin-bottom: 4px;
    display: block;
}
.stat-value {
    font-size: 1.25em;
    font-weight: bold;
    color: #e0fcff;
}
/* --- 스탯 변경 시 네온 애니메이션 --- */
.stat-changed-neon {
    animation: neon-flash 1s ease-in-out;
}
@keyframes neon-flash {
    50% {
        box-shadow: 0 0 5px #fff, 0 0 10px #00fffa, 0 0 15px #00fffa;
        border-color: #00fffa;
        transform: scale(1.03);
    }
}
/* --- 연승/연패 아이콘 및 애니메이션 --- */
.fire-animation {
    display: inline-block;
    animation: fire-burn 1.2s infinite ease-in-out;
    text-shadow: 0 0 5px #ff5722, 0 0 10px #ff5722, 0 0 15px #ff9800;
}
@keyframes fire-burn {
    0%, 100% { transform: scale(1.0) rotate(-1deg); }
    50% { transform: scale(1.15) rotate(1deg); }
}
.skull-animation {
    display: inline-block;
    animation: skull-shake 0.4s infinite linear;
    text-shadow: 0 0 5px #f44336, 0 0 10px #f44336;
}
@keyframes skull-shake {
    0%, 100% { transform: translateY(0) rotate(0); }
    25% { transform: translateY(1px) rotate(-3deg); }
    75% { transform: translateY(-1px) rotate(3deg); }
}

/* --- AI 예측 및 대기 UI --- */
.next-prediction-box { font-size: 1.8em; font-weight: bold; color: #00fffa!important; animation: prediction-pop-in .5s ease-out; }
@keyframes prediction-pop-in { 0%{transform:scale(.5);opacity:0} 100%{transform:scale(1);opacity:1} }

.ai-waiting-bar {
    background: linear-gradient(90deg, rgba(43,41,0,0.6) 0%, rgba(80,70,0,0.9) 50%, rgba(43,41,0,0.6) 100%);
    border-radius: 10px;
    padding: 12px;
    margin: 10px 0 15px 0;
    color: #ffd400;
    font-size: 1.2em;
    font-weight: 900;
    text-align: center;
    width: 100%;
}
.rotating-hourglass { display: inline-block; animation: rotate 2s linear infinite; }
@keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

/* --- 적중/미적중 팝업 --- */
.top-notification { text-align:center; font-weight:bold; color:white; padding:8px; margin:-10px -10px 10px -10px; border-radius:8px; animation:slide-in-out 2.5s ease-in-out forwards; }
.top-notification.hit { background: linear-gradient(90deg, #28a745, #1f8336); }
.top-notification.miss { background: linear-gradient(90deg, #dc3545, #b32a38); }
@keyframes slide-in-out { 0%{transform:translateY(-100%);opacity:0} 15%{transform:translateY(0);opacity:1} 85%{transform:translateY(0);opacity:1} 100%{transform:translateY(-100%);opacity:0} }

/* --- 6매 기록 --- */
.sixgrid-symbol{ border-radius:50%; font-weight:bold; padding:1.5px 7px; display:inline-block; }
.sixgrid-fluo{ color:#d4ffb3; background:rgba(100,255,110,.25); }
.sixgrid-miss{ color:#ffb3b3; background:rgba(255,100,110,.25); }
.latest-result-pop { animation: pop-in 0.6s ease-out; } /* <-- 6매 기록 애니메이션 */
@keyframes pop-in {
    0% { transform: scale(0.5); }
    50% { transform: scale(1.4); }
    100% { transform: scale(1.0); }
}

/* --- 모바일 반응형 --- */
@media (max-width: 768px) {
    .stButton>button { padding: 10px 18px; font-size: 1.0em; }
    .next-prediction-box { font-size: 1.5em; }
    .top-stats-grid { grid-template-columns: 1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)

# --- 상단 고정 스탯 바 ---
s = pred.get_stats()
prev_s = st.session_state.prev_stats

win_anim_class = "stat-changed-neon" if s['현재연승'] > 0 and s['현재연승'] != prev_s.get('현재연승', 0) else ""
loss_anim_class = "stat-changed-neon" if s['현재연패'] > 0 and s['현재연패'] != prev_s.get('현재연패', 0) else ""

win_icon = f"<span class='fire-animation'>🔥</span>" if s['현재연승'] > 0 else "⚪"
loss_icon = f"<span class='skull-animation'>💀</span>" if s['현재연패'] > 0 else "⚪"

st.markdown(f"""
<div class="top-stats-grid">
    <div class="stat-item">
        <span class="stat-label">🎯 적중률</span>
        <span class="stat-value">{s['적중률(%)']}%</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">📊 총 베팅</span>
        <span class="stat-value">{pred.bet_count}/{s['총입력']}</span>
    </div>
    <div class="stat-item {win_anim_class}">
        <span class="stat-label">{win_icon} 연승</span>
        <span class="stat-value">{s['현재연승']} (최대 {s['최대연승']})</span>
    </div>
    <div class="stat-item {loss_anim_class}">
        <span class="stat-label">{loss_icon} 연패</span>
        <span class="stat-value">{s['현재연패']} (최대 {s['최대연패']})</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.session_state.prev_stats = s.copy()


if pred.popup_trigger:
    result_class = "hit" if pred.popup_type == "hit" else "miss"
    result_text = "🎉 적중!" if pred.popup_type == "hit" else "💥 미적중!"
    st.markdown(f'<div class="top-notification {result_class}">{result_text}</div>', unsafe_allow_html=True)
    st.session_state.pred.popup_trigger = False 

st.markdown(f"🧠 **AI 분석**: {pred.analysis_text}", unsafe_allow_html=True)

npred, should_bet = pred.next_prediction, pred.should_bet_now
if should_bet and npred:
    ICONS = {'P':'🔵','B':'🔴'}
    st.markdown(f'<div style="text-align:center;margin:5px 0 15px 0;"><span class="next-prediction-box">NEXT {ICONS[npred]}</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="ai-waiting-bar"><span class="rotating-hourglass">⏳</span> AI 분석중...</div>', unsafe_allow_html=True)

def handle_click(result):
    if 'stack' not in st.session_state: st.session_state.stack = []
    st.session_state.stack.append(copy.deepcopy(st.session_state.pred))
    st.session_state.pred.handle_input(result)
    # st.rerun() # <-- No-op 오류를 막기 위해 제거

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
            
            # 마지막 결과에 애니메이션 클래스 추가
            anim_class = "latest-result-pop" if idx == len(history) - 1 else ""
            cell_content = f'<span class="sixgrid-symbol {color_class} {anim_class}">{val}</span>'
            
        six_html += f"<td style='padding:1px;text-align:center'>{cell_content}</td>"
    six_html += "</tr>"
six_html += '</tbody></table>'
st.markdown(six_html, unsafe_allow_html=True)

# --- 알고리즘 성과 보기 (맨 아래로 이동) ---
st.markdown('<hr style="border:1px solid #222; margin: 25px 0 15px 0;">', unsafe_allow_html=True)
with st.expander("📈 알고리즘 전체 성과 보기"):
    predictor_labels = list(pred.predictor_stats.keys())
    sub_cols = st.columns(len(predictor_labels))
    for i, key in enumerate(predictor_labels):
        stats = pred.predictor_stats[key]
        rate = round(stats['hits'] / stats['bets'] * 100, 1) if stats['bets'] > 0 else 0
        with sub_cols[i]:
            st.metric(label=f"{key.upper()}", value=f"{rate}%", help=f"{stats['hits']} / {stats['bets']}")
