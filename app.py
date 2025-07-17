import streamlit as st
import random, copy, time
from collections import Counter, defaultdict, deque

# ---------- AI 클래스 및 유틸리티 함수 ----------
class ConcordanceAI:
    def __init__(self):
        self.history, self.pb_history, self.hit_record = [], [], []
        self.bet_count, self.correct, self.incorrect = 0, 0, 0
        self.current_win, self.current_loss, self.max_win, self.max_loss = 0, 0, 0, 0
        self.cooldown_turns = 0
        
        # --- 새로운 통합 경쟁 모델 ---
        self.strategies = ['ngram_3', 'ngram_4', 'ngram_5', 'china_pure', 'trend_pure']
        self.recent_performance = {s: deque(maxlen=20) for s in self.strategies}
        self.ngram_counts = {3: defaultdict(Counter), 4: defaultdict(Counter), 5: defaultdict(Counter)}

        self.analysis_text = "AI: 초기 데이터 수집 중..."
        self.next_prediction, self.should_bet_now = None, False
        self.popup_trigger, self.popup_type = False, None

    def _get_legacy_predictions(self, pb_history):
        pure = [x for x in pb_history if x in 'PB']
        if len(pure) < 6: return {k: random.choice(['P','B']) for k in ['china', 'trend']}
        preds = {}
        cnt = Counter(pure[-4:])
        preds['trend'] = 'P' if cnt['P'] > cnt['B'] else ('B' if cnt['B'] > cnt['P'] else random.choice(['P','B']))
        preds['china'] = random.choice(['P','B']) # Placeholder
        return preds

    def _get_ngram_prediction(self, history, n):
        if len(history) < n - 1: return random.choice(['P', 'B'])
        key = tuple(history[-(n-1):])
        counter = self.ngram_counts[n].get(key, None)
        if not counter: return random.choice(['P', 'B'])
        return max(counter, key=counter.get)

    def _analyze_meta_clarity(self):
        pure = [x for x in self.pb_history if x in 'PB']
        if len(pure) < 6: return "혼돈"
        last6_str = "".join(pure[-6:])
        if last6_str.count(last6_str[0]) >= 5 or last6_str in ("PBPBPB", "BPBPBP"): return "선명함"
        if "PPP" in last6_str or "BBB" in last6_str or "PBP" in last6_str or "BPB" in last6_str: return "보통"
        return "혼돈"

    def _calculate_skip_turns(self, on_win):
        clarity = self._analyze_meta_clarity()
        if on_win: # 적중 시 1~2턴 스킵
            if clarity == "선명함": return 1
            return 2
        else: # 3연패 이상 시 3~4턴 스킵
            if clarity == "혼돈": return 4
            return 3

    def process_next_turn(self):
        self.should_bet_now = False
        self.next_prediction = None

        if len(self.pb_history) < 10:
            self.analysis_text = f"AI: 초기 데이터 수집 중..."
            return

        if self.cooldown_turns > 0:
            self.analysis_text = f"AI: 분석중 ({self.cooldown_turns}턴 남음)."
            self.cooldown_turns -= 1
            return
        
        # --- 통합 경쟁을 통해 '챔피언' 전략 결정 ---
        strategy_preds = {
            'ngram_3': self._get_ngram_prediction(self.pb_history, 3),
            'ngram_4': self._get_ngram_prediction(self.pb_history, 4),
            'ngram_5': self._get_ngram_prediction(self.pb_history, 5),
            'china_pure': self._get_legacy_predictions(self.pb_history)['china'],
            'trend_pure': self._get_legacy_predictions(self.pb_history)['trend'],
        }
        
        strategy_hit_rates = {s: (sum(p) / len(p) if p else 0) for s, p in self.recent_performance.items()}
        champion = max(strategy_hit_rates, key=strategy_hit_rates.get)
        
        self.next_prediction = strategy_preds.get(champion, random.choice(['P', 'B']))
        self.analysis_text = f"AI: [통합경쟁] '{champion}' 전략 선택 ({self.next_prediction} 예측)"
        self.should_bet_now = True

    def handle_input(self, r):
        # 학습 로직
        if r in 'PB' and len(self.pb_history) >= 6:
            # 베팅이 실행될 예정이었다면, 모든 전략의 성과를 기록
            if self.should_bet_now:
                # 베팅 전 시점의 기록으로 예측
                history_before_bet = self.pb_history
                preds_before_bet = {
                    'ngram_3': self._get_ngram_prediction(history_before_bet, 3),
                    'ngram_4': self._get_ngram_prediction(history_before_bet, 4),
                    'ngram_5': self._get_ngram_prediction(history_before_bet, 5),
                    'china_pure': self._get_legacy_predictions(history_before_bet)['china'],
                    'trend_pure': self._get_legacy_predictions(history_before_bet)['trend'],
                }
                for s_name, s_pred in preds_before_bet.items():
                    self.recent_performance[s_name].append(1 if s_pred == r else 0)
        
        if r == 'T':
            self.history.append(r); self.hit_record.append(None); self.process_next_turn()
            return

        # N-gram 카운트 업데이트
        if r in 'PB':
            temp_history = self.pb_history + [r]
            if len(temp_history) >= 3: self.ngram_counts[3][tuple(temp_history[-3:-1])][temp_history[-1]] += 1
            if len(temp_history) >= 4: self.ngram_counts[4][tuple(temp_history[-4:-1])][temp_history[-1]] += 1
            if len(temp_history) >= 5: self.ngram_counts[5][tuple(temp_history[-5:-1])][temp_history[-1]] += 1

        # 베팅 실행 및 결과 처리
        if self.should_bet_now:
            is_hit = (self.next_prediction == r)
            self.bet_count += 1
            self.hit_record.append("O" if is_hit else "X")
            
            if is_hit:
                self.correct += 1; self.current_win += 1; self.current_loss = 0
                self.max_win = max(self.max_win, self.current_win)
                self.cooldown_turns = self._calculate_skip_turns(on_win=True)
            else:
                self.incorrect += 1; self.current_win = 0; self.current_loss += 1
                self.max_loss = max(self.max_loss, self.current_loss)
                # 3연패 이상 시 스킵 발동
                if self.current_loss >= 3:
                    self.cooldown_turns = self._calculate_skip_turns(on_win=False)
                    self.current_loss = 0 # 스킵 후 연패 기록 초기화
        else:
            self.hit_record.append(None)
        
        self.history.append(r)
        if r in 'PB': self.pb_history.append(r)
        self.process_next_turn()

    def get_stats(self):
        return {'총입력':len(self.history),'적중률(%)':round(self.correct/self.bet_count*100,2) if self.bet_count else 0,'현재연승':self.current_win,'최대연승':self.max_win,'현재연패':self.current_loss,'최대연패':self.max_loss}


# ========== UI 파트 (변경 없음) ==========
if 'pred' not in st.session_state:
    st.session_state.pred = ConcordanceAI()
    st.session_state.pred.process_next_turn()
if 'stack' not in st.session_state: st.session_state.stack = []
if 'prev_stats' not in st.session_state: st.session_state.prev_stats = {}
pred = st.session_state.pred

st.set_page_config(layout="wide", page_title="Unified Competition AI", page_icon="🏆")

st.markdown("""
<style>
/* --- (스타일 코드는 이전과 동일하여 생략) --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #0c111b !important; color: #e0fcff !important; }
.stButton>button { border: none; border-radius: 12px; padding: 12px 24px; color: white; font-size: 1.1em; font-weight: bold; transition: all .3s ease-out; }
.stButton>button:hover { transform: translateY(-3px); filter: brightness(1.3); }
div[data-testid="stHorizontalBlock"]>div:nth-child(1) .stButton>button { background: #0c3483; box-shadow: 0 0 8px #3b82f6, 0 0 12px #3b82f6; }
div[data-testid="stHorizontalBlock"]>div:nth-child(2) .stButton>button { background: #880e4f; box-shadow: 0 0 8px #f06292, 0 0 12px #f06292; }
div[data-testid="stHorizontalBlock"]>div:nth-child(3) .stButton>button { background: #1b5e20; box-shadow: 0 0 8px #4caf50, 0 0 12px #4caf50; }
.top-stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: -10px -10px 15px -10px; }
.stat-item { background: rgba(30, 45, 60, .7); border-radius: 10px; padding: 8px; text-align: center; border: 1px solid #2a3b4e; transition: all 0.4s ease-out; }
.stat-label { font-size: 0.9em; color: #a0c8d0; margin-bottom: 4px; display: block; }
.stat-value { font-size: 1.25em; font-weight: bold; color: #e0fcff; }
.stat-changed-neon { animation: neon-flash 1s ease-in-out; }
@keyframes neon-flash { 50% { box-shadow: 0 0 5px #fff, 0 0 10px #00fffa, 0 0 15px #00fffa; border-color: #00fffa; transform: scale(1.03); } }
.fire-animation { display: inline-block; animation: fire-burn 1.2s infinite ease-in-out; text-shadow: 0 0 5px #ff5722, 0 0 10px #ff5722, 0 0 15px #ff9800; }
@keyframes fire-burn { 0%, 100% { transform: scale(1.0) rotate(-1deg); } 50% { transform: scale(1.15) rotate(1deg); } }
.skull-animation { display: inline-block; animation: skull-shake 0.4s infinite linear; text-shadow: 0 0 5px #f44336, 0 0 10px #f44336; }
@keyframes skull-shake { 0%, 100% { transform: translateY(0) rotate(0); } 25% { transform: translateY(1px) rotate(-3deg); } 75% { transform: translateY(-1px) rotate(3deg); } }
.next-prediction-box { font-size: 1.8em; font-weight: bold; color: #00fffa!important; animation: prediction-pop-in .5s ease-out; }
@keyframes prediction-pop-in { 0%{transform:scale(.5);opacity:0} 100%{transform:scale(1);opacity:1} }
.ai-waiting-bar { background: linear-gradient(90deg, rgba(43,41,0,0.6) 0%, rgba(80,70,0,0.9) 50%, rgba(43,41,0,0.6) 100%); border-radius: 10px; padding: 12px; margin: 10px 0 15px 0; color: #ffd400; font-size: 1.2em; font-weight: 900; text-align: center; width: 100%; }
.rotating-hourglass { display: inline-block; animation: rotate 2s linear infinite; }
@keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
.top-notification { text-align:center; font-weight:bold; color:white; padding:8px; margin:-10px -10px 10px -10px; border-radius:8px; animation:slide-in-out 2.5s ease-in-out forwards; }
.top-notification.hit { background: linear-gradient(90deg, #28a745, #1f8336); }
.top-notification.miss { background: linear-gradient(90deg, #dc3545, #b32a38); }
@keyframes slide-in-out { 0%{transform:translateY(-100%);opacity:0} 15%{transform:translateY(0);opacity:1} 85%{transform:translateY(0);opacity:1} 100%{transform:translateY(-100%);opacity:0} }
.sixgrid-symbol{ border-radius:50%; font-weight:bold; padding:1.5px 7px; display:inline-block; }
.sixgrid-fluo{ color:#d4ffb3; background:rgba(100,255,110,.25); }
.sixgrid-miss{ color:#ffb3b3; background:rgba(255,100,110,.25); }
.latest-result-pop { animation: pop-in 0.6s ease-out; }
@keyframes pop-in { 0% { transform: scale(0.5); } 50% { transform: scale(1.4); } 100% { transform: scale(1.0); } }
@media (max-width: 768px) { .stButton>button { padding: 10px 18px; font-size: 1.0em; } .next-prediction-box { font-size: 1.5em; } .top-stats-grid { grid-template-columns: 1fr 1fr; } }
</style>
""", unsafe_allow_html=True)

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
            
            anim_class = "latest-result-pop" if idx == len(history) - 1 else ""
            cell_content = f'<span class="sixgrid-symbol {color_class} {anim_class}">{val}</span>'
            
        six_html += f"<td style='padding:1px;text-align:center'>{cell_content}</td>"
    six_html += "</tr>"
six_html += '</tbody></table>'
st.markdown(six_html, unsafe_allow_html=True)
