import streamlit as st
import pyupbit
import pandas as pd
import sqlite3
import time
import streamlit.components.v1 as components
from datetime import datetime
import threading
import os
from dotenv import load_dotenv

# --- 1. API 키 및 초기 설정 (.env 파일 로드) ---
load_dotenv()
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")

upbit = pyupbit.Upbit(access, secret)

def init_db():
    conn = sqlite3.connect("upbit_trading.db")
    cur = conn.cursor()
    
    # [핵심 수정] 테이블 생성 시 모든 필요한 컬럼을 포함하여 정의합니다.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            ticker TEXT PRIMARY KEY,
            is_active INTEGER DEFAULT 0,
            budget INTEGER DEFAULT 5000,
            stop_loss REAL DEFAULT 0.03,
            max_daily_buy INTEGER DEFAULT 100000,
            max_daily_sell INTEGER DEFAULT 100000,
            target_profit REAL DEFAULT 0.05,
            ai_mode INTEGER DEFAULT 0
        )
    """)
    
    # [핵심 수정] 기존에 이미 생성된 DB 파일이 있을 경우, 누락된 컬럼을 하나씩 체크하여 추가합니다.
    cur.execute("PRAGMA table_info(user_settings)")
    columns = [column[1] for column in cur.fetchall()]
    
    # 추가해야 할 컬럼 리스트와 해당 타입/기본값 정의
    required_columns = {
        "max_daily_buy": "INTEGER DEFAULT 100000",
        "max_daily_sell": "INTEGER DEFAULT 100000",
        "target_profit": "REAL DEFAULT 0.05",
        "ai_mode": "INTEGER DEFAULT 0"
    }
    
    for col_name, col_def in required_columns.items():
        if col_name not in columns:
            try:
                cur.execute(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_def}")
                print(f"DB Update: {col_name} 컬럼 추가 완료")
            except Exception as e:
                print(f"DB Update Error ({col_name}): {e}")
                
    conn.commit()
    conn.close()

# 애플리케이션 시작 시 DB 초기화 및 컬럼 체크 강제 실행
init_db()

# [추가] 주문 결과를 정밀하게 체크하여 실제 에러 메시지를 반환하는 함수
def check_order_result(res):
    if res is None:
        return False, "업비트 서버로부터 응답이 없습니다. (API 키 혹은 네트워크 확인)"
    if isinstance(res, dict) and 'error' in res:
        # 업비트가 보내주는 실제 에러 메시지 추출 (예: 권한 없음, IP 미등록 등)
        err_msg = res.get('error', {}).get('message', '알 수 없는 오류')
        return False, f"업비트 거절 사유: {err_msg}"
    if isinstance(res, dict) and 'uuid' in res:
        return True, "실제 업비트 주문 접수 성공!"
    return False, f"비정상 응답 발생: {res}"

# AI 타겟 가격 계산 함수 (AI 모드용)
def get_ai_target_prices(ticker):
    try:
        df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        if df is None or len(df) < 2:
            return None, None
        # AI 감시가 (변동성 돌파 타겟)
        target_buy = df.iloc[1]['open'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * 0.5
        # AI 익절가 (감시가 대비 2% 상단 예시)
        target_sell = target_buy * 1.02 
        return target_buy, target_sell
    except:
        return None, None

# --- 2. 1년 백테스트 분석 로직 (캐싱 적용) ---
@st.cache_data(ttl=3600)
def get_backtest_report(ticker):
    try:
        # 지난 1년(365일) 일봉 데이터 호출
        df = pyupbit.get_ohlcv(ticker, interval="day", count=365)
        if df is None or len(df) < 365:
            return None
            
        # 변동성 돌파 전략 계산 (K=0.5)
        df['range'] = (df['high'] - df['low']) * 0.5
        df['target'] = df['open'] + df['range'].shift(1)
        
        # 수익률 계산 (돌파 시 매수, 당일 종가 매도 가정)
        df['ror'] = df.apply(lambda x: x['close'] / x['target'] if x['high'] > x['target'] else 1, axis=1)
        
        # 누적 수익률 및 MDD 계산
        df['hpr'] = df['ror'].cumprod()
        df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
        
        total_ror = (df['hpr'].iloc[-1] - 1) * 100
        mdd = df['dd'].max()
        trade_count = len(df[df['ror'] != 1])
        win_rate = (df['ror'] > 1).sum() / trade_count * 100 if trade_count > 0 else 0
        
        return {
            "수익률": total_ror,
            "MDD": mdd,
            "승률": win_rate,
            "거래횟수": trade_count
        }
    except:
        return None

# --- 3. 실시간 자동매매 엔진 (백그라운드 스레드) ---
def trading_engine():
    while True:
        try:
            conn = sqlite3.connect("upbit_trading.db")
            # 모든 컬럼(*)을 명시적으로 호출하여 판다스 데이터프레임으로 변환
            active_bots = pd.read_sql("SELECT * FROM user_settings WHERE is_active = 1", conn)
            conn.close()

            for _, bot in active_bots.iterrows():
                ticker = bot['ticker']
                curr_p = pyupbit.get_current_price(ticker)
                avg_buy_p = upbit.get_avg_buy_price(ticker)
                
                # [AI 모드 로직 시작]
                if bot['ai_mode'] == 1:
                    ai_buy_p, ai_sell_p = get_ai_target_prices(ticker)
                    
                    # 1. 미보유 중일 때 AI 감시가 돌파 시 매수
                    if avg_buy_p == 0 and curr_p >= ai_buy_p:
                        krw_bal = upbit.get_balance("KRW")
                        if krw_bal >= bot['budget'] and bot['budget'] >= 5000:
                            upbit.buy_market_order(ticker, bot['budget'])
                            print(f"[{ticker}] AI 감시가 돌파: 자동 매수 완료")
                    
                    # 2. 보유 중일 때 AI 익절가 도달 시 매도
                    elif avg_buy_p > 0 and curr_p >= ai_sell_p:
                        coin_bal = upbit.get_balance(ticker)
                        if coin_bal > 0:
                            upbit.sell_market_order(ticker, coin_bal)
                            print(f"[{ticker}] AI 익절가 도달: 자동 매도 완료")
                
                # [수동/공통 감시 로직]
                if avg_buy_p > 0:
                    current_ror = (curr_p / avg_buy_p) - 1
                    
                    # 3. 자동 손절 감시 (공통 적용)
                    if current_ror <= -bot['stop_loss']:
                        coin_bal = upbit.get_balance(ticker)
                        if coin_bal > 0:
                            upbit.sell_market_order(ticker, coin_bal)
                            print(f"[{ticker}] 손절선 도달: 자동 매도 완료")
                    
                    # 4. 수동 모드일 때만 사용자가 설정한 익절치 적용
                    elif bot['ai_mode'] == 0 and current_ror >= bot['target_profit']:
                        coin_bal = upbit.get_balance(ticker)
                        if coin_bal > 0:
                            upbit.sell_market_order(ticker, coin_bal)
                            print(f"[{ticker}] 사용자 설정 익절 도달: 자동 매도 완료")
            
            time.sleep(1)
        except Exception as e:
            print(f"엔진 오류: {e}")
            time.sleep(5)

if 'engine_thread' not in st.session_state:
    thread = threading.Thread(target=trading_engine, daemon=True)
    thread.start()
    st.session_state['engine_thread'] = True

# --- 4. UI 설정 및 스타일 ---
st.set_page_config(page_title="Professional Trading System", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, span, p, label, .stMarkdown { color: #333333 !important; }
    .up { color: #d60000 !important; font-weight: bold; }
    .down { color: #0051c7 !important; font-weight: bold; }
    .header-box { background-color: #ffffff; padding: 25px; border-radius: 4px; border: 1px solid #e9ecf1; border-bottom: 3px solid #f1f1f4; margin-bottom: 25px; }
    .orderbook-scroll-container { height: 400px; overflow-y: auto; border: 1px solid #f1f1f4; background-color: #ffffff; }
    .orderbook-scroll-container table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    div[data-baseweb="tab-list"] { background-color: #ffffff; border-bottom: 2px solid #f1f1f4; }
    div[data-baseweb="tab-panel"] { background-color: #ffffff; padding-top: 25px; }
    .badge-bid { background-color: #fff0f0; color: #d60000; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 12px; }
    .badge-ask { background-color: #f0f4ff; color: #0051c7; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. 사이드바 (종목 선택 및 요청하신 자산 현황) ---
all_tickers = pyupbit.get_tickers(fiat="KRW")
selected_ticker = st.sidebar.selectbox("🎯 종목 선택", all_tickers, index=all_tickers.index("KRW-BTC"))
coin_symbol = selected_ticker.split("-")[1]

st.sidebar.divider()
st.sidebar.subheader("💰 자산 현황")

try:
    balances = upbit.get_balances()
    total_buy_cash = 0    
    total_eval_cash = 0   
    
    for b in balances:
        if b['currency'] == "KRW":
            total_buy_cash += float(b['balance'])
            total_eval_cash += float(b['balance'])
        else:
            ticker = f"KRW-{b['currency']}"
            current_p = pyupbit.get_current_price(ticker)
            if current_p:
                buy_p = float(b['avg_buy_price'])
                amount = float(b['balance']) + float(b['locked'])
                total_buy_cash += buy_p * amount
                total_eval_cash += current_p * amount

    total_profit_rate = ((total_eval_cash / total_buy_cash) - 1) * 100 if total_buy_cash > 0 else 0
    
    st.sidebar.metric("총 보유 자산", f"{total_eval_cash:,.0f} KRW")
    st.sidebar.metric("총 평가 금액", f"{total_eval_cash:,.0f} KRW")
    st.sidebar.metric("총 수익률", f"{total_profit_rate:+.2f}%", delta=f"{total_eval_cash - total_buy_cash:+,.0f} KRW")

except:
    st.sidebar.warning("자산 정보 로드 실패")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 1년 백테스트 요약")
bt_res = get_backtest_report(selected_ticker)

if bt_res:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("연간 수익률", f"{bt_res['수익률']:.1f}%")
    c2.metric("승률", f"{bt_res['승률']:.1f}%")
    st.sidebar.caption(f"최대 낙폭(MDD): {bt_res['MDD']:.1f}% / 거래: {bt_res['거래횟수']}회")
    
    if bt_res['수익률'] > 15:
        st.sidebar.success("✅ 자동매매에 적합한 추세")
    elif bt_res['수익률'] < 0:
        st.sidebar.warning("⚠️ 하락장 (전략 주의)")
else:
    st.sidebar.info("데이터 분석 중...")

# --- 6. 실시간 데이터 전광판 ---
curr_price = pyupbit.get_current_price(selected_ticker)
df_day = pyupbit.get_ohlcv(selected_ticker, interval="day", count=2)
prev_close = df_day.iloc[0]['close']
change_val = curr_price - prev_close
change_rate = (change_val / prev_close) * 100
color_class = "up" if change_val >= 0 else "down"

st.markdown(f"""
<div class="header-box">
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <h2 style="margin: 0; color: #333; font-weight: 700;">{selected_ticker}</h2>
        <h1 class="{color_class}" style="margin: 0; font-size: 3rem; letter-spacing: -1px;">{curr_price:,.0f}</h1>
        <div style="display: flex; flex-direction: column; line-height: 1.2;">
            <span class="{color_class}" style="font-size: 1.1rem;">{change_rate:+.2f}%</span>
            <span class="{color_class}" style="font-size: 1.1rem;">{change_val:+,f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 7. 메인 대시보드 (미체결 탭 포함하여 수정) ---
tab_main, tab_history, tab_wait = st.tabs(["📈 트레이딩", "📄 거래 기록", "⏳ 미체결"])

with tab_main:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        chart_mode = st.radio("차트 모드", ["트레이딩뷰", "기본 차트"], horizontal=True, label_visibility="collapsed")
        chart_style = "1" if chart_mode == "트레이딩뷰" else "2"
        
        tv_html = f"""
        <div style="height:800px; border: 1px solid #e9ecf1; border-radius: 4px; overflow: hidden;">
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <div id="tv_chart" style="height:100%;"></div>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true, "symbol": "UPBIT:{coin_symbol}KRW", "interval": "30",
            "timezone": "Asia/Seoul", "theme": "light", "style": "{chart_style}",
            "locale": "ko", "container_id": "tv_chart"
          }});
          </script>
        </div>
        """
        components.html(tv_html, height=800)

    with col_right:
        if bt_res:
            st.info(f"📊 **{coin_symbol}** 1년 분석: 수익 **{bt_res['수익률']:.1f}%**, 승률 **{bt_res['승률']:.1f}%**")

        st.write("### 📊 호가")
        try:
            orderbook = pyupbit.get_orderbook(selected_ticker)
            units = orderbook['orderbook_units']
            items = [{"가격": u['ask_price'], "잔량": f"{u['ask_size']:.3f}", "구분": "매도"} for u in reversed(units[:10])] + \
                    [{"가격": u['bid_price'], "잔량": f"{u['bid_size']:.3f}", "구분": "매수"} for u in units[:10]]
            df_ob = pd.DataFrame(items)
            def style_ob(row):
                color = '#d60000' if row['구분'] == '매도' else '#0051c7'
                bg = '#fff5f5' if row['구분'] == '매도' else '#f0f7ff'
                return [f'color: {color}; background-color: {bg}'] * len(row)
            st.markdown(f'<div class="orderbook-scroll-container">{df_ob.style.apply(style_ob, axis=1).format({"가격": "{:,.0f}"}).hide(axis="index").to_html()}</div>', unsafe_allow_html=True)
        except: st.warning("호가 데이터 로딩 중...")

        st.write("### 🛒 주문 제어")
        ai_target_buy, ai_target_sell = get_ai_target_prices(selected_ticker)
        
        o_tab1, o_tab2 = st.tabs(["매수", "매도"])
        
        with o_tab1: # 매수 탭
            st.caption(f"💡 AI 감시가: {ai_target_buy:,.0f}" if ai_target_buy else "")
            b_price = st.number_input("매수 가격(KRW)", value=int(curr_price), key=f"bp_{selected_ticker}")
            b_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=0.1, format="%.4f", key=f"bq_{selected_ticker}")
            order_total_cost = int(b_price * b_qty)
            st.write(f"➔ 예상 결제 금액: **{order_total_cost:,.0f}** KRW")
            
            max_b_limit = st.number_input("일일 매수 한도(KRW)", min_value=0, value=100000, step=10000, key=f"mbl_{selected_ticker}")
            if st.button("즉시 매수", use_container_width=True, type="primary"): 
                if order_total_cost < 5000:
                    st.error("❌ 업비트 최소 주문 금액은 5,000원입니다.")
                else:
                    res = upbit.buy_limit_order(selected_ticker, b_price, b_qty)
                    success, msg = check_order_result(res)
                    if success: st.success(msg)
                    else: st.error(msg)
        
        with o_tab2: # 매도 탭
            st.caption(f"💡 AI 익절가: {ai_target_sell:,.0f}" if ai_target_sell else "")
            s_price = st.number_input("매도 가격(KRW)", value=int(curr_price), key=f"sp_{selected_ticker}")
            s_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=0.1, format="%.4f", key=f"sq_{selected_ticker}")
            order_sell_total = int(s_price * s_qty)
            st.write(f"➔ 예상 수령 금액: **{order_sell_total:,.0f}** KRW")
            
            max_s_limit = st.number_input("일일 매도 한도(KRW)", min_value=0, value=100000, step=10000, key=f"msl_{selected_ticker}")
            if st.button("즉시 매도", use_container_width=True): 
                if order_sell_total < 5000:
                    st.error("❌ 업비트 최소 주문 금액은 5,000원입니다.")
                else:
                    res = upbit.sell_limit_order(selected_ticker, s_price, s_qty)
                    success, msg = check_order_result(res)
                    if success: st.success(msg)
                    else: st.error(msg)

        with st.expander("🤖 자동매매(Bot) 상세 설정"):
            is_active_bot = st.checkbox("이 종목 엔진 가동", value=False, key=f"ab_{selected_ticker}")
            is_ai_mode = st.toggle("✨ AI 자동 감시 모드 활성화", help="활성화 시 AI 감시가에 매수하고 AI 익절가에 매도합니다.")
            
            if is_ai_mode and ai_target_buy:
                st.info(f"📍 **AI 감시가**: {ai_target_buy:,.0f} / **AI 익절가**: {ai_target_sell:,.0f}")
            
            st_loss = st.slider("손절(%)", 1, 15, 3, key=f"sl_{selected_ticker}")
            target_profit = st.slider("익절(%)", 1, 50, 5, key=f"tp_{selected_ticker}")
            
            if st.button("모든 설정 저장 및 가동"):
                conn = sqlite3.connect("upbit_trading.db")
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO user_settings 
                    (ticker, is_active, budget, stop_loss, max_daily_buy, max_daily_sell, target_profit, ai_mode) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (selected_ticker, 1 if is_active_bot else 0, order_total_cost, st_loss/100, max_b_limit, max_s_limit, target_profit/100, 1 if is_ai_mode else 0))
                conn.commit()
                conn.close()
                st.toast("저장 완료! 엔진이 감시를 시작합니다.")

# --- 8. 거래 기록 탭 ---
with tab_history:
    st.write("### 📜 최근 거래 내역")
    try:
        trades = upbit.get_order(selected_ticker, state='done')
        if trades:
            df_t = pd.DataFrame(trades)[['market', 'created_at', 'side', 'price', 'volume']]
            df_t['side'] = df_t['side'].replace({'bid': '🔵 매수', 'ask': '🔴 매도'})
            st.dataframe(df_t, use_container_width=True)
        else:
            st.info("거래 내역이 없거나 데이터를 불러올 수 없습니다.")
    except:
        st.info("거래 내역 로드 중 오류가 발생했습니다.")

# --- 9. [새로 추가된 기능] 미체결 및 간편 재주문 탭 ---
with tab_wait:
    st.write("### ⏳ 미체결 주문 내역")
    try:
        unfilled_orders = upbit.get_order(selected_ticker) # 미체결 상태인 주문만 로드
        if not unfilled_orders:
            st.info("현재 미체결된 주문이 없습니다.")
        else:
            # 일괄 취소 버튼 (UI 상단 배치)
            col_header1, col_header2 = st.columns([5, 1])
            if col_header2.button("일괄취소", type="primary", use_container_width=True):
                for order in unfilled_orders:
                    upbit.cancel_order(order['uuid'])
                st.rerun()

            st.divider()

            for order in unfilled_orders:
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
                    side_label = "매수" if order['side'] == 'bid' else "매도"
                    side_class = "badge-bid" if order['side'] == 'bid' else "badge-ask"
                    
                    c1.write(f"**{order['created_at'][5:16]}**")
                    c2.markdown(f"<span class='{side_class}'>{side_label}</span> **{float(order['price']):,.0f} KRW**", unsafe_allow_html=True)
                    c3.write(f"{float(order['volume']):.4f} {coin_symbol}")
                    
                    if c4.button("취소", key=f"btn_can_{order['uuid']}", use_container_width=True):
                        upbit.cancel_order(order['uuid'])
                        st.rerun()
                    
                    # [간편 재주문 기능 - Expander로 구현]
                    with st.expander("간편 재주문"):
                        st.caption("기존 주문을 취소하고 새로운 조건으로 주문을 넣습니다.")
                        re_price = st.number_input("수정 주문 가격", value=float(order['price']), key=f"re_p_{order['uuid']}")
                        re_vol = st.number_input("수정 주문 수량", value=float(order['volume']), key=f"re_v_{order['uuid']}")
                        
                        col_re_btn = st.columns([1, 1])
                        if col_re_btn[0].button("취소 및 재주문", type="primary", key=f"re_exec_{order['uuid']}", use_container_width=True):
                            upbit.cancel_order(order['uuid'])
                            time.sleep(0.5) # 업비트 API 처리 시간 대기
                            if order['side'] == 'bid':
                                upbit.buy_limit_order(selected_ticker, re_price, re_vol)
                            else:
                                upbit.sell_limit_order(selected_ticker, re_price, re_vol)
                            st.rerun()
                st.divider()
    except Exception as e:
        st.error(f"미체결 목록 로드 실패: {e}")

# --- 10. 하단 상태 표시 (감시 중인 종목 리스트) ---
st.sidebar.divider()
try:
    conn = sqlite3.connect("upbit_trading.db")
    active_df = pd.read_sql("SELECT ticker as '종목', budget as '회당예산', CASE WHEN ai_mode=1 THEN 'AI모드' ELSE '수동모드' END as '모드' FROM user_settings WHERE is_active = 1", conn)
    conn.close()
    if not active_df.empty:
        st.sidebar.caption("📡 감시 중")
        st.sidebar.table(active_df)
except:
    pass