import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import time
import streamlit.components.v1 as components
from datetime import datetime
import threading
import os
import requests
from dotenv import load_dotenv
import warnings

# --- 0. 경고 메시지 차단 및 로그 함수 추가 ---
warnings.filterwarnings("ignore")

def log_trade(msg):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("trading_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{now}] {msg}\n")

# --- 1. API 키 및 초기 설정 (.env 파일 로드) ---
load_dotenv()
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")

upbit = pyupbit.Upbit(access, secret)

def init_db():
    conn = sqlite3.connect("upbit_trading.db")
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            ticker TEXT PRIMARY KEY,
            is_active INTEGER DEFAULT 0,
            budget INTEGER DEFAULT 5000,
            stop_loss REAL DEFAULT 0.03,
            max_daily_buy INTEGER DEFAULT 100000,
            max_daily_sell INTEGER DEFAULT 100000,
            target_profit REAL DEFAULT 0.05,
            ai_mode INTEGER DEFAULT 0,
            bot_type TEXT DEFAULT 'BUY'
        )
    """)
    
    cur.execute("PRAGMA table_info(user_settings)")
    columns = [column[1] for column in cur.fetchall()]
    
    required_columns = {
        "max_daily_buy": "INTEGER DEFAULT 100000",
        "max_daily_sell": "INTEGER DEFAULT 100000",
        "target_profit": "REAL DEFAULT 0.05",
        "ai_mode": "INTEGER DEFAULT 0",
        "bot_type": "TEXT DEFAULT 'BUY'"
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

init_db()

def check_order_result(res):
    if res is None:
        return False, "업비트 서버로부터 응답이 없습니다. (API 키 확인)"
    if isinstance(res, dict) and 'error' in res:
        err_msg = res.get('error', {}).get('message', '알 수 없는 오류')
        return False, f"업비트 거절 사유: {err_msg}"
    if isinstance(res, dict) and 'uuid' in res:
        return True, "실제 업비트 주문 접수 성공!"
    return False, f"비정상 응답 발생: {res}"

# [퀀트 알고리즘 핵심] 현재가 트레일링 스탑 & 프랙탈 기반 동적 타점
def get_ai_target_prices(ticker, avg_buy_p=0):
    try:
        # 1. 거시적 맥락 분석을 위한 과거 1년 데이터 로드
        df_day = pyupbit.get_ohlcv(ticker, interval="day", count=365)
        if df_day is None or len(df_day) < 30:
            return None, None, None, False, 0
            
        # 1-1. 백테스트 지표 실시간 산출 (승률, MDD)
        df_day['range_d'] = (df_day['high'] - df_day['low']) * 0.5
        df_day['target_d'] = df_day['open'] + df_day['range_d'].shift(1)
        df_day['ror_d'] = np.where(df_day['high'] > df_day['target_d'], df_day['close'] / df_day['target_d'], 1.0)
        df_day['hpr_d'] = df_day['ror_d'].cumprod()
        df_day['dd_d'] = (df_day['hpr_d'].cummax() - df_day['hpr_d']) / df_day['hpr_d'].cummax()
        
        mdd_risk = df_day['dd_d'].max()
        trade_count = len(df_day[df_day['ror_d'] != 1.0])
        win_rate = (df_day['ror_d'] > 1.0).sum() / trade_count if trade_count > 0 else 0.5
        
        # 1-2. 프랙탈 패턴 매칭 (현재 5일과 가장 비슷한 과거 패턴 탐색)
        recent_5 = df_day['close'].iloc[-5:].values
        norm_recent = (recent_5 - recent_5.min()) / (recent_5.max() - recent_5.min() + 1e-9)
        
        best_sim = -1
        expected_future_return = 0
        
        for i in range(len(df_day) - 10):
            past_5 = df_day['close'].iloc[i:i+5].values
            norm_past = (past_5 - past_5.min()) / (past_5.max() - past_5.min() + 1e-9)
            dist = np.sum((norm_recent - norm_past)**2)
            sim = 1 / (1 + dist)
            
            if sim > best_sim:
                best_sim = sim
                expected_future_return = (df_day['close'].iloc[i+8] - df_day['close'].iloc[i+4]) / df_day['close'].iloc[i+4]

        # 2. 미세 타점(단기 모멘텀)을 위한 60분봉 분석
        df_min = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
        if df_min is None or len(df_min) < 20:
            return None, None, None, False, 0
            
        # 2-1. ATR (순수 변동성) 및 시장 노이즈 비율 계산
        high_low = df_min['high'] - df_min['low']
        high_close = (df_min['high'] - df_min['close'].shift()).abs()
        low_close = (df_min['low'] - df_min['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        direction = (df_min['close'] - df_min['open']).abs()
        volatility = (df_min['high'] - df_min['low']).replace(0, 0.0001)
        noise = 1.0 - (direction / volatility)
        k_val = noise.rolling(20).mean().iloc[-1]
        k_val = max(0.2, min(k_val, 0.8))
        
        curr_p = pyupbit.get_current_price(ticker)
        prev_candle = df_min.iloc[-2]
        curr_candle = df_min.iloc[-1]
        
        # 3. AI 목표 매수가 계산
        pattern_weight = 1.0 - max(-0.5, min(expected_future_return, 0.5))
        target_buy = curr_candle['open'] + (prev_candle['high'] - prev_candle['low']) * k_val * pattern_weight
        
        # 4. 백테스트 데이터를 반영한 [리스크 조정 ATR 계수]
        profit_multi = 1.5 + (win_rate - 0.5) + (expected_future_return * 2)
        profit_multi = max(1.0, min(profit_multi, 3.0)) 
        
        loss_multi = 1.0 - (mdd_risk * 0.5) 
        loss_multi = max(0.5, min(loss_multi, 1.5)) 
        
        # 5. [핵심 모순 해결] 현재가 기반 동적 트레일링 타겟 연산
        if avg_buy_p == 0:
            # 매수 전: 계획된 매수 타점을 중심으로 가상의 익/손절 라인 형성
            target_sell = target_buy + (atr * profit_multi)
            target_sl = target_buy - (atr * loss_multi)
        else:
            # 매수 후(보유 중): 무조건 '현재가'를 중심으로 익/손절을 실시간 트레일링 적용
            target_sell = curr_p + (atr * profit_multi)
            target_sl = curr_p - (atr * loss_multi)
            
            # 단, 급등락에 의해 손절가가 현재가를 찌르는 오류 방지용 안전장치
            if target_sl >= curr_p:
                target_sl = curr_p * 0.995
        
        ma15 = df_min['close'].rolling(15).mean().iloc[-1]
        trend_ok = (curr_p >= ma15) and (expected_future_return > -0.03)
        
        return target_buy, target_sell, target_sl, trend_ok, expected_future_return
    except:
        return None, None, None, False, 0

# [기존 1년 백테스트 분석 - UI 요약 표시용 (단기 60분봉 기반)]
@st.cache_data(ttl=3600)
def get_backtest_report(ticker):
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=720)
        if df is None or len(df) < 100:
            return None
            
        df['ma15'] = df['close'].rolling(15).mean()
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        direction = (df['close'] - df['open']).abs()
        volatility = (df['high'] - df['low']).replace(0, 0.0001)
        df['noise'] = 1.0 - (direction / volatility)
        df['k_val'] = df['noise'].rolling(20).mean().clip(0.2, 0.8)
        
        df['range'] = df['high'] - df['low']
        df['target'] = df['open'] + df['range'].shift(1) * df['k_val'].shift(1)
        
        def calc_ror(row):
            if pd.isna(row['target']) or pd.isna(row['atr']) or pd.isna(row['ma15']):
                return 1.0
            if row['high'] > row['target'] and row['open'] > row['ma15']:
                tp = row['target'] + (row['atr'] * 1.5)
                sl = row['target'] - (row['atr'] * 1.0)
                if row['low'] <= sl:
                    return (sl / row['target']) - 0.001
                elif row['high'] >= tp:
                    return (tp / row['target']) - 0.001
                else:
                    return (row['close'] / row['target']) - 0.001
            return 1.0

        df['ror'] = df.apply(calc_ror, axis=1)
        df['hpr'] = df['ror'].cumprod()
        df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
        
        total_ror = (df['hpr'].iloc[-1] - 1) * 100
        mdd = df['dd'].max()
        trade_count = len(df[df['ror'] != 1.0])
        win_rate = (df['ror'] > 1.0).sum() / trade_count * 100 if trade_count > 0 else 0
        
        return {
            "수익률": total_ror,
            "MDD": mdd,
            "승률": win_rate,
            "거래횟수": trade_count
        }
    except:
        return None

# --- 3. 실시간 자동매매 엔진 ---
def trading_engine():
    while True:
        try:
            conn = sqlite3.connect("upbit_trading.db")
            active_bots = pd.read_sql("SELECT * FROM user_settings WHERE is_active = 1 OR ai_mode = 1", conn)
            conn.close()

            for _, bot in active_bots.iterrows():
                ticker = bot['ticker']
                bot_type = bot.get('bot_type', 'BUY')
                curr_p = pyupbit.get_current_price(ticker)
                avg_buy_p = upbit.get_avg_buy_price(ticker)
                
                # [매수 봇 로직]
                if bot_type == 'BUY':
                    if bot['ai_mode'] == 1:
                        ai_buy_p, ai_sell_p, ai_sl_p, trend_ok, _ = get_ai_target_prices(ticker, avg_buy_p)
                        if ai_buy_p and avg_buy_p == 0 and curr_p >= ai_buy_p and trend_ok:
                            krw_bal = upbit.get_balance("KRW")
                            if krw_bal >= bot['budget'] and bot['budget'] >= 5000:
                                res = upbit.buy_market_order(ticker, bot['budget'])
                                # 로그 추가
                                log_trade(f"🛒 [AI 매수실행] 종목: {ticker} | 체결가: {curr_p:,.0f} | AI 목표가: {ai_buy_p:,.0f}")
                                
                        elif avg_buy_p > 0:
                            coin_bal = upbit.get_balance(ticker)
                            if coin_bal > 0:
                                # 익절 로그 추가
                                if curr_p >= ai_sell_p:
                                    upbit.sell_market_order(ticker, coin_bal)
                                    log_trade(f"💰 [AI 익절완료] 종목: {ticker} | 매도가: {curr_p:,.0f} | AI 익절가: {ai_sell_p:,.0f} | 평단: {avg_buy_p:,.0f}")
                                # 손절 로그 추가
                                elif curr_p <= ai_sl_p:
                                    upbit.sell_market_order(ticker, coin_bal)
                                    log_trade(f"📉 [AI 손절완료] 종목: {ticker} | 매도가: {curr_p:,.0f} | AI 손절가: {ai_sl_p:,.0f} | 평단: {avg_buy_p:,.0f}")
                    
                    elif bot['is_active'] == 1 and avg_buy_p > 0:
                        current_ror = (curr_p / avg_buy_p) - 1
                        if current_ror <= -bot['stop_loss'] or current_ror >= bot['target_profit']:
                            coin_bal = upbit.get_balance(ticker)
                            if coin_bal > 0: 
                                upbit.sell_market_order(ticker, coin_bal)
                                log_trade(f"📢 [수동봇 매도] 종목: {ticker} | 매도가: {curr_p:,.0f} | 평단: {avg_buy_p:,.0f}")

                # [매도 봇 로직]
                elif bot_type == 'SELL':
                    if bot['ai_mode'] == 1:
                        _, ai_sell_p, ai_sl_p, _, _ = get_ai_target_prices(ticker, avg_buy_p)
                        if ai_sell_p and avg_buy_p > 0:
                            coin_bal = upbit.get_balance(ticker)
                            if coin_bal > 0:
                                if curr_p >= ai_sell_p:
                                    upbit.sell_market_order(ticker, coin_bal)
                                    log_trade(f"💰 [AI 익절완료(SELL봇)] 종목: {ticker} | 매도가: {curr_p:,.0f} | AI 익절가: {ai_sell_p:,.0f}")
                                elif curr_p <= ai_sl_p:
                                    upbit.sell_market_order(ticker, coin_bal)
                                    log_trade(f"📉 [AI 손절완료(SELL봇)] 종목: {ticker} | 매도가: {curr_p:,.0f} | AI 손절가: {ai_sl_p:,.0f}")
                    
                    elif bot['is_active'] == 1 and avg_buy_p > 0:
                        current_ror = (curr_p / avg_buy_p) - 1
                        if current_ror <= -bot['stop_loss'] or current_ror >= bot['target_profit']:
                            coin_bal = upbit.get_balance(ticker)
                            if coin_bal > 0: 
                                upbit.sell_market_order(ticker, coin_bal)
                                log_trade(f"📢 [수동봇 매도] 종목: {ticker} | 매도가: {curr_p:,.0f}")
            
            time.sleep(1)
        except Exception as e:
            time.sleep(5)

if 'engine_thread' not in st.session_state:
    thread = threading.Thread(target=trading_engine, daemon=True)
    thread.start()
    st.session_state['engine_thread'] = True

# --- 4. 설정 로드 다이얼로그 (Dialog) ---
@st.dialog("⚙️ 자동매매 설정 로드")
def load_config_dialog(ticker):
    st.write(f"### {ticker} 종목의 저장된 세팅")
    
    curr_p = pyupbit.get_current_price(ticker)
    avg_buy_p = upbit.get_avg_buy_price(ticker)
    ai_buy, ai_sell, ai_sl, trend_ok, exp_ret = get_ai_target_prices(ticker, avg_buy_p)
    
    conn = sqlite3.connect("upbit_trading.db")
    cfg = pd.read_sql("SELECT * FROM user_settings WHERE ticker = ?", conn, params=(ticker,))
    conn.close()
    
    if not cfg.empty:
        row = cfg.iloc[0]
        bot_type = row.get('bot_type', 'BUY')
        st.info(f"이전에 저장한 [{ '매수' if bot_type == 'BUY' else '매도' }] 설정을 불러왔습니다.")
        
        db_ai = True if row['ai_mode'] == 1 else False
        new_ai = st.toggle("✨ AI 자동 감시 모드 활성화", value=db_ai, key="diag_ai_toggle_early")
        
        coin_symbol = ticker.split("-")[1]
        
        if bot_type == 'BUY':
            b_price = st.number_input("매수 가격(KRW)", value=int(curr_p), disabled=new_ai, key="diag_bp")
            default_qty = float(row['budget'] / curr_p) if curr_p > 0 else 0.1
            b_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=default_qty, format="%.4f", disabled=new_ai, key="diag_bq")
            
            order_total_cost = int(b_price * b_qty)
            st.write(f"➔ 예상 결제 금액: **{order_total_cost:,.0f}** KRW")
            
            new_max_buy = st.number_input("일일 매수 한도 (KRW) - (AI 전용)", value=int(row['max_daily_buy']), step=10000, disabled=not new_ai)
            new_max_sell = int(row['max_daily_sell'])
            final_budget = order_total_cost if not new_ai else int(row['budget'])
            
        else:
            s_price = st.number_input("매도 가격(KRW)", value=int(curr_p), disabled=new_ai, key="diag_sp")
            default_qty = float(row['budget'] / curr_p) if curr_p > 0 else 0.1
            s_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=default_qty, format="%.4f", disabled=new_ai, key="diag_sq")
            
            order_sell_total = int(s_price * s_qty)
            st.write(f"➔ 예상 수령 금액: **{order_sell_total:,.0f}** KRW")
            
            new_max_buy = int(row['max_daily_buy'])
            new_max_sell = st.number_input("일일 매도 한도 (KRW) - (AI 전용)", value=int(row['max_daily_sell']), step=10000, disabled=not new_ai)
            final_budget = order_sell_total if not new_ai else int(row['budget'])
        
        st.divider()
        
        if new_ai and ai_buy:
            trend_str = f"🟢 상승장 (매수 허용)" if trend_ok else f"🔴 하락장 (매수 보류)"
            sim_str = "상승 기대" if exp_ret > 0 else "하락 우려"
            
            if bot_type == 'BUY':
                st.info(f"📍 **AI 매수가:** {ai_buy:,.0f} / **AI 익절가:** {ai_sell:,.0f} / **AI 손절가:** {ai_sl:,.0f}\n\n📊 모멘텀 필터: {trend_str}\n🧠 프랙탈 예측: {sim_str} ({exp_ret*100:+.2f}%)")
            else:
                st.info(f"📍 **AI 익절가:** {ai_sell:,.0f} / **AI 손절가:** {ai_sl:,.0f}\n\n📊 모멘텀 필터: {trend_str}\n🧠 프랙탈 예측: {sim_str} ({exp_ret*100:+.2f}%)")
            st.caption("※ AI 모드 작동 중에는 엔진 가동 상태 및 수동 익절/손절/입력창이 비활성화됩니다.")
        
        is_disabled = new_ai
        new_active = st.checkbox("이 종목 엔진 가동", value=True if row['is_active'] == 1 else False, disabled=is_disabled, key="diag_active_chk")
        new_sl = st.slider("손절 제한 (%)", 1, 20, int(row['stop_loss'] * 100), disabled=is_disabled)
        new_tp = st.slider("익절 목표 (%)", 1, 50, int(row['target_profit'] * 100), disabled=is_disabled)
        
        if st.button("✅ 설정 업데이트 및 엔진 재가동", use_container_width=True, type="primary"):
            if bot_type == 'BUY' and not new_ai and final_budget < 5000:
                st.error("❌ 매수 금액(예상 결제 금액)은 최소 5,000원 이상이어야 합니다.")
            else:
                conn = sqlite3.connect("upbit_trading.db")
                conn.cursor().execute("""
                    UPDATE user_settings 
                    SET budget=?, stop_loss=?, max_daily_buy=?, max_daily_sell=?, target_profit=?, ai_mode=?, is_active=?
                    WHERE ticker=?
                """, (final_budget, new_sl/100, new_max_buy, new_max_sell, new_tp/100, 1 if new_ai else 0, 1 if new_active else 0, ticker))
                conn.commit()
                conn.close()
                st.toast(f"{ticker} 설정 업데이트 완료!")
                time.sleep(0.5)
                st.rerun()
    else:
        st.error("해당 종목의 저장된 설정이 없습니다.")

# --- 5. UI 설정 및 스타일 ---
st.set_page_config(page_title="Quant Trading Bot v4", layout="wide")

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
    .min-order-alert { background-color: #fff9db; border: 1px solid #ffe066; color: #f08c00; padding: 10px; border-radius: 5px; font-weight: bold; margin-bottom: 15px; text-align: center; }
    .trend-card { border: 1px solid #f1f1f4; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. 사이드바 ---
all_tickers = pyupbit.get_tickers(fiat="KRW")

if 'main_ticker' not in st.session_state:
    st.session_state['main_ticker'] = "KRW-BTC"

def on_ticker_change():
    st.session_state['main_ticker'] = st.session_state['sb_ticker_key']

selected_ticker = st.sidebar.selectbox(
    "🎯 종목 선택", 
    all_tickers, 
    index=all_tickers.index(st.session_state['main_ticker']), 
    key="sb_ticker_key",
    on_change=on_ticker_change
)

current_view_ticker = st.session_state['main_ticker']
coin_symbol = current_view_ticker.split("-")[1]

st.sidebar.divider()
st.sidebar.subheader("💰 자산 현황")

try:
    balances = upbit.get_balances()
    total_buy_cash = 0.0     
    total_eval_cash = 0.0    
    krw_balance = 0.0        
    
    for b in balances:
        if b['currency'] == "KRW":
            krw_balance = float(b['balance'])
        else:
            t_ticker = f"KRW-{b['currency']}"
            c_price = pyupbit.get_current_price(t_ticker)
            if c_price:
                avg_buy_p = float(b['avg_buy_price'])
                amount = float(b['balance']) + float(b['locked'])
                total_buy_cash += avg_buy_p * amount
                total_eval_cash += c_price * amount

    total_profit_val = total_eval_cash - total_buy_cash
    total_profit_rate = ((total_eval_cash / total_buy_cash) - 1) * 100 if total_buy_cash > 0 else 0.0
    
    st.sidebar.metric("보유 KRW", f"{krw_balance:,.0f} KRW")
    st.sidebar.metric("총 평가손익", f"{total_profit_val:+,.0f} KRW", delta=f"{total_profit_rate:+.2f}%")
    st.sidebar.metric("총 보유자산", f"{(krw_balance + total_eval_cash):,.0f} KRW")
except:
    st.sidebar.warning("자산 정보 로드 실패")

st.sidebar.divider()
st.sidebar.subheader("📊 백테스트 분석 (ATR 동적 모델)")
bt_res = get_backtest_report(current_view_ticker)

if bt_res:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("예상 수익률", f"{bt_res['수익률']:.1f}%")
    c2.metric("승률", f"{bt_res['승률']:.1f}%")
    st.sidebar.caption(f"최대 낙폭(MDD): {bt_res['MDD']:.1f}% / 거래: {bt_res['거래횟수']}회")
    
    if bt_res['수익률'] > 5: st.sidebar.success("✅ 단기 모멘텀 적합")
    elif bt_res['수익률'] < 0: st.sidebar.warning("⚠️ 변동성 리스크 주의")
else:
    st.sidebar.info("데이터 분석 중...")

st.sidebar.divider()
st.sidebar.subheader("📡 감시 엔진 관리")

try:
    conn = sqlite3.connect("upbit_trading.db")
    active_df_full = pd.read_sql("SELECT * FROM user_settings WHERE is_active = 1 OR ai_mode = 1", conn)
    
    if not active_df_full.empty:
        manage_target = st.sidebar.selectbox("수정/삭제할 종목", active_df_full['ticker'].tolist(), key="manage_target_box")
        m_col1, m_col2 = st.sidebar.columns(2)
        
        if m_col1.button("감시 삭제", use_container_width=True):
            conn.cursor().execute("UPDATE user_settings SET is_active = 0, ai_mode = 0 WHERE ticker = ?", (manage_target,))
            conn.commit(); st.toast(f"{manage_target} 감시 종료!"); time.sleep(0.5); st.rerun()
            
        if m_col2.button("설정 로드", use_container_width=True):
            st.session_state['main_ticker'] = manage_target
            load_config_dialog(manage_target)

        st.sidebar.caption("현재 가동 리스트")
        summary_view = active_df_full[['ticker', 'budget', 'ai_mode', 'bot_type']].copy()
        
        summary_view['구분'] = summary_view['bot_type'].apply(lambda x: "매수" if x == 'BUY' else "매도")
        summary_view['모드'] = summary_view['ai_mode'].apply(lambda x: "AI" if x==1 else "수동")
        
        st.sidebar.table(summary_view[['ticker', '구분', '모드', 'budget']].rename(columns={'ticker':'종목', 'budget':'예산'}))
    else:
        st.sidebar.info("가동 중인 엔진 없음")
    conn.close()
except:
    pass

st.sidebar.divider()
st.sidebar.subheader("🚀 실시간 AI 예상 (TOP 5)")
try:
    url = "https://api.upbit.com/v1/ticker?markets=" + ",".join(all_tickers)
    resp = requests.get(url, timeout=3).json()
    sorted_resp = sorted(resp, key=lambda x: x['signed_change_rate'], reverse=True)
    
    top_5 = sorted_resp[:5]
    bottom_5 = sorted_resp[-5:]
    
    c_top, c_bot = st.sidebar.columns(2)
    with c_top:
        st.markdown("**🔥 수익 유력**")
        for item in top_5:
            rate = item['signed_change_rate'] * 100
            st.markdown(f"<div class='trend-card'><span style='color:#d60000; font-weight:bold; font-size:0.8rem;'>{item['market'].split('-')[1]}<br>+{rate:.2f}%</span></div>", unsafe_allow_html=True)
            
    with c_bot:
        st.markdown("**❄️ 손실 위험**")
        for item in bottom_5:
            rate = item['signed_change_rate'] * 100
            st.markdown(f"<div class='trend-card'><span style='color:#0051c7; font-weight:bold; font-size:0.8rem;'>{item['market'].split('-')[1]}<br>{rate:.2f}%</span></div>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.warning("실시간 랭킹 로드 실패")

# --- 6. 실시간 데이터 전광판 ---
curr_price = pyupbit.get_current_price(current_view_ticker)
df_day = pyupbit.get_ohlcv(current_view_ticker, interval="day", count=2)
prev_close = df_day.iloc[0]['close']
change_val = curr_price - prev_close
change_rate = (change_val / prev_close) * 100
color_class = "up" if change_val >= 0 else "down"

st.markdown(f"""
<div class="header-box">
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <h2 style="margin: 0; color: #333; font-weight: 700;">{current_view_ticker}</h2>
        <h1 class="{color_class}" style="margin: 0; font-size: 3rem; letter-spacing: -1px;">{curr_price:,.0f}</h1>
        <div style="display: flex; flex-direction: column; line-height: 1.2;">
            <span class="{color_class}" style="font-size: 1.1rem;">{change_rate:+.2f}%</span>
            <span class="{color_class}" style="font-size: 1.1rem;">{change_val:+,f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 7. 메인 대시보드 ---
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
            st.info(f"📊 **{coin_symbol}** 퀀트 분석: 수익 **{bt_res['수익률']:.1f}%**, 승률 **{bt_res['승률']:.1f}%**")

        st.write("### 📊 호가")
        try:
            orderbook = pyupbit.get_orderbook(current_view_ticker)
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
        st.write(f"**현재가: {curr_price:,.0f} KRW**")
        st.markdown('<div class="min-order-alert">⚠️ 업비트 최소 주문 금액은 5,000 KRW입니다.</div>', unsafe_allow_html=True)
        
        current_avg_buy_p = upbit.get_avg_buy_price(current_view_ticker)
        ai_target_buy, ai_target_sell, ai_target_sl, trend_ok, exp_ret = get_ai_target_prices(current_view_ticker, current_avg_buy_p)
        
        buy_ai_key = f"buy_ai_{current_view_ticker}"
        sell_ai_key = f"sell_ai_{current_view_ticker}"
        
        conn = sqlite3.connect("upbit_trading.db")
        db_buy_cfg = pd.read_sql("SELECT * FROM user_settings WHERE ticker = ? AND bot_type = 'BUY'", conn, params=(current_view_ticker,))
        db_sell_cfg = pd.read_sql("SELECT * FROM user_settings WHERE ticker = ? AND bot_type = 'SELL'", conn, params=(current_view_ticker,))
        conn.close()
        
        if buy_ai_key not in st.session_state:
            st.session_state[buy_ai_key] = True if (not db_buy_cfg.empty and db_buy_cfg.iloc[0]['ai_mode'] == 1) else False
        if sell_ai_key not in st.session_state:
            st.session_state[sell_ai_key] = True if (not db_sell_cfg.empty and db_sell_cfg.iloc[0]['ai_mode'] == 1) else False

        o_tab1, o_tab2 = st.tabs(["매수", "매도"])
        
        # --- 매수 탭 ---
        with o_tab1:
            st.caption(f"💡 AI 감시가: {ai_target_buy:,.0f}" if ai_target_buy else "")
            
            is_buy_locked = st.session_state[buy_ai_key]
            b_price = st.number_input("매수 가격(KRW)", value=int(curr_price), key=f"bp_{current_view_ticker}", disabled=is_buy_locked)
            b_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=0.1, format="%.4f", key=f"bq_{current_view_ticker}", disabled=is_buy_locked)
            order_total_cost = int(b_price * b_qty)
            st.write(f"➔ 예상 결제 금액: **{order_total_cost:,.0f}** KRW")
            
            max_b_limit = st.number_input("일일 매수 한도(KRW) - AI 전용", min_value=0, value=100000, step=10000, key=f"mbl_{current_view_ticker}", disabled=not is_buy_locked)
            
            if st.button("즉시 매수", use_container_width=True, type="primary", disabled=is_buy_locked): 
                if order_total_cost < 5000: st.error("❌ 5,000원 이상이어야 합니다.")
                else:
                    res = upbit.buy_limit_order(current_view_ticker, b_price, b_qty)
                    success, msg = check_order_result(res)
                    if success: st.success(msg)
                    else: st.error(msg)
            
            with st.expander("🤖 자동매매(Bot) 상세 설정 [매수]", expanded=True):
                db_active = False; db_sl = 3; db_tp = 5
                if not db_buy_cfg.empty:
                    db_active = True if db_buy_cfg.iloc[0]['is_active'] == 1 else False
                    db_sl = int(db_buy_cfg.iloc[0]['stop_loss'] * 100)
                    db_tp = int(db_buy_cfg.iloc[0]['target_profit'] * 100)

                is_ai_mode = st.toggle("✨ AI 자동 감시 모드 활성화", key=buy_ai_key)
                
                if is_ai_mode and ai_target_buy:
                    trend_str = "🟢 상승장 (매수 허용)" if trend_ok else "🔴 하락장 (매수 보류)"
                    sim_str = "상승 기대" if exp_ret > 0 else "하락 우려"
                    st.info(f"📍 **AI 매수가:** {ai_target_buy:,.0f} / **AI 익절가:** {ai_target_sell:,.0f} / **AI 손절가:** {ai_target_sl:,.0f}\n\n📊 모멘텀 필터: {trend_str}\n🧠 프랙탈 예측: {sim_str} ({exp_ret*100:+.2f}%)")
                    st.caption("※ AI 모드 활성화 시 수동 입력창 및 익절/손절 설정이 잠깁니다.")

                is_active_bot = st.checkbox("이 종목 엔진 가동", value=db_active, key=f"buy_ab_{current_view_ticker}", disabled=is_ai_mode)
                slider_final_disabled = is_ai_mode or (not is_active_bot)
                st_loss = st.slider("손절(%)", 1, 15, db_sl, disabled=slider_final_disabled, key=f"buy_sl_{current_view_ticker}")
                target_profit = st.slider("익절(%)", 1, 50, db_tp, disabled=slider_final_disabled, key=f"buy_tp_{current_view_ticker}")

                if st.button("매수 봇 설정 저장 및 가동", key=f"buy_save_{current_view_ticker}"):
                    if not is_ai_mode and order_total_cost < 5000: 
                        st.error("❌ 회당 예산이 5,000원 이상이어야 합니다.")
                    else:
                        conn = sqlite3.connect("upbit_trading.db")
                        final_budget = order_total_cost if not is_ai_mode else (int(db_buy_cfg.iloc[0]['budget']) if not db_buy_cfg.empty else 10000)
                        conn.cursor().execute("""
                            INSERT OR REPLACE INTO user_settings 
                            (ticker, is_active, budget, stop_loss, max_daily_buy, max_daily_sell, target_profit, ai_mode, bot_type) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (current_view_ticker, 1 if is_active_bot else 0, final_budget, st_loss/100, max_b_limit, 100000, target_profit/100, 1 if is_ai_mode else 0, 'BUY'))
                        conn.commit(); conn.close(); st.toast("매수 봇 저장 완료!"); time.sleep(0.5); st.rerun()

        # --- 매도 탭 ---
        with o_tab2:
            st.caption(f"💡 AI 익절가: {ai_target_sell:,.0f} / 손절가: {ai_target_sl:,.0f}" if ai_target_sell else "")
            
            is_sell_locked = st.session_state[sell_ai_key]
            s_price = st.number_input("매도 가격(KRW)", value=int(curr_price), key=f"sp_{current_view_ticker}", disabled=is_sell_locked)
            s_qty = st.number_input(f"주문 수량({coin_symbol})", min_value=0.0001, value=0.1, format="%.4f", key=f"sq_{current_view_ticker}", disabled=is_sell_locked)
            order_sell_total = int(s_price * s_qty)
            st.write(f"➔ 예상 수령 금액: **{order_sell_total:,.0f}** KRW")
            
            max_s_limit = st.number_input("일일 매도 한도(KRW) - AI 전용", min_value=0, value=100000, step=10000, key=f"msl_{current_view_ticker}", disabled=not is_sell_locked)
            
            if st.button("즉시 매도", use_container_width=True, disabled=is_sell_locked): 
                if order_sell_total < 5000: st.error("❌ 5,000원 이상이어야 합니다.")
                else:
                    res = upbit.sell_limit_order(current_view_ticker, s_price, s_qty)
                    success, msg = check_order_result(res)
                    if success: st.success(msg)
                    else: st.error(msg)
            
            with st.expander("🤖 자동매매(Bot) 상세 설정 [매도]", expanded=True):
                db_active = False; db_sl = 3; db_tp = 5
                if not db_sell_cfg.empty:
                    db_active = True if db_sell_cfg.iloc[0]['is_active'] == 1 else False
                    db_sl = int(db_sell_cfg.iloc[0]['stop_loss'] * 100)
                    db_tp = int(db_sell_cfg.iloc[0]['target_profit'] * 100)

                is_ai_mode = st.toggle("✨ AI 자동 감시 모드 활성화", key=sell_ai_key)
                
                if is_ai_mode and ai_target_buy:
                    trend_str = "🟢 상승장 (매수 허용)" if trend_ok else "🔴 하락장 (매수 보류)"
                    sim_str = "상승 기대" if exp_ret > 0 else "하락 우려"
                    st.info(f"📍 **AI 익절가:** {ai_target_sell:,.0f} / **AI 손절가:** {ai_target_sl:,.0f}\n\n📊 모멘텀 필터: {trend_str}\n🧠 프랙탈 예측: {sim_str} ({exp_ret*100:+.2f}%)")
                    st.caption("※ 매도 봇은 보유 물량에 대해 익절/손절만 수행하며 수동 입력이 잠깁니다.")

                is_active_bot = st.checkbox("이 종목 엔진 가동", value=db_active, key=f"sell_ab_{current_view_ticker}", disabled=is_ai_mode)
                slider_final_disabled = is_ai_mode or (not is_active_bot)
                st_loss = st.slider("손절(%)", 1, 15, db_sl, disabled=slider_final_disabled, key=f"sell_sl_{current_view_ticker}")
                target_profit = st.slider("익절(%)", 1, 50, db_tp, disabled=slider_final_disabled, key=f"sell_tp_{current_view_ticker}")

                if st.button("매도 봇 설정 저장 및 가동", key=f"sell_save_{current_view_ticker}"):
                    conn = sqlite3.connect("upbit_trading.db")
                    conn.cursor().execute("""
                        INSERT OR REPLACE INTO user_settings 
                        (ticker, is_active, budget, stop_loss, max_daily_buy, max_daily_sell, target_profit, ai_mode, bot_type) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (current_view_ticker, 1 if is_active_bot else 0, 5000, st_loss/100, 100000, max_s_limit, target_profit/100, 1 if is_ai_mode else 0, 'SELL'))
                    conn.commit(); conn.close(); st.toast("매도 봇 저장 완료!"); time.sleep(0.5); st.rerun()

with tab_history:
    st.write(f"### 📜 {current_view_ticker} 거래 내역")
    h_filter_side = st.selectbox("🔄 거래 구분 필터", ["전체", "매수", "매도"], index=0, key=f"h_filter_{current_view_ticker}")
    try:
        trades = upbit.get_order(current_view_ticker, state='done')
        if trades:
            df_t = pd.DataFrame(trades)[['market', 'created_at', 'side', 'price', 'volume']]
            if h_filter_side == "매수": df_t = df_t[df_t['side'] == 'bid']
            elif h_filter_side == "매도": df_t = df_t[df_t['side'] == 'ask']
            df_t['side'] = df_t['side'].replace({'bid': '🔵 매수', 'ask': '🔴 매도'})
            df_t.columns = ['코인', '거래시간', '종류', '거래단가', '거래수량']
            st.dataframe(df_t, use_container_width=True)
        else: st.info(f"{current_view_ticker}의 완료된 거래 내역이 없습니다.")
    except Exception as e: st.error(f"거래 내역 로드 중 오류: {e}")

with tab_wait:
    st.write(f"### ⏳ {current_view_ticker} 미체결 주문")
    try:
        unfilled_orders = upbit.get_order(current_view_ticker) 
        if not unfilled_orders: st.info(f"현재 {current_view_ticker} 종목에 미체결 주문이 없습니다.")
        else:
            for order in unfilled_orders:
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
                    side_label = "매수" if order['side'] == 'bid' else "매도"
                    side_class = "badge-bid" if order['side'] == 'bid' else "badge-ask"
                    c1.write(f"**{order['market']}**\n{order['created_at'][5:16]}")
                    c2.markdown(f"<span class='{side_class}'>{side_label}</span> **{float(order['price']):,.0f} KRW**", unsafe_allow_html=True)
                    c3.write(f"{float(order['volume']):.4f} 수량")
                    if c4.button("주문 취소", key=f"wait_can_{order['uuid']}", use_container_width=True):
                        res = upbit.cancel_order(order['uuid'])
                        if res:
                            st.success(f"취소 성공!"); time.sleep(0.5); st.rerun()
                    with st.expander("간편 재주문"):
                        re_price = st.number_input("수정 가격", value=float(order['price']), key=f"re_p_{order['uuid']}")
                        re_vol = st.number_input("수정 수량", value=float(order['volume']), key=f"re_v_{order['uuid']}")
                        if st.button("취소 및 재주문 실행", type="primary", key=f"re_exec_{order['uuid']}", use_container_width=True):
                            upbit.cancel_order(order['uuid'])
                            time.sleep(0.5) 
                            if order['side'] == 'bid': upbit.buy_limit_order(order['market'], re_price, re_vol)
                            else: upbit.sell_limit_order(order['market'], re_price, re_vol)
                            st.rerun()
                st.divider()
    except Exception as e: st.error(f"미체결 목록 로드 실패: {e}")