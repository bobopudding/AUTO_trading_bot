import time
import pyupbit
import datetime
import sqlite3

# 1. API 키 설정
access = "YOUR_ACCESS_KEY"
secret = "YOUR_SECRET_KEY"
upbit = pyupbit.Upbit(access, secret)

def get_user_selected_tickers():
    """
    대시보드에서 사용자가 체크박스를 선택한 종목들만 DB에서 가져옵니다.
    (테이블명: user_settings, 컬럼: ticker, is_active)
    """
    try:
        conn = sqlite3.connect("upbit_trading.db")
        cur = conn.cursor()
        # 사용자가 자동매매 'ON' (is_active=1)으로 설정한 종목만 조회
        cur.execute("SELECT ticker FROM user_settings WHERE is_active = 1")
        rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        # DB가 아직 없거나 설정 전일 경우를 대비한 예외 처리
        print(f"💡 [안내] DB에서 선택된 종목을 찾는 중입니다... ({e})")
        return []

def get_target_price(ticker, k=0.5):
    try:
        df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        if df is None: return None
        return df.iloc[1]['open'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    except: return None

print("🚀 [사용자 맞춤형] 자동 매매 엔진 가동 중...")

while True:
    try:
        now = datetime.datetime.now()
        
        # [핵심] 사용자가 대시보드에서 실시간으로 체크/해제한 종목 리스트를 가져옴
        selected_tickers = get_user_selected_tickers()
        
        if not selected_tickers:
            print(f"[{now}] 대시보드에서 선택된 종목이 없습니다. 대기 중...")
            time.sleep(10)
            continue

        balances = upbit.get_balances()

        for ticker in selected_tickers:
            current_price = pyupbit.get_current_price(ticker)
            
            # 내 잔고에 해당 종목이 있는지 확인
            coin_info = next((b for b in balances if b['currency'] == ticker.split('-')[1]), None)

            # --- 1. 보유 중인 종목: 사용자가 체크를 해제하면 팔거나, 손절 체크 ---
            if coin_info:
                avg_price = float(coin_info['avg_buy_price'])
                earning_rate = (current_price - avg_price) / avg_price
                
                # 손절 조건 (예: -3%)
                if earning_rate <= -0.03:
                    print(f"🚨 [자동 손절] {ticker} 손실 제한 도달!")
                    # upbit.sell_market_order(ticker, coin_info['balance'])
                
            # --- 2. 미보유 종목: 사용자가 체크한 종목에 대해 매수 타점 감시 ---
            else:
                target_price = get_target_price(ticker)
                if target_price and current_price > target_price:
                    print(f"🎯 [자동 매수] 사용자가 선택한 {ticker}가 목표가를 돌파했습니다!")
                    # upbit.buy_market_order(ticker, 5000) # 설정 금액만큼 매수

            time.sleep(0.2) # API 호출 제한 방지

        print(f"[{now}] 현재 감시 중인 사용자 선택 종목: {selected_tickers}")
        time.sleep(5)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        time.sleep(5)