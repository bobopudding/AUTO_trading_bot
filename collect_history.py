import pyupbit
import sqlite3
import time

def collect_all_history():
    conn = sqlite3.connect("upbit_trading.db")
    cur = conn.cursor()
    
    # [중요!] 테이블이 없으면 자동으로 생성하는 로직 추가
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            price REAL,
            high REAL,
            low REAL,
            volume REAL,
            created_at TEXT
        )
    """)
    conn.commit()

    tickers = pyupbit.get_tickers(fiat="KRW")
    print(f"📂 총 {len(tickers)}개 종목의 1년치 역사적 데이터 수집을 시작합니다.")

    for i, ticker in enumerate(tickers):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=365)
            
            if df is not None:
                for index, row in df.iterrows():
                    str_date = str(index) 
                    cur.execute("""
                        INSERT INTO price_logs (ticker, price, high, low, volume, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (ticker, row['close'], row['high'], row['low'], row['volume'], str_date))
                
                conn.commit()
                print(f"[{i+1}/{len(tickers)}] {ticker} 적재 완료")
            
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"❌ {ticker} 수집 중 오류 발생: {e}")

    conn.close()
    print("✨ 이제 깨끗한 DB에 1년치 데이터가 가득 찼습니다!")

if __name__ == "__main__":
    collect_all_history()