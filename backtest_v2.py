import sqlite3
import pandas as pd
import numpy as np

# [함수] 개별 종목 백테스트 로직
def run_backtest_v2(ticker, conn):
    query = f"SELECT * FROM price_logs WHERE ticker = '{ticker}' ORDER BY created_at ASC"
    df = pd.read_sql(query, conn)

    if len(df) < 10:
        return None

    k = 0.5
    df['range'] = (df['high'].shift(1) - df['low'].shift(1))
    df['target'] = df['price'].shift(1) + (df['range'] * k)
    
    # 수익률 계산
    df['ror'] = np.where(df['high'] > df['target'], df['price'] / df['target'], 1.0)
    df['hpr'] = df['ror'].cumprod()
    
    # MDD 계산
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
    
    final_ror = (df['hpr'].iloc[-1] - 1) * 100
    mdd = df['dd'].max()
    trade_count = int((df['ror'] != 1.0).sum())
    win_rate = ( (df['ror'] > 1.0).sum() / trade_count * 100 ) if trade_count > 0 else 0

    return (ticker, final_ror, mdd, win_rate, trade_count)

def save_all_results_to_db():
    conn = sqlite3.connect("upbit_trading.db")
    cur = conn.cursor()

    # 1. 결과 저장용 테이블 생성 (기존에 있으면 삭제 후 새로 생성 - 최신화)
    cur.execute("DROP TABLE IF EXISTS backtest_results")
    cur.execute("""
        CREATE TABLE backtest_results (
            ticker TEXT PRIMARY KEY,
            ror REAL,
            mdd REAL,
            win_rate REAL,
            trade_count INTEGER
        )
    """)

    # 2. 모든 종목 리스트 가져오기
    cur.execute("SELECT DISTINCT ticker FROM price_logs")
    tickers = [row[0] for row in cur.fetchall()]
    
    print(f"🚀 총 {len(tickers)}개 종목 전수 조사를 시작합니다...")

    # 3. 전 종목 계산 및 DB 삽입
    for i, ticker in enumerate(tickers):
        res = run_backtest_v2(ticker, conn)
        if res:
            cur.execute("""
                INSERT INTO backtest_results (ticker, ror, mdd, win_rate, trade_count)
                VALUES (?, ?, ?, ?, ?)
            """, res)
            
            if (i+1) % 20 == 0:
                print(f"✅ {i+1}번째 종목 분석 완료...")
    
    conn.commit()
    conn.close()
    print("\n✨ 모든 분석 결과가 'upbit_trading.db'의 'backtest_results' 테이블에 저장되었습니다!")

if __name__ == "__main__":
    save_all_results_to_db()