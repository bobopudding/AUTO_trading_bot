# AUTO_trading_bot
Python-based automated cryptocurrency/stock trading bot using API
업비트(Upbit) API를 활용하여 실시간 시세 조회, 차트 분석, 그리고 AI 기반 변동성 돌파 전략을 통해 자동 매매를 수행하는 풀스택 트레이딩 대시보드 프로젝트입니다.

<img width="1264" height="668" alt="image" src="https://github.com/user-attachments/assets/4a4d7179-2a3b-4a32-8bdf-71db4e65b21d" />

---

1. 배경 및 필요성
감정 배제 트레이딩: 급변하는 암호화폐 시장에서 개인 투자자가 겪는 심리적 불안을 해소하고, 사전에 설정된 원칙에 따라 기계적인 매매를 수행합니다.

24시간 시장 대응: 물리적인 제약 없이 365일 24시간 실시간 모니터링 및 대응이 가능한 자동화 시스템이 필요합니다.

데이터 기반 의사결정: 단순한 감이 아닌, 과거 데이터를 통한 백테스트 결과와 실시간 보조지표(AI 타겟가)를 활용하여 투자 성공률을 높입니다.

---

2. 프로젝트 개요
본 프로젝트는 Streamlit 기반의 웹 대시보드와 SQLite3 데이터베이스를 결합한 트레이딩 시스템입니다. 사용자는 대시보드에서 종목별 자동매매 엔진을 제어할 수 있으며, 과거 누적된 데이터를 바탕으로 전략의 유효성을 즉각적으로 검증할 수 있습니다.

---

3. 주요 기능
실시간 트레이딩 대시보드: TradingView 차트와 실시간 호가창을 연동하여 전문적인 매매 환경 제공.
AI 자동 감시 모드: 변동성 돌파 전략($K=0.5$)을 적용하여 최적의 매수/익절 타겟가를 자동 계산 및 집행.
고도화된 주문 관리: 즉시 매수/매도뿐만 아니라 '미체결 주문' 리스트 조회 및 간편 재주문(취소 후 신규 주문) 기능 탑재.
전수 종목 백테스팅: DB에 저장된 1년치 역사적 데이터를 활용하여 전 종목의 수익률, MDD, 승률을 분석하고 결과 저장.
자산 현황 모니터링: 보유 자산의 총 평가 금액 및 수익률을 실시간 메트릭으로 시각화.

---

4. 프로젝트 차별성
데이터 적재 기반의 분석: 실시간 수집 방식 대신, 기존의 역사적 로그를 DB에 일괄 로드(collect_all_history)하여 백테스팅에 학습/활용하는 방식을 채택하여 분석 속도와 신뢰도를 확보했습니다.

사용자 정의 리스크 관리: 종목별로 손절(Stop-loss), 익절(Target-profit), 일일 매매 한도를 각각 설정하여 안전 장치를 마련했습니다.

하이브리드 제어: 완전 자동화(AI 모드)와 사용자 설정 수동 모드를 자유롭게 전환하며 전략을 운용할 수 있습니다.

---
5. 기술 스택 (Tech Stack)
환경 (Environment)
Language: Python 3.x

Library: pyupbit (API Wrapper), python-dotenv (환경 변수 관리)

프론트엔드 (Frontend)
Framework: Streamlit

Components: Streamlit Components V1 (TradingView 차트 임베딩)

Styling: Custom CSS (White-themed UI, Responsive Layout)

백엔드 & DB (Backend & Database)
Database: SQLite3 (유저 설정 및 거래 기록, 시세 로그 저장)

Data Analysis: Pandas, NumPy (백테스트 및 통계 계산)

Concurrency: Threading (Streamlit 메인 루프와 별개로 트레이딩 엔진 백그라운드 상시 구동)

