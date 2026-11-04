# 데이터 실현 가능성 확인 목록

## Kalshi
- [x] 시리즈 이름 확인: KXCPI, KXCPICORE, KXGDP, KXU3, KXPAYROLLS 모두 존재 (2026-11-04)
- [x] CPI 계약 구조: 상호배타적 bracket이 아니라 "X% 초과" threshold 계약(strike_type=greater).
      → P(A > x_j)의 생존함수로 읽고, 인접 threshold의 차이로 구간 확률을 만든다.
      → 합이 1이 되도록 정규화하는 방식은 쓰지 않는다. 대신 단조성 위반(높은 threshold가 더 비싼 경우)을 점검한다.
- [x] KXCPI 이벤트: 2023-01 정산분부터 49개 (과거분 ticker는 CPI-YYMMM 형식)
- [ ] 계약 규칙과 정산 기준 (data/raw/kalshi/rules/)
- [ ] 과거 일별 캔들의 접근 범위
- [x] 거래 없는 날: price.*_dollars가 NaN, yes_bid/yes_ask는 기록됨 → 체결가 대신 호가 중간값을 보조로 사용
- [x] 가격 단위: 달러(0~1). 일봉은 미국 동부시간 자정에 마감
- [x] 정산 기준: rules_primary에 "single-decimal" 명시 (BLS 발표 소수 1자리 기준)

## ForexFactory
- [ ] 403 차단 여부
- [ ] 표시 시간대
- [ ] 과거 forecast가 발표 당시 값인지

## 실제치
- [ ] ALFRED로 계산한 값과 BLS·BEA 발표치, Kalshi 정산값 일치 여부
- [ ] GDP 계약이 어느 발표 단계(속보·잠정)를 대상으로 하는지
