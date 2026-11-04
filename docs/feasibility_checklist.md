# 데이터 실현 가능성 확인 목록

## Kalshi
- [ ] 시리즈 이름 확인 (data/raw/kalshi/series_list.csv)
- [ ] CPI bracket 구조: 상호배타적이고 결과공간 전체를 덮는지 (strike_type, floor/cap_strike)
- [ ] 계약 규칙과 정산 기준 (data/raw/kalshi/rules/)
- [ ] 과거 일별 캔들의 접근 범위
- [ ] 거래 없는 날의 기록 방식, 거래량 0 처리
- [ ] 가격 단위 (센트 / 달러)

## ForexFactory
- [ ] 403 차단 여부
- [ ] 표시 시간대
- [ ] 과거 forecast가 발표 당시 값인지

## 실제치
- [ ] ALFRED로 계산한 값과 BLS·BEA 발표치, Kalshi 정산값 일치 여부
- [ ] GDP 계약이 어느 발표 단계(속보·잠정)를 대상으로 하는지
