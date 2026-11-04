# 데이터 수집 코드

## 준비
1. 패키지 설치 (프로젝트 루트에서)
   ```
   pip install -r scraping/requirements.txt
   ```
2. API 키를 프로젝트 루트의 `.env` 파일에 저장 (`.env.example` 참고)
   - FRED: https://fredaccount.stlouisfed.org/apikeys (무료)
   - ECOS: https://ecos.bok.or.kr/api/ (무료)
   - Kalshi 시장 데이터는 공개 엔드포인트라 키가 필요 없어

## 실행 순서 (프로젝트 루트에서)
| 순서 | 명령 | 결과 위치 |
|---|---|---|
| 1 | `python scraping/01_kalshi.py --series KXCPI --limit 5` (테스트) → 이상 없으면 `python scraping/01_kalshi.py` | data/raw/kalshi/ |
| 2 | `python scraping/02_forexfactory.py --start 2025-10-01 --end 2025-10-31` (테스트) → 전체 | data/raw/forexfactory/ |
| 3 | `python scraping/03_alfred.py` | data/raw/alfred/, data/raw/release_schedule/ |
| 4 | `python scraping/04_rates_fred.py` | data/raw/rates/ |
| 5 | `python scraping/05_ecos_ktb.py` | data/raw/rates/ |

- 모든 스크립트는 단계별 진행 상황을 화면과 `outputs/logs/` 에 기록해.
- 01번은 이미 받은 계약 파일을 건너뛰어서, 중간에 끊겨도 다시 실행하면 이어서 받아.
- MOVE 지수는 갖고 있는 파일을 `data/raw/move/move_daily.csv` 로 넣어줘.

## 수집 후 꼭 확인할 것 (docs/feasibility_checklist.md)
- `data/raw/kalshi/series_list.csv` 에서 시리즈 이름이 settings.yaml과 맞는지
- Kalshi 캔들의 가격 단위(센트/달러)와 거래 없는 날의 기록 방식
- ForexFactory: 403 차단 여부, 시간대, 과거 forecast의 시점
- ALFRED로 계산한 CPI 전월비가 BLS 발표치·Kalshi 정산값과 같은지

## 사용한 라이브러리 (GitHub 검색 결과)
- fredapi (mortada/fredapi): FRED·ALFRED, 최초 발표치와 vintage 조회 기능 제공
- ForexFactory 파싱: gist pohzipohzi/ad7942fc… 의 HTML 표 파싱 방식 참고
- Kalshi, ECOS: 공식 REST API를 requests로 직접 호출 (서드파티 래퍼 의존을 줄이기 위해)
