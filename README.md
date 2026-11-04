# Kalshi 시장 내재 확률분포의 정보가치

- 연구 설정: config/settings.yaml
- 원본 데이터: data/raw (수정 금지) → 정제: data/interim → 분석 패널: data/processed
- 결과: outputs/tables, outputs/figures (데이터에서 다시 만들어지므로 저장소에는 올리지 않음), 실행 로그: outputs/logs
- 방법 문서: docs/methods_kalshi_distribution.md

## 실행 순서 (프로젝트 루트에서)
| 단계 | 명령 |
|---|---|
| 준비 | `pip install -r scraping/requirements.txt`, `.env`에 FRED·ECOS 키 |
| 수집 | `python scraping/01_kalshi.py` ~ `05_ecos_ktb.py`, `python scraping/07_move.py` |
| 정제 | `python src/clean/01_kalshi_distributions.py` → `python src/clean/02_releases.py` |
| 패널 | `python src/features/01_event_panel.py` |
| RQ1 | `python src/analysis/rq1_forecast.py` (그림 포함) |
| RQ3 | `python src/analysis/rq3_width.py` |
| RQ2 | `python src/analysis/rq2_rates.py` (RQ1 결과 필요) |
| 검증(선택) | `python tests/verify_against_r.py --r-repo <원본저장소>` (R 필요), `python scraping/06_dkw_reference.py` → `python tests/compare_with_dkw.py` |
