"""ALFRED(FRED 과거 vintage)에서 실제 발표치와 모든 수정 이력 수집. fredapi 사용.

필요: FRED API 키 (무료, https://fredaccount.stlouisfed.org/apikeys)
      프로젝트 루트 .env 파일에 FRED_API_KEY=... 로 저장하거나 환경변수로 설정

저장 위치:
  data/raw/alfred/{series_id}_vintages.csv     모든 발표·수정 (date, realtime_start, value)
  data/raw/alfred/{series_id}_first_release.csv  최초 발표치
  data/raw/release_schedule/{series_id}_release_dates.csv  vintage(발표) 날짜 목록

주의: CPI는 지수 수준이라 전월 대비 %는 clean 단계에서 계산해.
      이 값이 BLS 공식 발표 %(소수 1자리) 및 Kalshi 정산값과 같은지 반드시 대조해.

실행: python scraping/03_alfred.py
"""
from __future__ import annotations

import os

import pandas as pd
from fredapi import Fred

from common import RAW_DIR, get_logger, load_env_file, load_settings


def main() -> None:
    load_env_file()
    logger = get_logger("03_alfred")
    key = os.environ.get("FRED_API_KEY")
    if not key:
        logger.error("FRED_API_KEY가 없어. .env 파일이나 환경변수로 설정해")
        return
    fred = Fred(api_key=key)
    cfg = load_settings()
    series = cfg["alfred"]["series"]
    out_dir = RAW_DIR / "alfred"
    sched_dir = RAW_DIR / "release_schedule"
    out_dir.mkdir(parents=True, exist_ok=True)
    sched_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["sample"]["start_date"]) - pd.DateOffset(years=5)  # σ 계산용 과거 포함
    logger.info("ALFRED 수집 시작 | %d개 시리즈 | 관측 시작 %s", len(series), start.date())

    for i, (sid, desc) in enumerate(series.items(), 1):
        logger.info("[%d/%d] %s (%s)", i, len(series), sid, desc)
        try:
            allrel = fred.get_series_all_releases(sid)
            allrel = allrel[pd.to_datetime(allrel["date"]) >= start]
            allrel.to_csv(out_dir / f"{sid}_vintages.csv", index=False)
            logger.info("  전체 vintage %d행 저장", len(allrel))

            first = fred.get_series_first_release(sid)
            first = first[first.index >= start].rename("value").reset_index()
            first.to_csv(out_dir / f"{sid}_first_release.csv", index=False)
            logger.info("  최초 발표치 %d행 저장", len(first))

            vdates = pd.Series(fred.get_series_vintage_dates(sid), name="vintage_date")
            vdates = vdates[pd.to_datetime(vdates) >= start]
            vdates.to_csv(sched_dir / f"{sid}_release_dates.csv", index=False)
            logger.info("  발표일 %d개 저장", len(vdates))
        except Exception as e:  # 시리즈별로 실패해도 다음 시리즈 진행
            logger.error("  %s 실패: %s", sid, e)
    logger.info("ALFRED 수집 종료")


if __name__ == "__main__":
    main()
