"""FRED에서 미국채 일별 수익률(DGS2, DGS10) 수집. fredapi 사용.

필요: FRED_API_KEY (.env 또는 환경변수)
저장: data/raw/rates/fred_{series_id}.csv
실행: python scraping/04_rates_fred.py
"""
from __future__ import annotations

import os

import pandas as pd
from fredapi import Fred

from common import RAW_DIR, get_logger, load_env_file, load_settings, utc_now_iso


def main() -> None:
    load_env_file()
    logger = get_logger("04_rates_fred")
    key = os.environ.get("FRED_API_KEY")
    if not key:
        logger.error("FRED_API_KEY가 없어. .env 파일이나 환경변수로 설정해")
        return
    fred = Fred(api_key=key)
    cfg = load_settings()
    start = pd.Timestamp(cfg["sample"]["start_date"]) - pd.DateOffset(days=30)  # t-7 계산 여유분
    out_dir = RAW_DIR / "rates"
    out_dir.mkdir(parents=True, exist_ok=True)
    series = cfg["rates_fred"]["series"]
    logger.info("FRED 금리 수집 시작 | %s | %s ~ %s", list(series), start.date(), cfg["sample"]["end_date"])
    for i, (sid, desc) in enumerate(series.items(), 1):
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=cfg["sample"]["end_date"])
            df = s.rename("value").rename_axis("date").reset_index()
            df["collected_at_utc"] = utc_now_iso()
            df.to_csv(out_dir / f"fred_{sid}.csv", index=False)
            logger.info("[%d/%d] %s (%s): %d행, 결측 %d행 (휴장일)", i, len(series), sid, desc,
                        len(df), df["value"].isna().sum())
        except Exception as e:
            logger.error("[%d/%d] %s 실패: %s", i, len(series), sid, e)
    logger.info("FRED 금리 수집 종료")


if __name__ == "__main__":
    main()
