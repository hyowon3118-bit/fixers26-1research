"""MOVE 지수(미국채 옵션 내재변동성) 일별 종가 수집. Yahoo Finance의 ^MOVE를 yfinance로 받음.

네가 이전 분석(Desktop/2.2test/mfed/get_move.py)에서 쓴 방식과 같고, 기간만 연구 표본에 맞춤.
필요: pip install yfinance
저장: data/raw/move/move_daily.csv (date, value)
실행: python scraping/07_move.py
"""
from __future__ import annotations

import pandas as pd

from common import RAW_DIR, get_logger, load_settings, utc_now_iso


def main() -> None:
    import yfinance as yf

    logger = get_logger("07_move")
    cfg = load_settings()
    start = (pd.Timestamp(cfg["sample"]["start_date"]) - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
    logger.info("[1/2] ^MOVE 내려받기 (%s ~)", start)
    df = yf.download("^MOVE", start=start, progress=False, auto_adjust=False)
    if df.empty:
        logger.error("  데이터 없음. 네트워크나 티커를 확인해")
        return
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "value"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["collected_at_utc"] = utc_now_iso()
    logger.info("[2/2] 저장")
    path = RAW_DIR / "move" / "move_daily.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("  %d행, %s ~ %s → %s", len(out), out["date"].min(), out["date"].max(), path)


if __name__ == "__main__":
    main()
