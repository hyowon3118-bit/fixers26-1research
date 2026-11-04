"""이 연구의 Kalshi 통계량(일봉 기반)을 연준 논문 공개 통계량(체결 단위 기반)과 비교해 오차를 잼.

차이가 생길 수 있는 원인
  (a) 날짜 경계: 원본은 체결 시각의 UTC 날짜, 이 연구는 미국 동부시간 날짜.
      UTC 자정 = 동부시간 19시(서머타임) 또는 20시라서, 그 이후 저녁 체결이 원본에서는 다음 날로 넘어감
  (b) 음수 기준선: 원본 정규식이 T-0.1 같은 계약을 읽지 못해 원본 분포에서 빠짐
  (c) 원본 공개 파일의 수집 시점 차이 (원본은 매주 다시 계산)
비교 결과는 (b)에 해당하는 이벤트를 따로 표시해서 (a)의 크기를 가늠할 수 있게 함.

입력: data/raw/dkw_reference/ (scraping/06_dkw_reference.py), data/interim/kalshi_moments.parquet,
      data/raw/kalshi/markets/{KXCPI,KXU3}_markets.parquet
출력: outputs/tables/compare_with_dkw.csv (이벤트×날짜별 차이), 요약은 로그
실행: python tests/compare_with_dkw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraping"))
from common import RAW_DIR, get_logger  # noqa: E402

PAIRS = {"KXCPI": "daily_moments_headline_cpi_releases_mom.csv",
         "KXU3": "daily_moments_unemployment_releases.csv"}
COLS = ["mean", "median", "mode", "variance"]


def main() -> None:
    logger = get_logger("compare_with_dkw")
    ours = pd.read_parquet(ROOT / "data" / "interim" / "kalshi_moments.parquet")
    ours["date"] = pd.to_datetime(ours["date"])
    rows = []
    for series, fname in PAIRS.items():
        path = RAW_DIR / "dkw_reference" / fname
        if not path.exists():
            logger.warning("[%s] 비교 파일 없음 (%s). 06_dkw_reference.py를 먼저 실행해", series, fname)
            continue
        ref = pd.read_csv(path, parse_dates=["date"])
        mk = pd.read_parquet(RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet")
        neg = set(mk.loc[mk["floor_strike"] < 0, "event_ticker"])
        o = ours[ours["series"] == series]
        m = o.merge(ref, on=["contract_preamble", "date"], suffixes=("_ours", "_dkw"), how="outer", indicator=True)
        m["series"] = series
        m["has_negative_strike"] = m["contract_preamble"].isin(neg)
        for c in COLS:
            m[f"diff_{c}"] = m[f"{c}_ours"] - m[f"{c}_dkw"]
        both = m[m["_merge"] == "both"]
        logger.info("[%s] 이벤트×날짜: 공통 %d, 이 연구에만 %d, 원본에만 %d",
                    series, len(both), int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum()))
        for label, g in [("음수 기준선 없는 이벤트", both[~both["has_negative_strike"]]),
                         ("음수 기준선 있는 이벤트", both[both["has_negative_strike"]])]:
            if g.empty:
                continue
            same = (g["diff_mean"].abs() < 1e-6).mean()
            logger.info("  %s %d건 | 평균 완전일치 %.0f%% | |평균 차이| 중앙값 %.4f, 95%%분위 %.4f, 최대 %.4f | "
                        "중앙값 다른 비율 %.0f%% | 최빈값 다른 비율 %.0f%%",
                        label, len(g), 100 * same, g["diff_mean"].abs().median(),
                        g["diff_mean"].abs().quantile(0.95), g["diff_mean"].abs().max(),
                        100 * (g["diff_median"].abs() > 1e-6).mean(), 100 * (g["diff_mode"].abs() > 1e-6).mean())
        rows.append(m.drop(columns="_merge").assign(match=m["_merge"].astype(str)))
    if rows:
        out = ROOT / "outputs" / "tables" / "compare_with_dkw.csv"
        pd.concat(rows, ignore_index=True).to_csv(out, index=False)
        logger.info("저장 %s", out)


if __name__ == "__main__":
    main()
