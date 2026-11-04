"""Kalshi threshold 계약 일봉 → 이벤트·날짜별 확률분포와 분포 통계량.

방법은 Diercks, Katz & Wright (2026)의 공개 코드를 따름 (src/clean/kalshi_dkw.py).
    1) 일별 가격: 그날 마지막 체결가 (원본 convert_to_daily, method='last')
    2) 거래 없는 날: 직전 체결가를 이어 씀 (fill_dataless_days)
    3) 단조성: 가격이 49에 가장 가까운 기준선에서 바깥쪽으로 강제 (clean_data, middle-out)
    4) 구간 확률: 인접 기준선 가격 차이, 빠진 기준선은 0으로 채우고 빈 구간은 중앙 쪽으로 이동 (convert_to_probabilities)
    5) 통계량: 평균·중앙값·최빈값·분산·왜도 (get_moments)

원본과 입력 자료가 다른 점: 원본은 체결 단위(trade-level) 자료에서 UTC 날짜별 마지막 체결가를 쓰고,
여기서는 Kalshi 일봉(미국 동부시간 자정 마감)의 종가(price.close)를 써.
설정(settings.yaml clean_kalshi)으로 강건성 검정용 대안을 고를 수 있어:
    price_method: last(기본) | mid(호가 중간값, 원본 부록 A의 비교 방식)
    monotone_method: middle-out(기본) | right-to-left | left-to-right

입력: data/raw/kalshi/markets/*.parquet, data/raw/kalshi/candles/{series}/*.parquet
출력 (data/interim/, 파일명 끝에 설정 이름이 붙음. 기본은 접미사 없음):
  kalshi_events.parquet    이벤트별 마감 시각·정산 결과 구간
  kalshi_daily.parquet     계약×관측일 가격 (보정 전·후)
  kalshi_dist.parquet      이벤트×관측일×구간 확률 (원본 daily_distribution_data에 대응)
  kalshi_moments.parquet   이벤트×관측일 통계량 (원본 daily_moments_data에 대응)
  kalshi_quality.parquet   이벤트×관측일 단조성 위반 수 등 품질 지표

실행 (프로젝트 루트에서): python src/clean/01_kalshi_distributions.py
강건성:                  python src/clean/01_kalshi_distributions.py --price mid --monotone right-to-left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, RAW_DIR, get_logger, load_settings  # noqa: E402
import kalshi_dkw as dkw  # noqa: E402

INTERIM = PROJECT_ROOT / "data" / "interim"
ET = "America/New_York"


# ---------------------------------------------------------------- 메타데이터
def load_markets(series: str, logger) -> pd.DataFrame:
    m = pd.read_parquet(RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet")
    n0 = len(m)
    m = m[(m["strike_type"] == "greater") & m["floor_strike"].notna()].copy()
    logger.info("  계약 %d개 중 threshold(greater) 계약 %d개 사용, %d개 제외", n0, len(m), n0 - len(m))
    # API가 4.1을 4.099999처럼 저장한 경우가 있어 소수 4자리로 반올림 (기준선 간격은 최소 0.1)
    m["strike"] = m["floor_strike"].astype(float).round(4)
    m["close_time_et"] = pd.to_datetime(m["close_time"], utc=True, format="ISO8601").dt.tz_convert(ET)
    return m[["ticker", "event_ticker", "strike", "close_time_et", "result"]].assign(series=series)


def event_outcomes(m: pd.DataFrame) -> pd.DataFrame:
    """정산 결과로 실제 발표치가 속한 구간 (outcome_lower, outcome_upper]."""
    rows = []
    for ev, g in m.groupby("event_ticker"):
        res = g["result"].fillna("")
        yes_max = g.loc[res == "yes", "strike"].max()
        no_min = g.loc[res == "no", "strike"].min()
        rows.append({
            "series": g["series"].iloc[0], "event_ticker": ev,
            "close_time_et": g["close_time_et"].min(),
            "release_date_et": g["close_time_et"].min().date(),
            "n_thresholds": len(g),
            "settled": bool(res.isin(["yes", "no"]).all()),
            "outcome_lower": yes_max if pd.notna(yes_max) else -np.inf,
            "outcome_upper": no_min if pd.notna(no_min) else np.inf,
            "outcome_consistent": not (pd.notna(yes_max) and pd.notna(no_min) and yes_max >= no_min),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 일별 가격
def load_daily(series: str, m: pd.DataFrame, price_method: str, max_spread: float, logger) -> pd.DataFrame:
    folder = RAW_DIR / "kalshi" / "candles" / series
    frames, missing = [], 0
    for t in m["ticker"]:
        f = folder / f"{t}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
        else:
            missing += 1
    logger.info("  캔들 파일 %d개 로드, 없음 %d개", len(frames), missing)
    if not frames:
        return pd.DataFrame()
    c = pd.concat(frames, ignore_index=True)

    def col(*names):
        # 실시간 엔드포인트는 *_dollars / *_fp, 과거 엔드포인트는 접미사 없는 이름 (둘 다 달러 단위)
        out = pd.Series(np.nan, index=c.index)
        for n in names:
            if n in c:
                out = out.fillna(pd.to_numeric(c[n], errors="coerce"))
        return out

    end_et = pd.to_datetime(c["end_period_ts"], unit="s", utc=True).dt.tz_convert(ET)
    d = pd.DataFrame({
        "ticker": c["ticker"],
        "candle_end_et": end_et,
        "date": pd.to_datetime((end_et - pd.Timedelta(minutes=1)).dt.date),  # 일봉 마감 직전 날짜 = 거래일
        "trade": col("price.close_dollars", "price.close"),
        "bid": col("yes_bid.close_dollars", "yes_bid.close"),
        "ask": col("yes_ask.close_dollars", "yes_ask.close"),
        "daily_volume": col("volume_fp", "volume"),
    })
    spread = d["ask"] - d["bid"]
    d["mid"] = np.where(d["bid"].notna() & d["ask"].notna() & (spread >= 0) & (spread <= max_spread),
                        (d["bid"] + d["ask"]) / 2, np.nan)
    d = d.merge(m[["ticker", "event_ticker", "strike", "close_time_et"]], on="ticker", how="left")
    after = d["candle_end_et"] > d["close_time_et"]
    d = d[~after]
    logger.info("  거래 마감 이후 캔들 %d행 제외", int(after.sum()))

    price = d["trade"] if price_method == "last" else d["mid"]
    # 원본과 같이 센트 단위로 바꾸고, Kalshi 가격 범위(1~99센트)에 맞춰 반올림
    d["yes_price"] = (price * 100).round(2)
    d = d.rename(columns={"event_ticker": "contract_preamble"})
    logger.info("  계약×관측일 %d행 중 가격 있는 행 %d (%s 기준)", len(d), d["yes_price"].notna().sum(), price_method)
    return d


def quality_before_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """보정 전 단조성 위반(높은 기준선이 더 비싼 경우) 개수와 최대 크기."""
    def stats(g):
        diffs = np.diff(g.sort_values("strike")["yes_price"].to_numpy())
        v = diffs[diffs > 0]
        return pd.Series({"n_strikes": len(g), "n_violations": len(v),
                          "max_violation_cents": v.max() if len(v) else 0.0,
                          "volume_total": g["daily_volume"].sum()})
    return df.groupby(["contract_preamble", "date"]).apply(stats, include_groups=False).reset_index()


def main() -> None:
    cfg = load_settings()
    ccfg = cfg["clean_kalshi"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", default=ccfg["price_method"], choices=["last", "mid"])
    ap.add_argument("--monotone", default=ccfg["monotone_method"],
                    choices=["middle-out", "right-to-left", "left-to-right"])
    args = ap.parse_args()
    suffix = "" if (args.price, args.monotone) == (ccfg["price_method"], ccfg["monotone_method"]) \
        else f"_{args.price}_{args.monotone}"

    logger = get_logger("clean_01_kalshi")
    INTERIM.mkdir(parents=True, exist_ok=True)
    logger.info("Kalshi 정제 시작 | 가격 %s | 단조성 %s | 출력 접미사 '%s'", args.price, args.monotone, suffix)

    outs = {k: [] for k in ["events", "daily", "dist", "moments", "quality"]}
    for series in cfg["kalshi"]["series"]:
        scfg = ccfg["series"].get(series)
        if scfg is None or not (RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet").exists():
            logger.warning("[%s] 설정 또는 메타데이터 없음, 건너뜀", series)
            continue
        logger.info("[%s] [1/6] 계약 메타데이터와 정산 결과", series)
        m = load_markets(series, logger)
        ev = event_outcomes(m)
        logger.info("  이벤트 %d개 (정산 완료 %d)", len(ev), ev["settled"].sum())

        logger.info("[%s] [2/6] 일별 가격 구성", series)
        d = load_daily(series, m, args.price, ccfg["max_spread"], logger)
        if d.empty:
            logger.warning("  캔들 없음, 건너뜀")
            continue
        base = d.dropna(subset=["yes_price"])[["date", "contract_preamble", "strike", "yes_price", "daily_volume"]]
        if base.empty:
            logger.warning("  가격 자료 없음, 건너뜀")
            continue

        logger.info("[%s] [3/6] 거래 없는 날 채우기 (직전 가격 이어 쓰기, 기간 %d일)", series, scfg["days_before_horizon"])
        filled = dkw.fill_dataless_days(base, scfg["days_before_horizon"])
        logger.info("  %d행 → %d행", len(base), len(filled))

        logger.info("[%s] [4/6] 단조성 점검과 보정 (%s)", series, args.monotone)
        q = quality_before_cleaning(filled)
        logger.info("  관측일 중 보정 전 위반이 있는 비율 %.1f%%, 최대 위반 %.1f센트",
                    100 * (q["n_violations"] > 0).mean(), q["max_violation_cents"].max())
        cleaned = dkw.clean_data(filled, args.monotone)

        logger.info("[%s] [5/6] 구간 확률 계산 (기준선 간격 %s)", series, scfg["strike_int"])
        dist = dkw.convert_to_probabilities(cleaned, scfg["strike_int"], scfg["days_before_horizon"])
        sums = dist.groupby(["contract_preamble", "date"])["probability"].sum()
        logger.info("  이벤트×관측일 %d개, 확률 합 %.4f~%.4f (100이어야 함)", len(sums), sums.min(), sums.max())

        logger.info("[%s] [6/6] 분포 통계량", series)
        mom = dkw.get_moments(dist, scfg["moment_adjustment"])
        logger.info("  통계량 %d행", len(mom))

        for k, v in zip(["events", "daily", "dist", "moments", "quality"], [ev, d, dist, mom, q]):
            outs[k].append(v.assign(series=series))

    if not outs["dist"]:
        logger.error("처리된 시리즈가 없어")
        return
    for k, frames in outs.items():
        df = pd.concat(frames, ignore_index=True)
        path = INTERIM / f"kalshi_{k}{suffix}.parquet"
        df.to_parquet(path, index=False)
        logger.info("저장 %s (%d행)", path.name, len(df))
    logger.info("Kalshi 정제 종료")


if __name__ == "__main__":
    main()
