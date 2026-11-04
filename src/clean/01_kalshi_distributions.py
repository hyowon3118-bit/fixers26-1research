"""Kalshi threshold 계약 가격 → 이벤트·날짜별 확률분포로 정제.

Kalshi 매크로 계약은 서로 겹치지 않는 구간(bracket)이 아니라
"발표치가 x_j를 초과하면 Yes"인 기준선(threshold) 계약이야.
그래서 각 가격은 생존함수 S(x_j) = P(A > x_j)의 추정치로 읽고,
구간 확률은 인접 기준선의 차이로 만든다.
    P(A <= x_1)          = 1 - S(x_1)
    P(x_j < A <= x_{j+1}) = S(x_j) - S(x_{j+1})
    P(A > x_J)           = S(x_J)

처리 순서
  [1] 계약 메타데이터: threshold 계약만 남기고, 정산 결과로 실제 발표 구간 계산
  [2] 일별 캔들: 미국 동부시간 기준 관측일과 가격 후보(호가 중간값, 체결가, 직전 체결가) 구성
  [3] 가격 선택: settings.yaml의 price_rules 순서대로 사용 가능한 값 선택
  [4] 단조성 점검과 보정: 높은 기준선이 더 비싼 위반을 세고, 감소 isotonic 회귀로 보정
  [5] 구간 확률 계산과 저장

입력: data/raw/kalshi/markets/*.parquet, data/raw/kalshi/candles/{series}/*.parquet
출력:
  data/interim/kalshi_events.parquet     이벤트별 메타·정산 결과
  data/interim/kalshi_snapshots.parquet  계약×관측일 가격 (원가격, 보정가격)
  data/interim/kalshi_dist.parquet       이벤트×관측일×구간 확률
  data/interim/kalshi_quality.parquet    이벤트×관측일 품질 지표 (유효 기준선 수, 단조성 위반)

실행 (프로젝트 루트에서): python src/clean/01_kalshi_distributions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
from common import PROJECT_ROOT, RAW_DIR, get_logger, load_settings  # noqa: E402

INTERIM = PROJECT_ROOT / "data" / "interim"
ET = "America/New_York"


# ---------------------------------------------------------------- [1] 메타데이터
def load_events(series: str, logger) -> pd.DataFrame:
    path = RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet"
    m = pd.read_parquet(path)
    n0 = len(m)
    m = m[(m["strike_type"] == "greater") & m["floor_strike"].notna()].copy()
    logger.info("  계약 %d개 중 threshold(greater) 계약 %d개 사용, %d개 제외", n0, len(m), n0 - len(m))
    m["threshold"] = m["floor_strike"].astype(float)
    m["close_time_et"] = pd.to_datetime(m["close_time"], utc=True, format="ISO8601").dt.tz_convert(ET)
    return m[["ticker", "event_ticker", "threshold", "close_time_et", "result", "status"]].assign(series=series)


def event_outcomes(m: pd.DataFrame) -> pd.DataFrame:
    """정산 결과로 실제 발표치가 속한 구간 (lower, upper]를 계산."""
    rows = []
    for ev, g in m.groupby("event_ticker"):
        res = g["result"].fillna("")
        settled = res.isin(["yes", "no"]).all() and len(g) > 0
        yes_max = g.loc[res == "yes", "threshold"].max()
        no_min = g.loc[res == "no", "threshold"].min()
        consistent = not (pd.notna(yes_max) and pd.notna(no_min) and yes_max >= no_min)
        rows.append({
            "series": g["series"].iloc[0],
            "event_ticker": ev,
            "close_time_et": g["close_time_et"].min(),
            "n_thresholds": len(g),
            "settled": settled,
            "outcome_lower": yes_max if pd.notna(yes_max) else -np.inf,   # A > lower
            "outcome_upper": no_min if pd.notna(no_min) else np.inf,      # A <= upper
            "outcome_consistent": consistent,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- [2] 캔들
def load_candles(series: str, tickers: list[str], logger) -> pd.DataFrame:
    folder = RAW_DIR / "kalshi" / "candles" / series
    frames, missing = [], 0
    for t in tickers:
        f = folder / f"{t}.parquet"
        if not f.exists():
            missing += 1
            continue
        frames.append(pd.read_parquet(f))
    logger.info("  캔들 파일 %d개 로드, 없음 %d개", len(frames), missing)
    if not frames:
        return pd.DataFrame()
    c = pd.concat(frames, ignore_index=True)

    def col(*names):
        """여러 후보 열 이름 중 값이 있는 것을 행마다 순서대로 채움."""
        out = pd.Series(np.nan, index=c.index)
        for name in names:
            if name in c:
                out = out.fillna(pd.to_numeric(c[name], errors="coerce"))
        return out

    end_et = pd.to_datetime(c["end_period_ts"], unit="s", utc=True).dt.tz_convert(ET)
    out = pd.DataFrame({
        "ticker": c["ticker"],
        "event_ticker": c["event_ticker"],
        "candle_end_et": end_et,
        # 일봉은 동부시간 자정에 마감 → 마감 1분 전 날짜를 관측일(거래일)로 사용
        "obs_date": (end_et - pd.Timedelta(minutes=1)).dt.date,
        # 실시간 엔드포인트는 *_dollars / *_fp, 과거(historical) 엔드포인트는 접미사 없는 이름을 써.
        # 두 형식 모두 달러(0~1) 단위라서 있는 쪽 값을 사용
        "bid": col("yes_bid.close_dollars", "yes_bid.close"),
        "ask": col("yes_ask.close_dollars", "yes_ask.close"),
        "trade": col("price.close_dollars", "price.close"),
        "prev": col("price.previous_dollars", "price.previous"),
        "volume": col("volume_fp", "volume"),
        "open_interest": col("open_interest_fp", "open_interest"),
    })
    return out


# ---------------------------------------------------------------- [3] 가격 선택
def choose_price(c: pd.DataFrame, rules: list[str], max_spread: float) -> pd.DataFrame:
    spread = c["ask"] - c["bid"]
    c["spread"] = spread
    valid_quote = c["bid"].notna() & c["ask"].notna() & (spread >= 0) & (spread <= max_spread)
    c["mid"] = np.where(valid_quote, (c["bid"] + c["ask"]) / 2, np.nan)
    c["p_raw"], c["price_source"] = np.nan, None
    for rule in rules:
        fill = c["p_raw"].isna() & c[rule].notna()
        c.loc[fill, "p_raw"] = c.loc[fill, rule]
        c.loc[fill, "price_source"] = rule
    return c


# ---------------------------------------------------------------- [4]·[5] 분포
def build_distribution(snap: pd.DataFrame, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
    snap = snap.sort_values(["event_ticker", "obs_date", "threshold"]).copy()
    snap["p_mono"] = np.nan
    dist_rows, qual_rows = [], []
    groups = snap.groupby(["event_ticker", "obs_date"], sort=False)
    n_groups = groups.ngroups
    for i, ((ev, d), g) in enumerate(groups, 1):
        valid = g[g["p_raw"].notna()]
        diffs = np.diff(valid["p_raw"].to_numpy())
        viol = diffs[diffs > 1e-9]
        qual = {
            "event_ticker": ev, "obs_date": d,
            "n_thresholds": len(g), "n_valid": len(valid),
            "n_violations": int(len(viol)),
            "max_violation": float(viol.max()) if len(viol) else 0.0,
            "volume_total": g["volume"].sum(min_count=1),
            "share_mid": (valid["price_source"] == "mid").mean() if len(valid) else np.nan,
        }
        qual_rows.append(qual)
        if len(valid) < 2:
            continue
        fitted = iso.fit_transform(valid["threshold"].to_numpy(), valid["p_raw"].to_numpy())
        snap.loc[valid.index, "p_mono"] = fitted
        x = valid["threshold"].to_numpy()
        s = fitted
        lowers = np.concatenate([[-np.inf], x])
        uppers = np.concatenate([x, [np.inf]])
        probs = np.concatenate([[1 - s[0]], s[:-1] - s[1:], [s[-1]]])
        for lo, up, p in zip(lowers, uppers, probs):
            dist_rows.append({"event_ticker": ev, "obs_date": d, "lower": lo, "upper": up, "prob": p})
        if i % 2000 == 0 or i == n_groups:
            logger.info("  분포 계산 %d/%d (이벤트×관측일)", i, n_groups)
    return snap, pd.DataFrame(dist_rows), pd.DataFrame(qual_rows)


def main() -> None:
    cfg = load_settings()
    ccfg = cfg["clean_kalshi"]
    logger = get_logger("clean_01_kalshi")
    INTERIM.mkdir(parents=True, exist_ok=True)
    logger.info("Kalshi 정제 시작 | 가격 규칙 %s | 최대 호가차 %.2f", ccfg["price_rules"], ccfg["max_spread"])

    all_events, all_snap, all_dist, all_qual = [], [], [], []
    for series in cfg["kalshi"]["series"]:
        if not (RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet").exists():
            logger.warning("[%s] 메타데이터 파일 없음, 건너뜀", series)
            continue
        logger.info("[%s] [1/5] 계약 메타데이터와 정산 결과", series)
        m = load_events(series, logger)
        ev = event_outcomes(m)
        logger.info("  이벤트 %d개 (정산 완료 %d, 결과 불일치 %d)",
                    len(ev), ev["settled"].sum(), (~ev["outcome_consistent"]).sum())

        logger.info("[%s] [2/5] 일별 캔들 로드", series)
        c = load_candles(series, m["ticker"].tolist(), logger)
        if c.empty:
            logger.warning("  캔들 없음, 건너뜀")
            continue

        logger.info("[%s] [3/5] 가격 선택", series)
        c = choose_price(c, ccfg["price_rules"], ccfg["max_spread"])
        snap = c.merge(m[["ticker", "threshold", "series", "close_time_et"]], on="ticker", how="left")
        # 거래 마감(발표 직전) 이후에 끝나는 캔들은 발표 결과가 섞이므로 제외
        after = snap["candle_end_et"] > snap["close_time_et"]
        snap = snap[~after].drop(columns="close_time_et")
        logger.info("  거래 마감 이후 캔들 %d행 제외", int(after.sum()))
        logger.info("  계약×관측일 %d행 | 가격 출처 %s | 가격 없음 %d행",
                    len(snap), snap["price_source"].value_counts().to_dict(), snap["p_raw"].isna().sum())

        logger.info("[%s] [4/5] 단조성 점검·보정, [5/5] 구간 확률", series)
        snap, dist, qual = build_distribution(snap, logger)
        if not qual.empty:
            logger.info("  관측일 중 단조성 위반이 있는 비율 %.1f%%, 최대 위반폭 %.3f",
                        100 * (qual["n_violations"] > 0).mean(), qual["max_violation"].max())
        if not dist.empty:
            chk = dist.groupby(["event_ticker", "obs_date"])["prob"].sum()
            logger.info("  구간 확률 합 점검: 최소 %.4f, 최대 %.4f (1이어야 함)", chk.min(), chk.max())
        dist["series"], qual["series"] = series, series
        all_events.append(ev); all_snap.append(snap); all_dist.append(dist); all_qual.append(qual)

    if not all_events:
        logger.error("처리할 시리즈가 없어")
        return
    outputs = {
        "kalshi_events.parquet": pd.concat(all_events, ignore_index=True),
        "kalshi_snapshots.parquet": pd.concat(all_snap, ignore_index=True),
        "kalshi_dist.parquet": pd.concat(all_dist, ignore_index=True),
        "kalshi_quality.parquet": pd.concat(all_qual, ignore_index=True),
    }
    for name, df in outputs.items():
        df.to_parquet(INTERIM / name, index=False)
        logger.info("저장 %s (%d행)", name, len(df))
    logger.info("Kalshi 정제 종료")


if __name__ == "__main__":
    main()
