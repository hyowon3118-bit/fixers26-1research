"""RQ1·RQ3용 이벤트 패널: Kalshi 이벤트 × 측정 시점(h)마다 분포 통계량과 연구 변수 계산.

변수 정의 (연구설계 수정본 기준)
  A        실제 발표치 (ForexFactory 실제치. ALFRED·Kalshi 정산 결과와 일치 확인됨)
  F        ForexFactory 컨센서스
  σ        해당 발표일 이전 (A − F)의 표준편차 (expanding, 코로나 기간 제외 설정 가능)
  S        (A − F) / σ                     표준화 서프라이즈
  K, M     Kalshi 분포의 평균, 중앙값 (원본 get_moments 값)
  Gap      (K − F) / σ,  Gap_median = (M − F) / σ
  W_iqr    (q75 − q25) / σ                 분포 폭 (주 지표)
  W_sd     분포 표준편차 / σ                분포 폭 (보조 지표)
  E_K      |A − K| / σ                     Kalshi 자체 예측오차
  cover50  q25 ≤ A ≤ q75,  cover80  q10 ≤ A ≤ q90   예측구간 적중 여부

분위수: 구간 값 = 기준선 + moment_adjustment (원본 get_moments와 같은 값 매핑),
        누적확률이 처음으로 p 이상이 되는 값 (내 정의, 근거 문헌 없음)
측정 시점: Kalshi 거래 마감일(예정 발표일) h일 전 이하에서 가장 가까운 관측일

입력: data/interim/kalshi_release_match.parquet, kalshi_moments.parquet, kalshi_dist.parquet, releases.parquet
출력: data/processed/event_panel.parquet
실행: python src/features/01_event_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
from common import PROJECT_ROOT, get_logger, load_settings  # noqa: E402

INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def expanding_sigma(rel: pd.DataFrame, exclude: tuple[str, str], min_obs: int) -> pd.DataFrame:
    """각 발표일마다, 그 이전 발표들의 (A − F) 표준편차."""
    r = rel.dropna(subset=["A_ff", "F"]).copy()
    r["d"] = pd.to_datetime(r["release_date_et"])
    r["err"] = r["A_ff"] - r["F"]
    use = ~r["d"].between(pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1]))
    rows = []
    for s, g in r.groupby("series"):
        g = g.sort_values("d")
        for d in g["d"]:
            past = g[(g["d"] < d) & use.loc[g.index]]["err"]
            rows.append({"series": s, "release_date_actual": d.date(), "n_sigma": len(past),
                         "sigma": past.std(ddof=1) if len(past) >= min_obs else np.nan})
    return pd.DataFrame(rows)


def quantiles(dist: pd.DataFrame, adj: dict) -> pd.DataFrame:
    """이벤트·날짜별 분포 분위수와 표준편차 (구간 값 = 기준선 + moment_adjustment)."""
    rows = []
    for (ev, d, s), g in dist.groupby(["contract_preamble", "date", "series"], sort=False):
        g = g.sort_values("strike")
        x = g["strike"].to_numpy(float) + adj[s]
        w = g["probability"].to_numpy(float)
        w = w / w.sum()
        cw = np.cumsum(w)
        q = {f"q{int(p*100)}": x[int(np.argmax(cw >= p - 1e-12))] for p in (0.10, 0.25, 0.75, 0.90)}
        mu = (w * x).sum()
        rows.append({"contract_preamble": ev, "date": d, **q, "dist_sd": float(np.sqrt((w * (x - mu) ** 2).sum()))})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_settings()
    pcfg = cfg["event_panel"]
    adj = {s: v["moment_adjustment"] for s, v in cfg["clean_kalshi"]["series"].items()}
    logger = get_logger("features_01_event_panel")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] 입력 로드")
    ev = pd.read_parquet(INTERIM / "kalshi_release_match.parquet")
    ev = ev.dropna(subset=["F", "A_ff"])
    mom = pd.read_parquet(INTERIM / "kalshi_moments.parquet")
    dist = pd.read_parquet(INTERIM / "kalshi_dist.parquet")
    rel = pd.read_parquet(INTERIM / "releases.parquet")
    logger.info("  컨센서스·실제치가 있는 정산 이벤트 %d개", len(ev))

    logger.info("[2/4] σ 계산 (제외 기간 %s, 최소 %d건)", pcfg["sigma_exclude"], pcfg["sigma_min_obs"])
    sig = expanding_sigma(rel, tuple(pcfg["sigma_exclude"]), pcfg["sigma_min_obs"])
    ev = ev.merge(sig, on=["series", "release_date_actual"], how="left")
    logger.info("  σ 없는 이벤트 %d개", int(ev["sigma"].isna().sum()))

    logger.info("[3/4] 분포 분위수 계산")
    qs = quantiles(dist, adj)
    snap = mom.merge(qs, on=["contract_preamble", "date"], how="left")
    snap["date"] = pd.to_datetime(snap["date"])

    logger.info("[4/4] 측정 시점별 패널 구성 (h=%s)", pcfg["horizons"])
    out = []
    for h in pcfg["horizons"]:
        for _, e in ev.iterrows():
            target = pd.Timestamp(e["release_date_et"]) - pd.Timedelta(days=h)
            cand = snap[(snap["contract_preamble"] == e["event_ticker"]) & (snap["date"] <= target)]
            row = {"h": h, "event_ticker": e["event_ticker"], "series": e["series"],
                   "ref_period": e["ref_period"], "release_date_et": e["release_date_et"],
                   "release_date_actual": e["release_date_actual"], "delay_days": e["delay_days"],
                   "A": e["A_ff"], "F": e["F"], "sigma": e["sigma"], "n_sigma": e["n_sigma"]}
            if not cand.empty:
                c = cand.sort_values("date").iloc[-1]
                stale = (target - c["date"]).days
                if stale <= pcfg["max_stale_days"]:
                    row.update({"obs_date": c["date"].date(), "stale_days": stale, "K": c["mean"], "M": c["median"],
                                "mode": c["mode"], "skewness": c["skewness"], "volume": c["daily_volume"],
                                **{k: c[k] for k in ["q10", "q25", "q75", "q90", "dist_sd"]}})
            out.append(row)
    p = pd.DataFrame(out)
    p["S"] = (p["A"] - p["F"]) / p["sigma"]
    p["Gap"] = (p["K"] - p["F"]) / p["sigma"]
    p["Gap_median"] = (p["M"] - p["F"]) / p["sigma"]
    p["W_iqr"] = (p["q75"] - p["q25"]) / p["sigma"]
    p["W_sd"] = p["dist_sd"] / p["sigma"]
    p["E_K"] = (p["A"] - p["K"]).abs() / p["sigma"]
    p["cover50"] = np.where(p["K"].isna(), np.nan, (p["A"] >= p["q25"] - 1e-9) & (p["A"] <= p["q75"] + 1e-9))
    p["cover80"] = np.where(p["K"].isna(), np.nan, (p["A"] >= p["q10"] - 1e-9) & (p["A"] <= p["q90"] + 1e-9))

    for (s, h), g in p.groupby(["series", "h"]):
        ok = g.dropna(subset=["S", "Gap"])
        logger.info("  %s h=%d: 분석 가능 %d/%d | corr(Gap,S) %.2f | |S| 평균 %.2f | W_iqr 중앙값 %.2f | 적중률 50%%구간 %.0f%% 80%%구간 %.0f%%",
                    s, h, len(ok), len(g), ok[["Gap", "S"]].corr().iloc[0, 1] if len(ok) > 2 else np.nan,
                    ok["S"].abs().mean(), ok["W_iqr"].median(), 100 * ok["cover50"].mean(), 100 * ok["cover80"].mean())
    p.to_parquet(PROCESSED / "event_panel.parquet", index=False)
    logger.info("저장 event_panel.parquet (%d행)", len(p))


if __name__ == "__main__":
    main()
