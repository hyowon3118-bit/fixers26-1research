"""RQ1·RQ3용 이벤트 패널: Kalshi 이벤트 × 측정 시점(h)마다 분포 통계량과 연구 변수 계산.

변수 정의 (연구설계 수정본 기준)
  A        실제 발표치 (가능한 시리즈는 ALFRED 최초 발표치, 없으면 ForexFactory actual)
  F        ForexFactory 컨센서스. settings.yaml event_panel.consensus_vintage로 선택
           - latest(기본): 가장 최근 수집본. 과거 값을 나중에 수집했으므로 사후 비교라는 한계를 명시
           - strict: 측정 horizon 날짜까지 실제 수집된 스냅샷만 (앞으로 발표 전에 수집해둔 자료가 쌓이면 사용)
           F_vintage_ok 열이 그 값이 측정 시점 이전 수집본인지 표시
  σ        같은 horizon에서 과거에 이용 가능했던 (A − F)의 expanding 표준편차
  S        (A − F) / σ                     표준화 서프라이즈
  K, M     Kalshi 분포의 평균, 중앙값 (원본 get_moments 값)
  Gap      (K − F) / σ,  Gap_median = (M − F) / σ
  W_iqr    (q75 − q25) / σ                 분포 폭 (주 지표, 구간 안 균등분포로 보간한 연속 분위수)
  W_iqr_disc                               이산 분위수로 계산한 IQR (강건성)
  W_sd     분포 표준편차 / σ                분포 폭 (보조 지표)
  E_K      |A − K| / σ                     Kalshi 자체 예측오차
  cover50  q25 ≤ A ≤ q75,  cover80  q10 ≤ A ≤ q90   예측구간 적중 여부

분위수: 내부 구간은 반올림 전 범위 (기준선 + 단위/2, 다음 기준선 + 단위/2] 안에서 균등하다고 가정.
        최저·최고 열린 꼬리 안에 분위수가 놓이면 유한한 경계가 식별되지 않으므로 해당 분위수는 결측.
        이산 분위수는 구간 값 = 기준선 + moment_adjustment에서 누적확률이 처음 p 이상인 값.
측정 시점: 실제 발표일 h일 전 이하에서 가장 가까운 Kalshi 관측일.

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


ET = "America/New_York"


def forecast_asof(events: pd.DataFrame, snapshots: pd.DataFrame, h: int,
                  event_date_col: str, vintage: str = "strict") -> pd.DataFrame:
    """컨센서스 선택.

    strict: h일 전까지 실제 수집된 마지막 스냅샷만 사용. 사후 수집뿐이면 결측.
    latest: 수집 시점과 관계없이 가장 최근 스냅샷 사용 (과거 컨센서스를 나중에 한꺼번에 수집한 경우.
            ForexFactory의 과거 forecast가 발표 당시 값인지 보장되지 않는다는 한계를 명시해야 함)
    F_vintage_ok는 그 값이 h일 전까지 실제로 수집된 것인지 표시 (latest에서도 기록).
    """
    ev = events[["series", event_date_col]].drop_duplicates().copy()
    ev[event_date_col] = pd.to_datetime(ev[event_date_col]).dt.normalize()
    snap = snapshots.dropna(subset=["F", "scraped_at_utc"]).copy()
    snap["scraped_at_utc"] = pd.to_datetime(snap["scraped_at_utc"], utc=True, errors="coerce")
    snap = snap.dropna(subset=["scraped_at_utc"])
    snap["release_date_et"] = pd.to_datetime(snap["release_date_et"]).dt.date
    snap["scrape_date_et"] = snap["scraped_at_utc"].dt.tz_convert(ET).dt.date
    out = []
    for _, e in ev.iterrows():
        release_day = e[event_date_col]
        target_day = (release_day - pd.Timedelta(days=h)).date()
        same = snap[(snap["series"] == e["series"]) & (snap["release_date_et"] == release_day.date())]
        timely = same[same["scrape_date_et"] <= target_day]
        cand = timely if vintage == "strict" else same
        f = cand.sort_values("scraped_at_utc").iloc[-1]["F"] if not cand.empty else np.nan
        out.append({"series": e["series"], event_date_col: release_day.date(), "F_asof": f,
                    "F_vintage_ok": not timely.empty})
    return pd.DataFrame(out)


def expanding_sigma(rel: pd.DataFrame, forecasts: pd.DataFrame,
                    exclude: tuple[str, str], min_obs: int) -> pd.DataFrame:
    """같은 horizon에서 이용 가능했던 과거 (A − F)만으로 expanding SD 계산."""
    r = rel.dropna(subset=["A"]).merge(forecasts, on=["series", "release_date_et"], how="left")
    r["d"] = pd.to_datetime(r["release_date_et"])
    r["err"] = r["A"] - r["F_asof"]
    use = ~r["d"].between(pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1]))
    rows = []
    for s, g in r.groupby("series"):
        g = g.sort_values("d")
        for d in g["d"]:
            mask = (g["d"] < d) & use.loc[g.index] & g["err"].notna()
            past = g.loc[mask, "err"]
            rows.append({"series": s, "release_date_et": d.date(), "n_sigma": len(past),
                         "sigma": past.std(ddof=1) if len(past) >= min_obs else np.nan})
    return pd.DataFrame(rows)


PROBS = (0.10, 0.25, 0.75, 0.90)


def bin_edges(strikes: np.ndarray, unit: float) -> tuple[np.ndarray, np.ndarray]:
    """내부 확률 구간의 경계. 최저·최고 확률 행은 열린 꼬리라 유한 경계를 두지 않는다."""
    lo = np.full(len(strikes), np.nan)
    hi = np.full(len(strikes), np.nan)
    if len(strikes) > 1:
        hi[0] = strikes[1] + unit / 2
        lo[-1] = strikes[-1] + unit / 2
    if len(strikes) > 2:
        lo[1:-1] = strikes[1:-1] + unit / 2
        hi[1:-1] = strikes[2:] + unit / 2
    return lo, hi


def continuous_quantile(lo, hi, w, p) -> float:
    """내부 구간 균등 가정으로 보간; 분위수가 열린 꼬리에 놓이면 식별 불가로 NaN."""
    cw = np.cumsum(w)
    crossed = np.flatnonzero(cw >= p - 1e-12)
    if not len(crossed):
        return np.nan
    j = int(crossed[0])
    prev = cw[j - 1] if j > 0 else 0.0
    if j == 0:
        return float(hi[0]) if np.isclose(p, cw[0], atol=1e-12) else np.nan
    if j == len(w) - 1:
        return float(lo[-1]) if np.isclose(p, prev, atol=1e-12) else np.nan
    if not np.isfinite(lo[j]) or not np.isfinite(hi[j]) or w[j] <= 0:
        return np.nan
    frac = (p - prev) / w[j]
    return float(lo[j] + frac * (hi[j] - lo[j]))


def quantiles(dist: pd.DataFrame, adj: dict) -> pd.DataFrame:
    """이벤트·날짜별 분위수와 표준편차.

    주 지표: 연속 분위수 qXX (구간 안 균등분포 보간)
    보조 지표: 이산 분위수 qXX_disc (구간 값 = 기준선 + moment_adjustment, 누적확률이 처음 p 이상인 값)
    moment_adjustment는 각 시리즈의 발표 단위(0.1%p, 1000명)와 같아서 반올림 단위로도 씀.
    """
    rows = []
    for (ev, d, s), g in dist.groupby(["contract_preamble", "date", "series"], sort=False):
        g = g.sort_values("strike")
        strikes = g["strike"].to_numpy(float)
        unit = adj[s]
        w = g["probability"].to_numpy(float)
        w = w / w.sum()
        x = strikes + unit
        cw = np.cumsum(w)
        lo, hi = bin_edges(strikes, unit)
        mu = (w * x).sum()
        row = {"contract_preamble": ev, "date": d,
               "dist_sd": float(np.sqrt((w * (x - mu) ** 2).sum()))}
        for p in PROBS:
            row[f"q{int(p*100)}"] = continuous_quantile(lo, hi, w, p)
            row[f"q{int(p*100)}_disc"] = x[int(np.argmax(cw >= p - 1e-12))]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_settings()
    pcfg = cfg["event_panel"]
    adj = {s: v["moment_adjustment"] for s, v in cfg["clean_kalshi"]["series"].items()}
    logger = get_logger("features_01_event_panel")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] 입력 로드")
    ev = pd.read_parquet(INTERIM / "kalshi_release_match.parquet")
    ev = ev.dropna(subset=["A"])
    mom = pd.read_parquet(INTERIM / "kalshi_moments.parquet")
    dist = pd.read_parquet(INTERIM / "kalshi_dist.parquet")
    rel = pd.read_parquet(INTERIM / "releases.parquet")
    forecasts = pd.read_parquet(INTERIM / "release_forecasts.parquet")
    logger.info("  컨센서스·실제치가 있는 정산 이벤트 %d개", len(ev))

    logger.info("[2/4] 분포 분위수 계산")
    qs = quantiles(dist, adj)
    snap = mom.merge(qs, on=["contract_preamble", "date"], how="left")
    snap["date"] = pd.to_datetime(snap["date"])

    vintage = pcfg.get("consensus_vintage", "latest")
    logger.info("[3/4] 측정 시점별 패널 구성 (h=%s, 컨센서스 기준 %s)", pcfg["horizons"], vintage)
    if vintage == "latest":
        logger.warning("  컨센서스는 가장 최근 수집본을 씀. 과거 forecast가 발표 당시 값인지 보장되지 않는 사후 비교라는 한계를 명시할 것")
    out = []
    for h in pcfg["horizons"]:
        f_ev = forecast_asof(ev, forecasts, h, "release_date_actual", vintage)
        f_rel = forecast_asof(rel, forecasts, h, "release_date_et", vintage)
        sig = expanding_sigma(rel, f_rel, tuple(pcfg["sigma_exclude"]), pcfg["sigma_min_obs"])
        logger.info("  h=%d: 컨센서스 결측 %d / 측정 시점 이전 수집본 %d개 / σ 미산출 %d",
                    h, int(f_ev["F_asof"].isna().sum()), int(f_ev["F_vintage_ok"].sum()),
                    int(sig["sigma"].isna().sum()))
        for _, e in ev.iterrows():
            target = pd.Timestamp(e["release_date_actual"]) - pd.Timedelta(days=h)
            cand = snap[(snap["contract_preamble"] == e["event_ticker"]) & (snap["date"] <= target)]
            frow = f_ev[(f_ev["series"] == e["series"]) &
                        (f_ev["release_date_actual"] == pd.Timestamp(e["release_date_actual"]).date())]
            f_h = frow["F_asof"].iloc[0] if not frow.empty else np.nan
            f_ok = bool(frow["F_vintage_ok"].iloc[0]) if not frow.empty else False
            srow = sig[(sig["series"] == e["series"]) &
                       (sig["release_date_et"] == pd.Timestamp(e["release_date_actual"]).date())]
            sigma = srow["sigma"].iloc[0] if not srow.empty else np.nan
            n_sigma = srow["n_sigma"].iloc[0] if not srow.empty else 0
            row = {"h": h, "event_ticker": e["event_ticker"], "series": e["series"],
                   "ref_period": e["ref_period"], "release_date_et": e["release_date_et"],
                   "release_date_actual": e["release_date_actual"], "delay_days": e["delay_days"],
                   "target_date": target.date(), "A": e["A"], "F": f_h, "F_vintage_ok": f_ok,
                   "sigma": sigma, "n_sigma": n_sigma}
            if not cand.empty:
                c = cand.sort_values("date").iloc[-1]
                stale = (target - c["date"]).days
                if stale <= pcfg["max_stale_days"]:
                    row.update({"obs_date": c["date"].date(), "stale_days": stale, "K": c["mean"], "M": c["median"],
                                "mode": c["mode"], "skewness": c["skewness"], "volume": c["daily_volume"],
                                **{k: c[k] for k in ["q10", "q25", "q75", "q90", "q10_disc", "q25_disc", "q75_disc", "q90_disc", "dist_sd"]}})
            out.append(row)
    p = pd.DataFrame(out)
    p["S"] = (p["A"] - p["F"]) / p["sigma"]
    p["Gap"] = (p["K"] - p["F"]) / p["sigma"]
    p["Gap_median"] = (p["M"] - p["F"]) / p["sigma"]
    p["W_iqr"] = (p["q75"] - p["q25"]) / p["sigma"]            # 주 지표: 연속 IQR
    p["W_iqr_disc"] = (p["q75_disc"] - p["q25_disc"]) / p["sigma"]  # 강건성: 이산 IQR
    p["W_sd"] = p["dist_sd"] / p["sigma"]
    p["E_K"] = (p["A"] - p["K"]).abs() / p["sigma"]
    p["cover50"] = pd.Series(pd.NA, index=p.index, dtype="boolean")
    valid50 = p["q25"].notna() & p["q75"].notna()
    p.loc[valid50, "cover50"] = ((p.loc[valid50, "A"] >= p.loc[valid50, "q25"] - 1e-9) &
                                  (p.loc[valid50, "A"] <= p.loc[valid50, "q75"] + 1e-9))
    p["cover80"] = pd.Series(pd.NA, index=p.index, dtype="boolean")
    valid80 = p["q10"].notna() & p["q90"].notna()
    p.loc[valid80, "cover80"] = ((p.loc[valid80, "A"] >= p.loc[valid80, "q10"] - 1e-9) &
                                  (p.loc[valid80, "A"] <= p.loc[valid80, "q90"] + 1e-9))

    for (s, h), g in p.groupby(["series", "h"]):
        ok = g.dropna(subset=["S", "Gap"])
        cover50 = g["cover50"].dropna().mean()
        cover80 = g["cover80"].dropna().mean()
        logger.info("  %s h=%d: 분석 가능 %d/%d | corr(Gap,S) %.2f | |S| 평균 %.2f | W_iqr 중앙값 %.2f | 적중률 50%%구간 %.0f%% 80%%구간 %.0f%%",
                    s, h, len(ok), len(g), ok[["Gap", "S"]].corr().iloc[0, 1] if len(ok) > 2 else np.nan,
                    ok["S"].abs().mean(), ok["W_iqr"].median(),
                    100 * cover50 if pd.notna(cover50) else np.nan,
                    100 * cover80 if pd.notna(cover80) else np.nan)
    logger.info("[4/4] 결과 저장")
    p.to_parquet(PROCESSED / "event_panel.parquet", index=False)
    logger.info("저장 event_panel.parquet (%d행)", len(p))


if __name__ == "__main__":
    main()
