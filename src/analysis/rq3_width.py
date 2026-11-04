"""RQ3: Kalshi 분포 폭은 실제 발표 불확실성을 얼마나 잘 나타내는가.

가설 (연구설계 수정본)
  H3a  |S| = FE_k + d1·W_iqr + d2·|Gap| + e 에서 d1 > 0       [R5]  (|Gap| 통제는 ※)
  H3b  E_K = FE_k + d·W_iqr + e 에서 d > 0                     ※  (분포 폭이 Kalshi 자체 오차를 예측하는가)
  H3c  예측구간 적중률이 명목값과 같다 (50% 구간 → 50%, 80% 구간 → 80%)   ※
  H3d  ΔMOVE = a + d1·W_iqr + d2·|S| + MOVE(t−1) + ΔMOVE(t−1) + e 에서 d1 < 0   [R2] 확장, 조건부 가설
       (MOVE는 발표일 단위 변수라 같은 날 발표된 지표의 W·|S|를 평균해 발표일 1행으로 만듦)

W_iqr: 연속 IQR / σ (구간 안 균등분포 보간, 주 지표). 강건성: W_iqr_disc(이산 IQR), W_sd(분포 표준편차)
추론: 발표일 군집 표준오차, wild cluster bootstrap p값 [R5], 적중률은 발표일 단위 bootstrap 신뢰구간
      주 분석 1차 가설(H3a의 d1, h별)은 Benjamini–Hochberg 보정 [R7]

입력: data/processed/event_panel.parquet, data/raw/move/move_daily.csv (없으면 H3d 건너뜀)
출력: outputs/tables/rq3_regressions.csv, rq3_coverage.csv, rq3_move.csv
실행: python src/analysis/rq3_width.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, RAW_DIR, get_logger, load_settings  # noqa: E402
from stats_utils import cluster_bootstrap_mean, cluster_ols, series_design, wild_cluster_p  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
SEED = 20261104
WIDTHS = ["W_iqr", "W_iqr_disc", "W_sd"]   # 첫 번째가 주 지표


def regress(df, y, xs, label, h, width, hyp, rng, logger) -> dict:
    d = df.dropna(subset=[y] + xs)
    X = series_design(d, xs)
    cl = d["release_date_actual"].astype(str).to_numpy()
    fit = cluster_ols(d[y], X, cl)
    key = xs[0]
    p_wild = wild_cluster_p(d[y], X, cl, key, rng)
    res = {"hypothesis": hyp, "sample": label, "h": h, "width": width, "n": len(d), "n_dates": len(set(cl)),
           "coef": fit.params[key], "se": fit.bse[key], "p_cluster": fit.pvalues[key], "p_wild": p_wild,
           "r2": fit.rsquared}
    if len(xs) > 1:
        res.update({f"coef_{xs[1]}": fit.params[xs[1]], f"p_{xs[1]}": fit.pvalues[xs[1]]})
    logger.info("  %s %s h=%d [%s]: 계수 %.3f (SE %.3f, wild p=%.3f)%s, R²=%.2f, n=%d",
                hyp, label, h, width, res["coef"], res["se"], p_wild,
                f", {xs[1]} 계수 {res[f'coef_{xs[1]}']:.3f}" if len(xs) > 1 else "", res["r2"], len(d))
    return res


def coverage(df, label, h, rng, logger) -> list[dict]:
    rows = []
    d = df.dropna(subset=["cover50", "cover80"])
    for col, nominal in [("cover50", 0.5), ("cover80", 0.8)]:
        lo, hi = cluster_bootstrap_mean(d[col], d["release_date_actual"].astype(str), rng)
        rate = d[col].mean()
        rows.append({"sample": label, "h": h, "interval": col, "nominal": nominal, "rate": rate,
                     "ci_lo": lo, "ci_hi": hi, "n": len(d), "nominal_in_ci": lo <= nominal <= hi})
        logger.info("  H3c %s h=%d %s: 적중률 %.0f%% (95%% CI %.0f~%.0f%%, 명목 %.0f%%) n=%d",
                    label, h, col, 100 * rate, 100 * lo, 100 * hi, 100 * nominal, len(d))
    return rows


def move_panel(panel: pd.DataFrame, logger) -> pd.DataFrame | None:
    path = RAW_DIR / "move" / "move_daily.csv"
    if not path.exists():
        logger.warning("  MOVE 파일 없음 (%s). scraping/07_move.py를 먼저 실행하면 H3d도 계산해", path)
        return None
    mv = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    mv["dMOVE"] = mv["value"].diff()
    mv["MOVE_lag"] = mv["value"].shift(1)
    mv["dMOVE_lag"] = mv["dMOVE"].shift(1)
    d = panel.copy()
    d["date"] = pd.to_datetime(d["release_date_actual"])
    d["absS"] = d["S"].abs()
    # 같은 날 여러 지표가 발표되면 폭과 서프라이즈 크기를 평균해 발표일 1행으로
    g = d.groupby(["date", "h"]).agg(W_iqr=("W_iqr", "mean"), W_iqr_disc=("W_iqr_disc", "mean"),
                                     W_sd=("W_sd", "mean"), absS=("absS", "mean"),
                                     n_series=("series", "nunique")).reset_index()
    out = g.merge(mv[["date", "dMOVE", "MOVE_lag", "dMOVE_lag"]], on="date", how="inner")
    logger.info("  MOVE와 연결된 발표일 %d / %d", out["date"].nunique(), g["date"].nunique())
    return out


def main() -> None:
    cfg = load_settings()
    primary = cfg.get("rq1", {}).get("primary_series", ["KXCPI", "KXCPICORE"])
    logger = get_logger("rq3_width")
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logger.info("[1/4] 패널 로드")
    panel = pd.read_parquet(PROCESSED / "event_panel.parquet").dropna(subset=["S", "Gap", "W_iqr"])
    panel["absS"], panel["absGap"] = panel["S"].abs(), panel["Gap"].abs()
    samples = {"primary": panel[panel["series"].isin(primary)]}
    for s in sorted(panel["series"].unique()):
        samples[s] = panel[panel["series"] == s]
    samples["all"] = panel
    logger.info("  분석 가능 행 %d | 주 분석 표본 %s", len(panel), primary)

    logger.info("[2/4] H3a (|S| ~ 폭 + |Gap|), H3b (E_K ~ 폭)")
    regs = []
    for label, df in samples.items():
        for h in sorted(df["h"].unique()):
            d = df[df["h"] == h]
            for w in WIDTHS:
                if label != "primary" and w != WIDTHS[0]:
                    continue            # 강건성 지표는 주 분석 표본에서만
                regs.append(regress(d, "absS", [w, "absGap"], label, h, w, "H3a", rng, logger))
                regs.append(regress(d, "E_K", [w], label, h, w, "H3b", rng, logger))
    regs = pd.DataFrame(regs)
    mask = (regs["sample"] == "primary") & (regs["hypothesis"] == "H3a") & (regs["width"] == WIDTHS[0])
    regs.loc[mask, "p_bh"] = multipletests(regs.loc[mask, "p_wild"], method="fdr_bh")[1]
    logger.info("  주 분석 H3a BH 보정 p: %s", np.round(regs.loc[mask, "p_bh"], 3).tolist())

    logger.info("[3/4] H3c 예측구간 적중률")
    cov = pd.DataFrame([r for label, df in samples.items() for h in sorted(df["h"].unique())
                        for r in coverage(df[df["h"] == h], label, h, rng, logger)])

    logger.info("[4/4] H3d MOVE 변화 (발표일 단위)")
    mrows = []
    for label in ["primary", "all"]:
        mp = move_panel(samples[label], logger)
        if mp is None:
            break
        for h in sorted(mp["h"].unique()):
            d = mp[mp["h"] == h].dropna().assign(series=label, release_date_actual=lambda x: x["date"])
            for w in WIDTHS:
                if label != "primary" and w != WIDTHS[0]:
                    continue
                mrows.append(regress(d, "dMOVE", [w, "absS", "MOVE_lag", "dMOVE_lag"], label, h, w, "H3d", rng, logger))
    mres = pd.DataFrame(mrows)

    regs.to_csv(TABLES / "rq3_regressions.csv", index=False)
    cov.to_csv(TABLES / "rq3_coverage.csv", index=False)
    mres.to_csv(TABLES / "rq3_move.csv", index=False)
    logger.info("저장 outputs/tables/rq3_regressions.csv, rq3_coverage.csv, rq3_move.csv")


if __name__ == "__main__":
    main()
