"""RQ1: Kalshi 평균은 컨센서스 대비 추가적인 발표 예측력을 갖는가.

가설 (연구설계 수정본)
  H1a  S = a + b·Gap + 지표 고정효과 + e 에서 b > 0                       [R1][R6]
  H1b  표본외에서 결합 예측(M2)의 RMSE가 컨센서스(M0)보다 작다             ※
  H1c  위 개선이 h=7보다 h=1에서 크다                                      ※ ([R6] 시점 설계 확장)

예측 모형 (모두 σ 단위 오차로 비교. σ로 나누면 지표 간 비교가 가능)
  M0  Â = F                    오차 e0 = S
  M1  Â = K                    오차 e1 = S − Gap
  M2  Â = F + σ·(â_k + b̂·Gap)  오차 e2 = S − Ŝ,  â_k·b̂는 발표일 이전 이벤트만으로 추정(expanding)
      같은 날 발표된 지표(CPI와 core CPI)는 학습에서 서로 제외해 미래 정보가 섞이지 않게 함

추론
  - 표본내 회귀: 발표일 군집 표준오차 + wild cluster bootstrap(Rademacher, 귀무가설 b=0 부과) p값   [R5]
  - 표본외 M2 vs M0: Clark & West(2007) 중첩모형 검정(단측)                                         [R7]
  - RMSE 비율 신뢰구간: 발표일 군집 단위 bootstrap                                                  ※
  - 1차 가설(H1a·H1b, h별)은 Benjamini–Hochberg로 보정                                             [R7]

표본 구분
  primary  : CPI + core CPI 통합 (주 분석, 계획서 기준)
  시리즈별  : 각 지표 단독
  all      : 다섯 지표 통합 (확장 분석)

입력: data/processed/event_panel.parquet
출력: outputs/tables/rq1_insample.csv, rq1_oos.csv, rq1_h1c.csv, data/processed/rq1_oos_predictions.parquet
실행: python src/analysis/rq1_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
from common import PROJECT_ROOT, get_logger, load_settings  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
SEED = 20261104


# ---------------------------------------------------------------- 공통 도구
def design(df: pd.DataFrame, series_fe: bool) -> pd.DataFrame:
    """설계행렬: 지표 고정효과(시리즈 더미, 상수항 대신) 또는 상수항 + Gap."""
    if series_fe and df["series"].nunique() > 1:
        X = pd.get_dummies(df["series"], prefix="fe", dtype=float)
    else:
        X = pd.DataFrame({"const": 1.0}, index=df.index)
    X["Gap"] = df["Gap"].to_numpy()
    return X


def wild_cluster_p(y, X, clusters, rng, reps=1999) -> float:
    """Gap 계수에 대한 wild cluster bootstrap-t p값 (양측, 귀무가설 b=0을 부과한 잔차 사용)."""
    full = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t_obs = full.tvalues["Gap"]
    Xr = X.drop(columns="Gap")
    restricted = sm.OLS(y, Xr).fit()
    fitted, resid = restricted.fittedvalues.to_numpy(), restricted.resid.to_numpy()
    codes, uniq = pd.factorize(clusters)
    t_star = np.empty(reps)
    for r in range(reps):
        signs = rng.choice([-1.0, 1.0], size=len(uniq))[codes]
        yb = fitted + resid * signs
        t_star[r] = sm.OLS(yb, X).fit(cov_type="cluster", cov_kwds={"groups": clusters}).tvalues["Gap"]
    return float((np.abs(t_star) >= abs(t_obs)).mean())


def clark_west(e0, e2, f0, f2, clusters) -> tuple[float, float]:
    """Clark & West(2007): f = e0² − (e2² − (f0 − f2)²)의 평균이 0보다 큰지 단측 검정.
    표준오차는 발표일 군집 기준 (CPI와 core CPI가 같은 날 발표되므로)."""
    f = e0 ** 2 - (e2 ** 2 - (f0 - f2) ** 2)
    res = sm.OLS(f, np.ones((len(f), 1))).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t = float(np.asarray(res.tvalues)[0])
    return t, float(1 - norm.cdf(t))


def cluster_bootstrap_ratio(df, num_col, den_col, rng, reps=2000) -> tuple[float, float]:
    """RMSE(num)/RMSE(den) 비율의 95% 신뢰구간 (발표일 단위로 재표집)."""
    groups = [g for _, g in df.groupby("release_date_actual")]
    out = np.empty(reps)
    for r in range(reps):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        b = pd.concat(pick)
        out[r] = np.sqrt((b[num_col] ** 2).mean()) / np.sqrt((b[den_col] ** 2).mean())
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ---------------------------------------------------------------- 표본내 회귀 (H1a)
def insample(df: pd.DataFrame, label: str, h: int, rng, logger) -> dict:
    X = design(df, series_fe=True)
    y = df["S"]
    cl = df["release_date_actual"].astype(str).to_numpy()
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cl})
    p_wild = wild_cluster_p(y, X, cl, rng)
    res = {"sample": label, "h": h, "n_events": len(df), "n_dates": len(set(cl)),
           "b_Gap": fit.params["Gap"], "se_cluster": fit.bse["Gap"], "t_cluster": fit.tvalues["Gap"],
           "p_cluster": fit.pvalues["Gap"], "p_wild": p_wild, "r2": fit.rsquared}
    logger.info("  [표본내] %s h=%d: b=%.3f (군집 SE %.3f, p=%.3f, wild p=%.3f), R²=%.2f, 이벤트 %d / 발표일 %d",
                label, h, res["b_Gap"], res["se_cluster"], res["p_cluster"], p_wild, res["r2"], len(df), len(set(cl)))
    return res


# ---------------------------------------------------------------- 표본외 비교 (H1b)
def out_of_sample(df: pd.DataFrame, label: str, h: int, min_train: int, rng, logger) -> tuple[dict, pd.DataFrame]:
    df = df.sort_values("release_date_actual").reset_index(drop=True)
    preds = []
    for i, row in df.iterrows():
        train = df[df["release_date_actual"] < row["release_date_actual"]]
        if len(train) < min_train or (train["series"] == row["series"]).sum() < 3:
            continue
        X = design(train, series_fe=True)
        fit = sm.OLS(train["S"], X).fit()
        fe_col = f"fe_{row['series']}"
        a = fit.params[fe_col] if fe_col in fit.params else fit.params.get("const", 0.0)
        s_hat = a + fit.params["Gap"] * float(row["Gap"])
        preds.append({"event_ticker": row["event_ticker"], "series": row["series"],
                      "release_date_actual": row["release_date_actual"], "h": h, "sample": label,
                      "S": row["S"], "Gap": row["Gap"], "S_hat": s_hat, "n_train": len(train)})
    p = pd.DataFrame(preds)
    if p.empty:
        logger.warning("  [표본외] %s h=%d: 학습 표본 부족으로 예측 없음", label, h)
        return {}, p
    p["e0"], p["e1"], p["e2"] = p["S"], p["S"] - p["Gap"], p["S"] - p["S_hat"]
    rmse = {k: float(np.sqrt((p[k] ** 2).mean())) for k in ["e0", "e1", "e2"]}
    mae = {k: float(p[k].abs().mean()) for k in ["e0", "e1", "e2"]}
    cl = p["release_date_actual"].astype(str).to_numpy()
    t_cw, p_cw = clark_west(p["e0"].to_numpy(), p["e2"].to_numpy(), np.zeros(len(p)), p["S_hat"].to_numpy(), cl)
    lo2, hi2 = cluster_bootstrap_ratio(p, "e2", "e0", rng)
    lo1, hi1 = cluster_bootstrap_ratio(p, "e1", "e0", rng)
    res = {"sample": label, "h": h, "n_oos": len(p), "first_oos_date": p["release_date_actual"].min(),
           "rmse_M0": rmse["e0"], "rmse_M1": rmse["e1"], "rmse_M2": rmse["e2"],
           "mae_M0": mae["e0"], "mae_M1": mae["e1"], "mae_M2": mae["e2"],
           "improve_M1": 1 - rmse["e1"] / rmse["e0"], "improve_M2": 1 - rmse["e2"] / rmse["e0"],
           "ratio_M1_ci_lo": lo1, "ratio_M1_ci_hi": hi1, "ratio_M2_ci_lo": lo2, "ratio_M2_ci_hi": hi2,
           "cw_t": t_cw, "cw_p": p_cw}
    logger.info("  [표본외] %s h=%d: 예측 %d건 | RMSE M0 %.3f, M1 %.3f, M2 %.3f | 개선율 M1 %+.1f%%, M2 %+.1f%% "
                "(M2/M0 비율 95%% CI %.2f~%.2f) | Clark–West t=%.2f, p=%.3f",
                label, h, len(p), rmse["e0"], rmse["e1"], rmse["e2"], 100 * res["improve_M1"],
                100 * res["improve_M2"], lo2, hi2, t_cw, p_cw)
    return res, p


# ---------------------------------------------------------------- H1c
def h1c(preds: pd.DataFrame, label: str, rng, logger, reps=2000) -> dict:
    """같은 이벤트에서 h=1과 h=7의 M2 개선율 차이와 발표일 단위 bootstrap 신뢰구간."""
    p = preds[preds["sample"] == label]
    wide = p.pivot_table(index=["event_ticker", "release_date_actual"], columns="h", values=["e0", "e2"]).dropna()
    if wide.empty:
        return {}
    wide.columns = [f"{a}_h{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    def diff(d):
        imp = lambda h: 1 - np.sqrt((d[f"e2_h{h}"] ** 2).mean()) / np.sqrt((d[f"e0_h{h}"] ** 2).mean())
        return imp(1) - imp(7)

    est = diff(wide)
    groups = [g for _, g in wide.groupby("release_date_actual")]
    boot = np.array([diff(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                     for _ in range(reps)])
    res = {"sample": label, "n_events": len(wide), "improve_diff_h1_minus_h7": est,
           "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
           "share_boot_positive": float((boot > 0).mean())}
    logger.info("  [H1c] %s: 개선율 차이(h1−h7) %+.1f%%p, 95%% CI %+.1f~%+.1f%%p, 이벤트 %d",
                label, 100 * est, 100 * res["ci_lo"], 100 * res["ci_hi"], len(wide))
    return res


def main() -> None:
    cfg = load_settings()
    acfg = cfg.get("rq1", {})
    primary = acfg.get("primary_series", ["KXCPI", "KXCPICORE"])
    min_train = acfg.get("min_train", 12)
    logger = get_logger("rq1_forecast")
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logger.info("[1/4] 패널 로드")
    panel = pd.read_parquet(PROCESSED / "event_panel.parquet").dropna(subset=["S", "Gap"])
    panel["release_date_actual"] = pd.to_datetime(panel["release_date_actual"])
    logger.info("  분석 가능 행 %d (이벤트×h) | 주 분석 표본 %s | 최소 학습 이벤트 %d", len(panel), primary, min_train)

    samples = {"primary": panel[panel["series"].isin(primary)]}
    for s in sorted(panel["series"].unique()):
        samples[s] = panel[panel["series"] == s]
    samples["all"] = panel

    ins, oos, preds = [], [], []
    logger.info("[2/4] 표본내 회귀 (H1a)와 [3/4] 표본외 비교 (H1b)")
    for label, df in samples.items():
        for h in sorted(df["h"].unique()):
            d = df[df["h"] == h]
            ins.append(insample(d, label, h, rng, logger))
            r, p = out_of_sample(d, label, h, min_train, rng, logger)
            if r:
                oos.append(r)
                preds.append(p)
    ins, oos, preds = pd.DataFrame(ins), pd.DataFrame(oos), pd.concat(preds, ignore_index=True)

    # 1차 가설(주 분석 표본의 H1a·H1b, h별) Benjamini–Hochberg 보정
    pri_ins = ins["sample"] == "primary"
    pri_oos = oos["sample"] == "primary"
    pvals = np.concatenate([ins.loc[pri_ins, "p_wild"].to_numpy(), oos.loc[pri_oos, "cw_p"].to_numpy()])
    adj = multipletests(pvals, method="fdr_bh")[1]
    ins.loc[pri_ins, "p_bh"] = adj[:pri_ins.sum()]
    oos.loc[pri_oos, "p_bh"] = adj[pri_ins.sum():]
    logger.info("  주 분석 1차 가설 BH 보정 p: 표본내 %s, 표본외 %s",
                np.round(adj[:pri_ins.sum()], 3).tolist(), np.round(adj[pri_ins.sum():], 3).tolist())

    logger.info("[4/4] H1c (h=1 vs h=7)")
    h1c_rows = [r for r in (h1c(preds, lab, rng, logger) for lab in samples) if r]

    ins.to_csv(TABLES / "rq1_insample.csv", index=False)
    oos.to_csv(TABLES / "rq1_oos.csv", index=False)
    pd.DataFrame(h1c_rows).to_csv(TABLES / "rq1_h1c.csv", index=False)
    preds.to_parquet(PROCESSED / "rq1_oos_predictions.parquet", index=False)
    logger.info("저장 outputs/tables/rq1_insample.csv, rq1_oos.csv, rq1_h1c.csv, data/processed/rq1_oos_predictions.parquet")


if __name__ == "__main__":
    main()
