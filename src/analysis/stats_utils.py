"""RQ2·RQ3에서 함께 쓰는 통계 도구."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def series_design(df: pd.DataFrame, regressors: list[str], series_fe: bool = True) -> pd.DataFrame:
    """지표 고정효과(시리즈 더미, 상수항 대신) 또는 상수항 + 설명변수."""
    if series_fe and df["series"].nunique() > 1:
        X = pd.get_dummies(df["series"], prefix="fe", dtype=float)
    else:
        X = pd.DataFrame({"const": 1.0}, index=df.index)
    for r in regressors:
        X[r] = df[r].to_numpy(dtype=float)
    return X


def cluster_ols(y, X, clusters):
    """발표일 군집 표준오차 OLS. 군집이 관측치 수와 같으면 HC1과 사실상 같음."""
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})


def wild_cluster_p(y, X, clusters, coef: str, rng, reps: int = 1999) -> float:
    """coef = 0 귀무가설을 부과한 wild cluster bootstrap-t p값 (Rademacher, 양측).  [R5]의 wild bootstrap 사용 근거"""
    t_obs = cluster_ols(y, X, clusters).tvalues[coef]
    restricted = sm.OLS(y, X.drop(columns=coef)).fit()
    fitted, resid = restricted.fittedvalues.to_numpy(), restricted.resid.to_numpy()
    codes, uniq = pd.factorize(np.asarray(clusters))
    t_star = np.empty(reps)
    for r in range(reps):
        yb = fitted + resid * rng.choice([-1.0, 1.0], size=len(uniq))[codes]
        t_star[r] = cluster_ols(yb, X, clusters).tvalues[coef]
    return float((np.abs(t_star) >= abs(t_obs)).mean())


def tost(est: float, se: float, margin: float) -> float:
    """동등성 검정(TOST): |계수| < margin 을 대립가설로 하는 p값 (정규 근사).  ※ 근거 문헌 미확보"""
    p_low = 1 - norm.cdf((est + margin) / se)
    p_high = norm.cdf((est - margin) / se)
    return float(max(p_low, p_high))


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series, rng, reps: int = 2000) -> tuple[float, float]:
    """평균(예: 적중률)의 발표일 단위 bootstrap 95% 신뢰구간."""
    df = pd.DataFrame({"v": values.to_numpy(dtype=float), "c": clusters.to_numpy()})
    groups = [g["v"].to_numpy() for _, g in df.groupby("c")]
    stats = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        stats[r] = np.concatenate([groups[i] for i in pick]).mean()
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
