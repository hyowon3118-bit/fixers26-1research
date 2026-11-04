"""RQ2: Kalshi가 예측한 서프라이즈와 예측하지 못한 서프라이즈는 금리 변화와 어떻게 연결되는가.

모형 (연구설계 수정본, Campbell & Sharpe 분해 [R1])
  ΔY = α + c₁·Ŝ + c₂·u + e,   u = S − Ŝ
  - Ŝ: RQ1의 표본외 예측값 (그 발표일 이전 이벤트만으로 추정한 값이라 미래 정보가 섞이지 않음) [R1][R7]
       CPI·core CPI는 주 분석 통합 모형의 예측, 나머지 지표는 각 지표 단독 모형의 예측을 씀
  - 지표마다 계수를 따로 추정 (CPI와 실업률은 같은 서프라이즈라도 금리 반응 방향이 다를 수 있음) ※
  - CPI·core CPI 통합 추정은 지표별 절편과 발표일 군집 표준오차 사용

종속변수 (bp)
  dY_rel  발표일 미국채 2년 변화 = Y(발표일) − Y(직전 거래일)          일별 대체는 ※ ([R1]은 분 단위)
  dY10    같은 방식의 미국채 10년 변화 (보조)                        ※
  dKTB    국고채 2년, 미국 발표(한국 시간 밤) 다음 한국 거래일 변화     ※
  dY_pre  발표 전 1주 변화 = Y(직전 거래일) − Y(발표일 7일 전 이하 마지막 거래일).
          이때는 시간 순서를 맞추려고 h=7 시점의 Ŝ만 씀                 ※

판정 (c₁)
  - 귀무가설 c₁ = 0 검정 (군집 SE, wild cluster bootstrap p값 [R5])
  - 동등성 검정(TOST): |c₁| < margin 이면 "사전 반영과 일관"이라고 말할 근거. margin은 settings.yaml rq2.equivalence_margin_bp ※
  - 결과 분류는 연구설계 수정본의 판정표를 따름

입력: data/processed/event_panel.parquet, rq1_oos_predictions.parquet, data/raw/rates/*.csv
출력: outputs/tables/rq2_regressions.csv
실행: python src/analysis/rq2_rates.py   (rq1_forecast.py를 먼저 실행해야 함)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, RAW_DIR, get_logger, load_settings  # noqa: E402
from stats_utils import cluster_ols, series_design, tost, wild_cluster_p  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
SEED = 20261104


def load_rate(name: str) -> pd.Series:
    """FRED(date, value)·ECOS(YYYYMMDD) 일별 금리 → 날짜 인덱스 시리즈(결측 행 제거)."""
    df = pd.read_csv(RAW_DIR / "rates" / name, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"].astype(str), format="mixed")
    return df.dropna(subset=["value"]).set_index("date")["value"].astype(float).sort_index()


def change_on(y: pd.Series, day: pd.Timestamp) -> float:
    """day의 값 − 직전 거래일 값 (bp). day가 휴장일이면 결측."""
    if day not in y.index:
        return np.nan
    prev = y[y.index < day]
    return np.nan if prev.empty else 100 * (y[day] - prev.iloc[-1])


def change_next_kr(y: pd.Series, us_day: pd.Timestamp) -> float:
    """미국 발표일(한국 시간 밤) 다음 한국 거래일 값 − 미국 발표일 이하 마지막 한국 거래일 값 (bp)."""
    after, before = y[y.index > us_day], y[y.index <= us_day]
    return np.nan if after.empty or before.empty else 100 * (after.iloc[0] - before.iloc[-1])


def change_pre(y: pd.Series, day: pd.Timestamp, days: int = 7) -> float:
    """직전 거래일 값 − 발표일 days일 전 이하 마지막 거래일 값 (bp)."""
    prev, base = y[y.index < day], y[y.index <= day - pd.Timedelta(days=days)]
    return np.nan if prev.empty or base.empty else 100 * (prev.iloc[-1] - base.iloc[-1])


def build_panel(primary: list[str], logger) -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "rq1_oos_predictions.parquet")
    # CPI 계열은 주 분석 통합 모형, 나머지는 지표 단독 모형의 표본외 예측
    use = preds[((preds["sample"] == "primary") & preds["series"].isin(primary)) |
                ((preds["sample"] == preds["series"]) & ~preds["series"].isin(primary))].copy()
    use["u"] = use["S"] - use["S_hat"]
    use["day"] = pd.to_datetime(use["release_date_actual"])
    y2, y10, ktb = load_rate("fred_DGS2.csv"), load_rate("fred_DGS10.csv"), load_rate("ecos_ktb2y.csv")
    use["dY_rel"] = [change_on(y2, d) for d in use["day"]]
    use["dY10"] = [change_on(y10, d) for d in use["day"]]
    use["dKTB"] = [change_next_kr(ktb, d) for d in use["day"]]
    use["dY_pre"] = [change_pre(y2, d) for d in use["day"]]
    logger.info("  이벤트×h %d행 | 금리 변화 결측: 2년 %d, 10년 %d, 국고채 %d, 사전 %d",
                len(use), use["dY_rel"].isna().sum(), use["dY10"].isna().sum(),
                use["dKTB"].isna().sum(), use["dY_pre"].isna().sum())
    return use


def estimate(d: pd.DataFrame, yvar: str, label: str, h: int, margin: float, rng, logger) -> dict | None:
    d = d.dropna(subset=[yvar, "S_hat", "u"])
    if len(d) < 10:
        return None
    X = series_design(d, ["S_hat", "u"])
    cl = d["day"].astype(str).to_numpy()
    fit = cluster_ols(d[yvar], X, cl)
    p1 = wild_cluster_p(d[yvar], X, cl, "S_hat", rng)
    p2 = wild_cluster_p(d[yvar], X, cl, "u", rng)
    c1, se1 = fit.params["S_hat"], fit.bse["S_hat"]
    p_eq = tost(c1, se1, margin)
    sig1, sig2 = p1 < 0.05, p2 < 0.05
    if not sig1 and sig2:
        verdict = "사전 반영과 일관" + (" (동등성 검정 통과)" if p_eq < 0.05 else " (동등성은 미확인: 신뢰구간이 넓음)")
    elif sig1 and sig2:
        verdict = "예측 가능한 정보도 발표일 금리와 연관"
    elif not sig1 and not sig2:
        verdict = "금리 반응 미검출, 사전 반영 입증 못 함"
    else:
        verdict = "Ŝ만 유의: 모형·정보 시점 재검토 필요"
    res = {"y": yvar, "sample": label, "h": h, "n": len(d), "n_dates": len(set(cl)),
           "c1_S_hat": c1, "se_c1": se1, "p_c1_wild": p1, "ci90_c1_lo": c1 - 1.645 * se1, "ci90_c1_hi": c1 + 1.645 * se1,
           "p_c1_equiv": p_eq, "c2_u": fit.params["u"], "se_c2": fit.bse["u"], "p_c2_wild": p2,
           "r2": fit.rsquared, "verdict": verdict}
    logger.info("  %s %s h=%d: c1(Ŝ)=%.2f (wild p=%.3f, 동등성 p=%.3f) c2(u)=%.2f (wild p=%.3f), n=%d → %s",
                yvar, label, h, c1, p1, p_eq, res["c2_u"], p2, len(d), verdict)
    return res


def main() -> None:
    cfg = load_settings()
    primary = cfg.get("rq1", {}).get("primary_series", ["KXCPI", "KXCPICORE"])
    margin = cfg.get("rq2", {}).get("equivalence_margin_bp", 1.0)
    logger = get_logger("rq2_rates")
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logger.info("[1/3] Ŝ·u와 금리 변화 결합 (동등성 기준 ±%.1fbp / 1σ)", margin)
    use = build_panel(primary, logger)

    logger.info("[2/3] 발표일 반응: h=1의 Ŝ 사용")
    groups = {"primary": use[use["series"].isin(primary)]}
    groups.update({s: use[use["series"] == s] for s in sorted(use["series"].unique())})
    rows = []
    for label, g in groups.items():
        for yvar in ["dY_rel", "dY10", "dKTB"]:
            r = estimate(g[g["h"] == 1], yvar, label, 1, margin, rng, logger)
            if r:
                rows.append(r)

    logger.info("[3/3] 발표 전 1주 반응: h=7의 Ŝ 사용")
    for label, g in groups.items():
        r = estimate(g[g["h"] == 7], "dY_pre", label, 7, margin, rng, logger)
        if r:
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "rq2_regressions.csv", index=False)
    logger.info("저장 outputs/tables/rq2_regressions.csv (%d행)", len(out))


if __name__ == "__main__":
    main()
