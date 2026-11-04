"""RQ1 이벤트 패널을 지표별·horizon별 그림과 요약표로 시각화.

입력: data/processed/event_panel.parquet (src/features/01_event_panel.py 출력)
출력: outputs/figures/rq1_{series}.png, outputs/tables/rq1_summary.csv
실행: python src/analysis/01_rq1_visualizations.py

회귀선과 상관계수는 탐색용 기술통계다. 이 그림만으로 예측력이나 인과성을 결론내리지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "processed" / "event_panel.parquet"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

NAMES = {
    "KXCPI": "Headline CPI m/m",
    "KXCPICORE": "Core CPI m/m",
    "KXU3": "Unemployment rate",
    "KXGDP": "Advance GDP q/q",
    "KXPAYROLLS": "Nonfarm payrolls",
}
COLORS = {7: "#2878B5", 1: "#E07A28"}


def make_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, h), g in panel.groupby(["series", "h"], sort=True):
        valid = g.dropna(subset=["S", "Gap"])
        slope = np.nan
        if len(valid) >= 2 and valid["Gap"].nunique() > 1:
            slope = float(np.polyfit(valid["Gap"], valid["S"], 1)[0])
        rows.append({
            "series": series, "indicator": NAMES.get(series, series), "h": int(h),
            "n_events": len(g), "n_valid_gap_surprise": len(valid),
            "corr_gap_surprise": valid["Gap"].corr(valid["S"]) if len(valid) >= 2 else np.nan,
            "descriptive_slope": slope,
            "median_W_iqr": g["W_iqr"].median(),
            "median_W_sd": g["W_sd"].median(),
            "cover50_rate": g["cover50"].mean(), "cover80_rate": g["cover80"].mean(),
        })
    return pd.DataFrame(rows)


def plot_series(panel: pd.DataFrame, series: str, output: Path) -> None:
    g = panel[panel["series"] == series].copy()
    if g.empty:
        return
    g["release_date_actual"] = pd.to_datetime(g["release_date_actual"], errors="coerce")
    horizons = sorted(g["h"].dropna().unique(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # RQ1 핵심: Kalshi-컨센서스 괴리와 표준화 발표 서프라이즈의 관계
    ax = axes[0]
    for h in horizons:
        sub = g[(g["h"] == h)].dropna(subset=["Gap", "S"])
        color = COLORS.get(int(h), None)
        ax.scatter(sub["Gap"], sub["S"], s=34, alpha=.78, color=color,
                   label=f"h={int(h)} days (n={len(sub)})")
        if len(sub) >= 2 and sub["Gap"].nunique() > 1:
            slope, intercept = np.polyfit(sub["Gap"], sub["S"], 1)
            xx = np.linspace(sub["Gap"].min(), sub["Gap"].max(), 100)
            ax.plot(xx, intercept + slope * xx, color=color, linewidth=1.6)
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.set(xlabel="Kalshi minus consensus gap (σ units)", ylabel="Realized surprise (σ units)",
           title="Gap vs. realized surprise")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)

    # 폭 지표의 시계열: 꼬리 때문에 식별되지 않은 연속 IQR은 결측으로 남는다.
    ax = axes[1]
    for h in horizons:
        sub = g[g["h"] == h].dropna(subset=["release_date_actual", "W_iqr"])
        ax.plot(sub["release_date_actual"], sub["W_iqr"], marker="o", markersize=3,
                linewidth=1.2, color=COLORS.get(int(h), None), label=f"h={int(h)} days")
    ax.set(xlabel="Release date", ylabel="Forecast IQR (σ units)", title="Distribution width")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    fig.suptitle(f"RQ1 / {NAMES.get(series, series)}", fontsize=13)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(f"Missing {PANEL}; build the event panel first.")
    panel = pd.read_parquet(PANEL)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for series in sorted(panel["series"].dropna().unique()):
        path = FIGURES / f"rq1_{series}.png"
        plot_series(panel, series, path)
        print(f"Saved {path.relative_to(ROOT)}")
    summary = make_summary(panel)
    path = TABLES / "rq1_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
