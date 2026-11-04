"""RQ1 그림 2개 (rq1_forecast.py가 만든 결과 파일을 읽어서 그림).

  그림 1  outputs/figures/rq1_oos_improvement.png
          표본별 표본외 RMSE 개선율(결합 예측 M2 vs 컨센서스 M0)과 95% 신뢰구간, h=7과 h=1 비교
  그림 2  outputs/figures/rq1_gap_vs_surprise.png
          주 분석 표본(CPI·core CPI), h=1에서 Kalshi–컨센서스 괴리(Gap)와 표준화 서프라이즈(S)의 산점도와 회귀선

색은 dataviz 기본 범주 팔레트의 1·2번(파랑·주황)을 고정 순서로 사용하고, 범례와 직접 라벨을 함께 둠.
실행: python src/analysis/rq1_figures.py   (rq1_forecast.py 마지막 단계에서도 자동 실행)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
from common import PROJECT_ROOT, get_logger  # noqa: E402

TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# dataviz 기본 팔레트(라이트 모드): 범주 1·2번, 글자·격자·배경 색
SERIES = ["#2a78d6", "#eb6834"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"primary": "주 분석 (CPI+core)", "KXCPI": "CPI", "KXCPICORE": "core CPI",
          "KXU3": "실업률", "KXPAYROLLS": "비농업 고용", "KXGDP": "GDP", "all": "전체 통합"}


def set_style() -> None:
    """한글 글꼴(맥 AppleGothic, 윈도 Malgun Gothic, 리눅스 Nanum/Noto 순)과 차분한 축 스타일."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    })


def fig_improvement(oos: pd.DataFrame, out: Path, min_oos: int = 10) -> None:
    d = oos[oos["n_oos"] >= min_oos].copy()                       # 표본외 예측이 너무 적은 표본(GDP 등)은 제외
    d["imp"] = 100 * d["improve_M2"]
    d["lo"], d["hi"] = 100 * (1 - d["ratio_M2_ci_hi"]), 100 * (1 - d["ratio_M2_ci_lo"])
    order = [s for s in LABELS if s in set(d["sample"])]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(order) + 1.4))
    for k, (h, off) in enumerate([(7, 0.16), (1, -0.16)]):
        g = d[d["h"] == h].set_index("sample").reindex(order)
        ax.hlines(y + off, g["lo"], g["hi"], color=SERIES[k], linewidth=2)
        ax.scatter(g["imp"], y + off, s=48, color=SERIES[k], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=f"발표 {h}일 전 (h={h})")
        top = order[0]
        ax.annotate(f"h={h}", (g.loc[top, "hi"], y[0] + off), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, [f"{LABELS[s]}  (n={int(d[(d['sample'] == s) & (d['h'] == 1)]['n_oos'].iloc[0])})" for s in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("표본외 RMSE 개선율 (%), 결합 예측(M2) vs 컨센서스(M0)\n점: 추정치, 선: 발표일 단위 bootstrap 95% 신뢰구간")
    ax.set_title("Kalshi 정보를 결합하면 컨센서스보다 발표치를 얼마나 더 잘 맞히나", loc="left", fontsize=12, color=TEXT)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_scatter(panel: pd.DataFrame, primary: list[str], out: Path, h: int = 1) -> None:
    d = panel[(panel["h"] == h) & panel["series"].isin(primary)].dropna(subset=["Gap", "S"])
    b, a = np.polyfit(d["Gap"], d["S"], 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for k, s in enumerate(primary):
        g = d[d["series"] == s]
        ax.scatter(g["Gap"], g["S"], s=40, color=SERIES[k], alpha=0.85, edgecolor=SURFACE, linewidth=1.5,
                   label=f"{LABELS[s]} (n={len(g)})", zorder=3)
    xs = np.linspace(d["Gap"].min(), d["Gap"].max(), 50)
    ax.plot(xs, a + b * xs, color=TEXT_2, linewidth=2, zorder=2)
    ax.annotate(f"통합 회귀선 (기울기 {b:.2f})", (xs[-1], a + b * xs[-1]), xytext=(0, -30), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_2)
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1, zorder=1)
    ax.set_xlabel("Gap = (Kalshi 평균 − 컨센서스) / σ")
    ax.set_ylabel("S = (실제치 − 컨센서스) / σ")
    ax.set_title(f"Kalshi가 컨센서스보다 높게 보면 실제 발표도 높았다 (발표 {h}일 전)", loc="left", fontsize=12, color=TEXT)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make_figures(primary: list[str], logger=None) -> None:
    logger = logger or get_logger("rq1_figures")
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    oos = pd.read_csv(TABLES / "rq1_oos.csv")
    panel = pd.read_parquet(PROCESSED / "event_panel.parquet")
    p1, p2 = FIGURES / "rq1_oos_improvement.png", FIGURES / "rq1_gap_vs_surprise.png"
    fig_improvement(oos, p1)
    fig_scatter(panel, primary, p2)
    logger.info("  그림 저장: %s, %s", p1.name, p2.name)


if __name__ == "__main__":
    from common import load_settings
    make_figures(load_settings().get("rq1", {}).get("primary_series", ["KXCPI", "KXCPICORE"]))
