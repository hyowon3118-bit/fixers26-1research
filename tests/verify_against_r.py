"""파이썬 포팅(src/clean/kalshi_dkw.py)이 원본 R 코드와 같은 결과를 내는지 대조.

순서
  1) data/interim/kalshi_daily.parquet의 일별 마지막 체결가를 원본 R이 읽는 체결 자료 형식(csv)으로 변환
  2) 같은 csv를 원본 R 함수(tests/run_dkw_r.R)와 파이썬 포팅에 각각 넣음
  3) 구간 확률과 통계량을 이벤트·날짜·기준선 단위로 맞춰 최대 차이를 출력

필요: R과 tidyverse, matrixStats, collapse 패키지, 원본 저장소 clone
실행: python tests/verify_against_r.py --r-repo <원본저장소경로>
참고: 원본 R은 음수 기준선(T-0.1 등)을 읽지 못해서 대조는 0 이상 기준선만으로 해.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "clean"))
import kalshi_dkw as dkw  # noqa: E402

CASES = [
    # (시리즈, strike_int, horizon, moment_adjustment)
    ("KXCPI", 0.1, 30, 0.1),
    ("KXCPICORE", 0.1, 30, 0.1),
    ("KXU3", 0.1, 30, 0.1),
    ("KXPAYROLLS", 50000, 30, 0),  # 빈 기준선 채우기·swap 경로 점검용 (원본 논문에 없는 시리즈)
]


def to_trade_csv(daily: pd.DataFrame, path: Path) -> pd.DataFrame:
    d = daily.dropna(subset=["trade"]).copy()
    d = d[d["strike"] >= 0]
    d["ticker"] = d["contract_preamble"] + "-T" + d["strike"].map(lambda s: f"{s:g}")
    trades = pd.DataFrame({
        "created_time": d["date"].dt.strftime("%Y-%m-%dT12:00:00Z"),
        "ticker": d["ticker"],
        "yes_price_dollars": d["trade"],
        "no_price_dollars": 1 - d["trade"],
        "count_fp": d["daily_volume"].fillna(0),
    })
    trades.to_csv(path, index=False)
    return trades


def python_pipeline(trades: pd.DataFrame, strike_int, horizon, adj):
    t = trades.copy()
    t["date"] = pd.to_datetime(t["created_time"].str[:10])
    t["contract_preamble"] = t["ticker"].str.replace(r"-T-?\d+\.?\d*$", "", regex=True)
    t["strike"] = t["ticker"].str.extract(r"-T(-?\d+\.?\d*)$")[0].astype(float)
    t["yes_price"] = t["yes_price_dollars"] * 100
    daily = (t.sort_values("created_time").groupby(["date", "contract_preamble", "strike"])
             .agg(yes_price=("yes_price", "last"), daily_volume=("count_fp", "sum")).reset_index())
    filled = dkw.fill_dataless_days(daily, horizon)
    cleaned = dkw.clean_data(filled, "middle-out")
    dist = dkw.convert_to_probabilities(cleaned, strike_int, horizon)
    return dist, dkw.get_moments(dist, adj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-repo", required=True)
    args = ap.parse_args()
    daily = pd.read_parquet(ROOT / "data" / "interim" / "kalshi_daily.parquet")
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for series, si, hz, adj in CASES:
            sub = daily[daily["series"] == series]
            if sub.empty:
                print(f"[{series}] 자료 없음, 건너뜀")
                continue
            trades = to_trade_csv(sub, tmp / f"{series}.csv")
            r = subprocess.run(["Rscript", str(ROOT / "tests" / "run_dkw_r.R"), args.r_repo,
                                str(tmp / f"{series}.csv"), str(tmp / f"{series}_dist.csv"),
                                str(tmp / f"{series}_mom.csv"), str(si), str(hz), str(adj)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{series}] R 실행 실패:\n{r.stderr[-1500:]}")
                all_ok = False
                continue
            rd = pd.read_csv(tmp / f"{series}_dist.csv", parse_dates=["date"])
            rm = pd.read_csv(tmp / f"{series}_mom.csv", parse_dates=["date"])
            pd_, pm = python_pipeline(trades, si, hz, adj)

            key = ["contract_preamble", "date", "strike"]
            for df in (rd, pd_):
                df["strike"] = df["strike"].round(6)
            mdist = rd[key + ["probability"]].merge(pd_[key + ["probability"]], on=key, how="outer",
                                                    suffixes=("_r", "_py"), indicator=True)
            only = (mdist["_merge"] != "both").sum()
            pdiff = (mdist["probability_r"] - mdist["probability_py"]).abs().max()
            mkey = ["contract_preamble", "date"]
            mm = rm.merge(pm, on=mkey, suffixes=("_r", "_py"))
            # 누적확률이 부동소수점 오차 범위에서 정확히 50%인 분포는 중앙값이 두 값 사이 경계에 걸려,
            # R(확장 정밀도 합산)과 파이썬의 마지막 자리 차이로 다른 값을 고를 수 있어. 이런 경우는 따로 셈
            tie = (pd_.sort_values("strike").groupby(mkey)["probability"]
                   .apply(lambda p: bool((np.abs(np.cumsum(p.to_numpy()) / p.sum() - 0.5) < 1e-9).any()))
                   .rename("tie").reset_index())
            mm = mm.merge(tie, on=mkey, how="left")
            core = mm[~mm["tie"]]
            mdiffs = {c: float((core[f"{c}_r"] - core[f"{c}_py"]).abs().max())
                      for c in ["mean", "median", "mode", "variance", "skewness"]}
            scale = {c: max(1.0, float(rm[c].abs().max())) for c in mdiffs}
            n_tie_diff = int((mm["tie"] & ((mm["skewness_r"] - mm["skewness_py"]).abs() > 1e-8)).sum())
            ok = only == 0 and pdiff < 1e-8 and all(v / scale[c] < 1e-8 for c, v in mdiffs.items()) and len(mm) == len(rm) == len(pm)
            all_ok &= ok
            print(f"[{series}] 판정 {'일치' if ok else '불일치'} | 분포 행 R {len(rd)} / Py {len(pd_)}, 한쪽에만 있는 행 {only}, "
                  f"확률 최대차 {pdiff:.2e} | 통계량 행 R {len(rm)} / Py {len(pm)}, 최대차 "
                  + ", ".join(f"{k} {v:.1e}" for k, v in mdiffs.items())
                  + f" | 50% 경계 동률 {int(mm['tie'].sum())}건 중 왜도가 다른 경우 {n_tie_diff}건")
    print("전체 판정:", "모두 일치" if all_ok else "불일치 있음")


if __name__ == "__main__":
    main()
