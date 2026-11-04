"""Diercks, Katz & Wright (2026)의 Kalshi 분포 구성 방법을 파이썬으로 옮긴 함수 모음.

원본: github.com/hyowon3118-bit/Prediction_Markets_Public (포크)
      code/convert_trades_to_pdfs/convert_trade_level_data_cdfs.R
각 함수 이름 옆에 대응하는 원본 R 함수를 적었어. 가격 단위는 원본과 같이 센트(1~99).

원본과 의도적으로 다른 점 (docs/methods_kalshi_distribution.md에 정리)
  (a) 기준선 값은 ticker 문자열이 아니라 API의 floor_strike를 써.
      원본의 정규식은 음수 기준선(예: T-0.1)을 읽지 못해서 그 계약이 빠져.
  (b) swap 단계에서 보호 행(protected)을 정할 때, 원본은 49 초과 가격이 없으면
      그룹이 아닌 전체 데이터의 which.max를 써(원본 코드의 df 참조). 여기서는 그룹 안에서 찾아.
그 밖의 계산 순서와 조건은 원본과 같아. 동일성은 tests/verify_against_r.py로 확인해.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIDDLE_TARGET = 49  # 원본: Kalshi 가격이 1~99라서 중앙값을 50이 아니라 49로 둠


# ------------------------------------------------------------ fill_dataless_days
def fill_dataless_days(df: pd.DataFrame, days_before_horizon: int) -> pd.DataFrame:
    """[R] fill_dataless_days: 거래 없는 날을 직전 체결가로 채우고 기간 밖 행을 제거.

    df: date, contract_preamble, strike, yes_price, daily_volume
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    combos = df[["contract_preamble", "strike"]].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    full = combos.merge(dates, how="cross")
    df = full.merge(df, on=["contract_preamble", "strike", "date"], how="left")
    df = df.sort_values(["contract_preamble", "strike", "date"])

    # 만기일 = 해당 이벤트에서 가격이 관측된 마지막 날
    expiry = (df.dropna(subset=["yes_price"]).groupby("contract_preamble")["date"].max()
              .rename("expiry_date"))
    df = df.merge(expiry, on="contract_preamble", how="left")
    df = df[df["date"] >= df["expiry_date"] - pd.Timedelta(days=days_before_horizon + 10)]

    df["yes_price"] = df.groupby(["contract_preamble", "strike"])["yes_price"].ffill()
    df["daily_volume"] = df["daily_volume"].fillna(0)
    df = df.dropna(subset=["yes_price", "expiry_date"])
    df = df[df["date"] <= df["expiry_date"]]
    return df.reset_index(drop=True)


# ------------------------------------------------------------ clean_data
def _middle_out(yes: np.ndarray, target: float = MIDDLE_TARGET) -> np.ndarray:
    """[R] adjust_middle_out: 49에 가장 가까운 기준선에서 양쪽으로 단조성 강제."""
    n = len(yes)
    k = int(np.argmin(np.abs(yes - target)))  # R which.min과 같이 동률이면 첫 번째
    adj = yes.astype(float).copy()
    if k > 0:  # 왼쪽(낮은 기준선)은 가격이 더 높아야 함
        seq_left = np.concatenate([[yes[k]], yes[:k][::-1]])
        adj[:k] = np.maximum.accumulate(seq_left)[1:][::-1]
    if k < n - 1:  # 오른쪽(높은 기준선)은 가격이 더 낮아야 함
        seq_right = np.concatenate([[yes[k]], yes[k + 1:]])
        adj[k + 1:] = np.minimum.accumulate(seq_right)[1:]
    return adj


def clean_data(df: pd.DataFrame, method: str = "middle-out") -> pd.DataFrame:
    """[R] clean_data: 무차익(단조성) 조건 부과. middle-out(기본), right-to-left, left-to-right."""
    df = df.sort_values(["contract_preamble", "date", "strike"]).copy()
    g = df.groupby(["contract_preamble", "date"], sort=False)["yes_price"]
    if method == "middle-out":
        df["adjusted_yes_price"] = g.transform(lambda s: _middle_out(s.to_numpy()))
    elif method == "right-to-left":
        df["adjusted_yes_price"] = g.transform(lambda s: np.maximum.accumulate(s.to_numpy()[::-1])[::-1])
    elif method == "left-to-right":
        df["adjusted_yes_price"] = g.transform(lambda s: -np.maximum.accumulate(-s.to_numpy()))
    else:
        raise ValueError(f"알 수 없는 단조성 방법: {method}")
    return df


# ------------------------------------------------------------ convert_to_probabilities
def _swap_probabilities(prob: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """[R] swap_probabilities: 빈 구간(확률 0)이 생기면 바깥 구간 확률을 중앙 쪽으로 옮김.

    원본은 1부터 세는 인덱스를 쓰므로 i를 그대로 옮기고 배열 접근에서 1을 뺌.
    """
    prob = prob.astype(float).copy()
    # 원본: adjusted_yes_price의 0을 NA로 바꾼 뒤 아래 행 값으로 위를 채움(fill up)
    adj = pd.Series(np.where(adj == 0, np.nan, adj)).bfill().to_numpy()
    n = len(prob)
    nrows = n - 1
    above = np.where(adj > MIDDLE_TARGET)[0]
    protected = (above.max() + 1) if len(above) else (int(np.nanargmax(adj)) + 1)  # 1부터 세는 번호
    swapped = np.zeros(n, dtype=bool)
    j = 0
    while swapped.any() or j == 0:
        swapped[:] = False
        if nrows > 2:
            for i in range(1, nrows + 1):  # R: for (i in 1:nrows)
                a, b = i - 1, i            # x[i], x[i+1]을 0부터 세는 위치로
                if adj[a] > MIDDLE_TARGET and prob[a] != 0 and prob[b] == 0 and i != protected:
                    prob[b] = prob[a]
                    prob[a] = 0
                    swapped[b] = True
                if (adj[a] < MIDDLE_TARGET and prob[a] == 0 and prob[b] != 0
                        and i != protected and (i + 1) != protected):
                    prob[a] = prob[b]
                    prob[b] = 0
                    swapped[a] = True
        j += 1
    return prob


def convert_to_probabilities(df: pd.DataFrame, strike_int: float | None,
                             days_before_horizon: int, swap: bool = True) -> pd.DataFrame:
    """[R] convert_to_probabilities: 조정 가격 → 구간 확률(합 100).

    각 행의 strike s_j에 붙는 확률은 P(s_j < A <= s_{j+1})이고,
    맨 아래에 추가한 행(min strike - strike_int)은 P(A <= 최저 기준선)이야.
    strike_int=None이면 기준선 간격이 불규칙한 시리즈로 보고, 빈 기준선 채우기를 하지 않아 (원본에 없는 처리).
    """
    out = []
    for (pre, d), g in df.groupby(["contract_preamble", "date"], sort=False):
        g = g.sort_values("strike")
        step = strike_int if strike_int is not None else float(np.diff(g["strike"]).min()) if len(g) > 1 else 1.0
        low = pd.DataFrame({"contract_preamble": [pre], "date": [d],
                            "expiry_date": [g["expiry_date"].iloc[0]],
                            "strike": [g["strike"].min() - step]})
        g = pd.concat([g, low], ignore_index=True).sort_values("strike").reset_index(drop=True)
        adj = g["adjusted_yes_price"].to_numpy(dtype=float)
        nxt = np.append(adj[1:], np.nan)
        prob = np.where(np.arange(len(g)) == 0, 99 - nxt,
                        np.where(np.arange(len(g)) < len(g) - 1, adj - nxt, adj - 1))
        g["probability"] = prob

        if strike_int is not None:  # 빠진 기준선을 확률 0으로 채움
            g["k"] = np.round((g["strike"] - g["strike"].min()) / strike_int).astype(int)
            g = g.drop_duplicates("k", keep="first")
            grid = pd.DataFrame({"k": np.arange(g["k"].min(), g["k"].max() + 1)})
            g = grid.merge(g, on="k", how="left")
            for c in ["yes_price", "adjusted_yes_price", "probability", "daily_volume"]:
                g[c] = g[c].fillna(0)
            g["contract_preamble"], g["date"] = pre, d
            g["expiry_date"] = g["expiry_date"].ffill().bfill()
            base = g["strike"].min()
            g["strike"] = base + g["k"] * strike_int
            g = g.drop(columns="k")

        if swap:
            g["probability"] = _swap_probabilities(g["probability"].to_numpy(),
                                                   g["adjusted_yes_price"].to_numpy())
        g["probability"] = g["probability"] * 100 / g["probability"].sum()
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res = res[res["date"] >= res["expiry_date"] - pd.Timedelta(days=days_before_horizon)]
    return res.reset_index(drop=True)


# ------------------------------------------------------------ get_moments
def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """matrixStats::weightedMedian(interpolate=FALSE, ties=NULL)과 같은 값을 내도록 구현.

    R에서 확인한 동작: 가중치 0인 값은 무시하고, 누적 가중치가 정확히 절반에서 끝나면
    그 값과 다음 값의 단순 평균을 돌려줌 (예: 0과 0.1에 25씩, 0.2와 0.3에 25씩 → 0.15).
    """
    keep = w > 0
    x, w = x[keep], w[keep]
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    # R과 같이 허용오차 없이 정확히 비교 (부동소수점 결과까지 원본과 맞추기 위해)
    cw = np.cumsum(w)
    half = w.sum() / 2
    k = int(np.argmax(cw >= half))
    if cw[k] == half and k + 1 < len(x):
        return float((x[k] + x[k + 1]) / 2)
    return float(x[k])


def _gm_skew(x: np.ndarray, w: np.ndarray) -> float:
    """[R] weightedGMSkew: (평균 - 가중 중앙값) / 가중 평균절대편차.

    원본은 여기서 중앙값을 따로 계산해: 가중치 0도 포함하고, 정규화 누적 가중치가
    처음으로 0.5 이상이 되는 값을 씀 (위 _weighted_median과 규칙이 달라).
    """
    w = w / w.sum()
    mu = (w * x).sum()
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    m_w = x[o][int(np.argmax(cw >= 0.5))]
    mad = (w * np.abs(x - m_w)).sum()
    return float((mu - m_w) / mad) if mad > 0 else float("nan")  # 분포가 한 점에 몰리면 정의되지 않음


def get_moments(df: pd.DataFrame, moment_adjustment: float) -> pd.DataFrame:
    """[R] get_moments: 평균·중앙값·최빈값·분산·왜도. 첨도(DescTools::Kurt)는 연구에 쓰지 않아 제외."""
    rows = []
    for (d, pre, exp), g in df.groupby(["date", "contract_preamble", "expiry_date"], sort=True):
        x = g["strike"].to_numpy(float)
        w = g["probability"].to_numpy(float)
        mu = (w * x).sum() / w.sum()
        med = _weighted_median(x, w)
        # collapse::fmode(ties='first'): 가중치 합이 가장 큰 값, 동률이면 먼저 나온 값
        mode_tab = pd.Series(w).groupby(x, sort=False).sum()
        mode = float(mode_tab.idxmax())
        rows.append({
            "date": d, "contract_preamble": pre, "expiry_date": exp,
            "daily_volume": g["daily_volume"].sum(),
            "mean": mu + moment_adjustment,
            "median": med + moment_adjustment,
            "mode": mode + moment_adjustment,
            "skewness": _gm_skew(x, w),
            "variance": (w * (x - mu) ** 2).sum() / w.sum(),
        })
    return pd.DataFrame(rows)
