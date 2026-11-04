"""지표 발표 단위 테이블: 컨센서스(ForexFactory), 실제치(ForexFactory·ALFRED), Kalshi 정산 결과 대조.

처리 순서
  [1] ForexFactory: 연구 대상 지표만 남기고 값 파싱(%, K, M), 날짜 연도 오류 수정
      - 수집 스크립트가 주 시작일의 연도를 붙여서, 12월 말에 시작한 주의 1월 발표는 연도가 1 작게 기록됨
      - 표시 시각은 한국 시간. 미국 오전 발표는 한국 시간으로도 같은 날이라 날짜는 그대로 미국 발표일로 씀
  [2] ALFRED: 최초 발표 시점 기준 값 계산
      - CPI·core CPI: 지수라서 최초 발표 시점에 알려진 전월 지수로 전월비(%)를 계산하고 소수 1자리로 반올림
      - 실업률·GDP 속보치: 최초 발표치 그대로
      - 고용(비농업)은 ALFRED를 받지 않아서 ForexFactory 실제치만 있음
  [3] Kalshi 이벤트와 발표일로 연결하고, 각 실제치가 Kalshi 정산 구간 (lower, upper] 안에 드는지 확인

출력: data/interim/releases.parquet (지표×발표일), data/interim/kalshi_release_match.parquet (Kalshi 이벤트별)
실행: python src/clean/02_releases.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraping"))
from common import PROJECT_ROOT, RAW_DIR, get_logger  # noqa: E402

INTERIM = PROJECT_ROOT / "data" / "interim"

# ForexFactory 이벤트 이름 → 연구용 시리즈 이름
FF_EVENTS = {
    "CPI m/m": "KXCPI",
    "Core CPI m/m": "KXCPICORE",
    "Unemployment Rate": "KXU3",
    "Non-Farm Employment Change": "KXPAYROLLS",
    "Advance GDP q/q": "KXGDP",   # Kalshi GDP 계약은 속보치(Advance Estimate) 기준
    "PPI m/m": "PPI",              # RQ1 통제변수용
}
# 시리즈 → (ALFRED ID, 계산 방식)
ALFRED = {
    "KXCPI": ("CPIAUCSL", "mom"),
    "KXCPICORE": ("CPILFESL", "mom"),
    "KXU3": ("UNRATE", "level"),
    "KXGDP": ("A191RL1Q225SBEA", "level"),
    "PPI": ("PPIFIS", "mom"),
}


def parse_value(s) -> float:
    """'0.3%', '275K', '-32K', '1.2M' → 숫자. 고용은 명 단위로 바꿈."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().replace(",", "")
    m = re.match(r"^([<>]?)(-?\d+(?:\.\d+)?)([%KMB]?)$", s)
    if not m:
        return np.nan
    v = float(m.group(2))
    return v * {"K": 1e3, "M": 1e6, "B": 1e9}.get(m.group(3), 1.0)


def load_forexfactory(logger) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW_DIR / "forexfactory" / "calendar_*.json"))):
        rows += json.loads(Path(f).read_text(encoding="utf-8"))
    ff = pd.DataFrame(rows)
    logger.info("  원본 %d건", len(ff))
    ff = ff[(ff["currency"] == "USD") & ff["event"].isin(FF_EVENTS)].copy()
    ff["series"] = ff["event"].map(FF_EVENTS)

    week_start = pd.to_datetime(ff["week"], format="%b%d.%Y")
    md = ff["date_text"].str.extract(r"^\w{3} (\w{3}) (\d{1,2})")
    d = pd.to_datetime(md[0] + " " + md[1] + " " + week_start.dt.year.astype(str), format="%b %d %Y")
    fix = d < week_start - pd.Timedelta(days=1)
    d = d.where(~fix, d + pd.DateOffset(years=1))
    logger.info("  연초 연도 오류 수정 %d건", int(fix.sum()))
    ff["release_date_et"] = d.dt.date

    for c in ["actual", "forecast", "previous"]:
        ff[c + "_num"] = ff[c].map(parse_value)
    bad = ff["forecast"].notna() & (ff["forecast"] != "") & ff["forecast_num"].isna()
    logger.info("  forecast 파싱 실패 %d건, 빈칸 %d건", int(bad.sum()), int((ff["forecast"].fillna("") == "").sum()))

    ff = ff.sort_values("scraped_at_utc").drop_duplicates(["series", "release_date_et"], keep="last")
    out = ff.rename(columns={"forecast_num": "F", "actual_num": "A_ff", "previous_num": "prev_ff"})
    logger.info("  지표별 발표 건수 %s", out["series"].value_counts().to_dict())
    return out[["series", "release_date_et", "time_text", "F", "A_ff", "prev_ff", "scraped_at_utc"]]


def alfred_first_release(series: str, sid: str, how: str) -> pd.DataFrame:
    v = pd.read_csv(RAW_DIR / "alfred" / f"{sid}_vintages.csv", parse_dates=["realtime_start", "date"])
    v = v.dropna(subset=["value"]).sort_values(["date", "realtime_start"])
    first = v.groupby("date", as_index=False).first()          # 기준월별 최초 발표
    if how == "level":
        first["A_alfred"] = first["value"]
    else:
        vals = []
        for _, r in first.iterrows():
            prev = v[(v["date"] == r["date"] - pd.DateOffset(months=1)) & (v["realtime_start"] <= r["realtime_start"])]
            vals.append(np.nan if prev.empty else
                        round((r["value"] / prev.iloc[-1]["value"] - 1) * 100, 1))
        first["A_alfred"] = vals
    return pd.DataFrame({"series": series, "ref_period": first["date"].dt.date,
                         "release_date_et": first["realtime_start"].dt.date, "A_alfred": first["A_alfred"]})


MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def kalshi_ref_period(row) -> pd.Timestamp:
    """Kalshi 이벤트 이름의 기준월. 월간 지표는 '-25SEP' → 2025-09-01.
    GDP는 '-25OCT30'처럼 발표 예정일이 붙어 있어서, 발표 월의 직전 분기 시작일을 기준분기로 씀."""
    t = row["event_ticker"]
    q = re.search(r"(\d{4}) Q(\d)$", t)              # 예외 형식: 'GDP-232022 Q4'
    if q:
        return pd.Timestamp(int(q.group(1)), (int(q.group(2)) - 1) * 3 + 1, 1)
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})?[A-Z]?$", t)  # 예외 형식: 끝에 문자 하나(-25DECT, -23DECB)
    if not m:
        return pd.NaT
    y, mon = 2000 + int(m.group(1)), MONTHS[m.group(2)]
    if row["series"] == "KXGDP":
        q_start = (mon - 1) // 3 * 3 + 1          # 발표 월이 속한 분기의 시작 월
        ref = pd.Timestamp(y, q_start, 1) - pd.DateOffset(months=3)
        return ref
    return pd.Timestamp(y, mon, 1)


def add_ref_period(rel: pd.DataFrame) -> pd.DataFrame:
    """발표 테이블에 기준월 부여. ALFRED가 있으면 그 값을 쓰고,
    고용(ALFRED 없음)은 같은 날 발표되는 실업률의 기준월을 씀 (둘 다 고용상황 보고서)."""
    rel = rel.copy()
    rel["ref_period"] = pd.to_datetime(rel["ref_period"])
    u3 = rel[rel["series"] == "KXU3"][["release_date_et", "ref_period"]].rename(columns={"ref_period": "u3_ref"})
    rel = rel.merge(u3, on="release_date_et", how="left")
    is_nfp = rel["series"] == "KXPAYROLLS"
    rel.loc[is_nfp, "ref_period"] = rel.loc[is_nfp, "u3_ref"]
    return rel.drop(columns="u3_ref")


def main() -> None:
    logger = get_logger("clean_02_releases")
    logger.info("[1/3] ForexFactory 컨센서스와 실제치")
    ff = load_forexfactory(logger)

    logger.info("[2/3] ALFRED 최초 발표치")
    al = pd.concat([alfred_first_release(s, sid, how) for s, (sid, how) in ALFRED.items()], ignore_index=True)
    rel = ff.merge(al, on=["series", "release_date_et"], how="left")
    both = rel.dropna(subset=["A_ff", "A_alfred"])
    agree = np.isclose(both["A_ff"], both["A_alfred"], atol=1e-9)
    logger.info("  ALFRED 연결 %d건 / ForexFactory %d건", rel["A_alfred"].notna().sum(), len(rel))
    for s, g in both.assign(agree=agree).groupby("series"):
        logger.info("  %s: ForexFactory 실제치와 ALFRED 일치 %d/%d", s, g["agree"].sum(), len(g))
    rel.to_parquet(INTERIM / "releases.parquet", index=False)
    logger.info("  저장 releases.parquet (%d행)", len(rel))

    logger.info("[3/3] Kalshi 이벤트와 연결, 정산 구간 대조")
    ev = pd.read_parquet(INTERIM / "kalshi_events.parquet")
    ev = ev[ev["settled"]].copy()
    # 기준월(기준분기)로 연결: 발표가 미뤄진 경우(예: 2025년 연방정부 셧다운)에도 같은 달 지표끼리 연결됨
    ev["ref_period"] = ev.apply(kalshi_ref_period, axis=1)
    # 같은 기준월에 이벤트가 둘 이상이면(예: PAYROLLS-23DEC와 PROLLS-23DECB) 기준선이 많은 쪽을 남김
    ev = ev.sort_values("n_thresholds", ascending=False)
    dropped = ev[ev.duplicated(["series", "ref_period"], keep="first")]
    if len(dropped):
        logger.info("  같은 기준월 중복 이벤트 제외: %s", dropped["event_ticker"].tolist())
    ev = ev.drop_duplicates(["series", "ref_period"], keep="first")
    rel = add_ref_period(rel)
    r2 = rel.rename(columns={"release_date_et": "release_date_actual"})
    m = ev.merge(r2, on=["series", "ref_period"], how="left")
    m["delay_days"] = (pd.to_datetime(m["release_date_actual"]) - pd.to_datetime(m["release_date_et"])).dt.days
    dup = m.dropna(subset=["release_date_actual"]).duplicated(["series", "release_date_actual"], keep=False)
    if dup.any():
        logger.warning("  같은 발표에 연결된 Kalshi 이벤트가 둘 이상: %d건", int(dup.sum()))
    delayed = m[m["delay_days"] != 0].dropna(subset=["delay_days"])
    logger.info("  Kalshi 마감일과 실제 발표일이 다른 이벤트 %d개: %s", len(delayed),
                delayed[["event_ticker", "delay_days"]].astype(str).agg(" ".join, axis=1).tolist())
    for col in ["A_ff", "A_alfred"]:
        m[f"{col}_in_kalshi"] = np.where(m[col].isna(), np.nan,
                                         (m[col] > m["outcome_lower"] + 1e-9) & (m[col] <= m["outcome_upper"] + 1e-9))
    for s, g in m.groupby("series"):
        logger.info("  %s: 정산 이벤트 %d, 컨센서스 연결 %d, 정산 구간 일치 ForexFactory %s / ALFRED %s",
                    s, len(g), g["F"].notna().sum(),
                    f"{int(np.nansum(g['A_ff_in_kalshi']))}/{g['A_ff_in_kalshi'].notna().sum()}",
                    f"{int(np.nansum(g['A_alfred_in_kalshi']))}/{g['A_alfred_in_kalshi'].notna().sum()}")
    unmatched = m[m["F"].isna()][["series", "event_ticker", "release_date_et"]]
    if len(unmatched):
        logger.warning("  컨센서스와 연결 안 된 Kalshi 이벤트 %d개:\n%s", len(unmatched), unmatched.to_string(index=False))
    m.to_parquet(INTERIM / "kalshi_release_match.parquet", index=False)
    logger.info("  저장 kalshi_release_match.parquet (%d행)", len(m))


if __name__ == "__main__":
    main()
