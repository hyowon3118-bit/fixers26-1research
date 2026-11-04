"""Kalshi 매크로 계약의 메타데이터와 일별 캔들 수집.

사용하는 공개 엔드포인트 (Kalshi 공식 문서 기준):
  - GET /historical/cutoff                       : 실시간/과거 데이터 경계 시각
  - GET /series?category=Economics               : 시리즈 목록 (이름 확인용)
  - GET /markets?series_ticker=...               : cutoff 이후 계약
  - GET /historical/markets?series_ticker=...    : cutoff 이전에 정산된 계약
  - GET /series/{s}/markets/{t}/candlesticks     : cutoff 이후 계약의 캔들
  - GET /historical/markets/{t}/candlesticks     : cutoff 이전 계약의 캔들

저장 위치:
  data/raw/kalshi/series_list.csv
  data/raw/kalshi/markets/{series}_markets.parquet   (원본 JSON 열 포함)
  data/raw/kalshi/candles/{series}/{market_ticker}.parquet

실행: python scraping/01_kalshi.py            (전체)
      python scraping/01_kalshi.py --series KXCPI --limit 5   (테스트)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from common import RAW_DIR, get_logger, http_get, load_settings, utc_now_iso

HEADERS = {"Accept": "application/json"}


def to_unix(iso_str: str | None) -> int | None:
    if not iso_str:
        return None
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())


def paginate(url: str, params: dict, key: str, logger, sleep: float) -> list[dict]:
    """cursor 기반 페이지네이션으로 목록 전체를 가져옴."""
    items, cursor, page = [], None, 0
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        resp = http_get(url, logger, params=p, headers=HEADERS, sleep_sec=sleep)
        if resp is None:
            logger.error("목록 수집 중단: %s", url)
            break
        data = resp.json()
        batch = data.get(key, []) or []
        items.extend(batch)
        page += 1
        logger.info("  페이지 %d: %d건 (누적 %d)", page, len(batch), len(items))
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    return items


def step_cutoff(base: str, logger, sleep: float) -> dict:
    logger.info("[1/4] 과거 데이터 경계(cutoff) 조회")
    resp = http_get(f"{base}/historical/cutoff", logger, headers=HEADERS, sleep_sec=sleep)
    cutoff = resp.json() if resp is not None else {}
    logger.info("  cutoff 응답: %s", cutoff)
    return cutoff


def step_series_list(base: str, logger, sleep: float) -> None:
    logger.info("[2/4] Economics 카테고리 시리즈 목록 저장 (시리즈 이름 확인용)")
    resp = http_get(f"{base}/series", logger, params={"category": "Economics"},
                    headers=HEADERS, sleep_sec=sleep)
    if resp is None:
        logger.warning("  시리즈 목록을 못 가져옴. settings.yaml의 시리즈 이름을 직접 확인해")
        return
    series = resp.json().get("series", []) or []
    df = pd.DataFrame([{"ticker": s.get("ticker"), "title": s.get("title"),
                        "frequency": s.get("frequency"), "category": s.get("category")}
                       for s in series])
    out = RAW_DIR / "kalshi" / "series_list.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("  %d개 시리즈 저장 → %s", len(df), out)


def step_markets(base: str, series: str, start_ts: int, logger, sleep: float) -> pd.DataFrame:
    logger.info("[3/4] %s 계약 메타데이터 수집", series)
    logger.info("  (a) cutoff 이후 계약")
    live = paginate(f"{base}/markets", {"series_ticker": series}, "markets", logger, sleep)
    logger.info("  (b) cutoff 이전에 정산된 과거 계약")
    hist = paginate(f"{base}/historical/markets", {"series_ticker": series}, "markets", logger, sleep)

    rows = []
    for src, items in (("live", live), ("historical", hist)):
        for m in items:
            rows.append({
                "series": series,
                "source": src,
                "ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"),
                "title": m.get("title"),
                "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
                "strike_type": m.get("strike_type"),
                "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"),
                "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "status": m.get("status"),
                "result": m.get("result"),
                "raw_json": json.dumps(m, ensure_ascii=False),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("  %s: 계약이 0건. 시리즈 이름을 확인해", series)
        return df
    before = len(df)
    df = df.drop_duplicates("ticker", keep="last")
    after_dedup = len(df)
    df["close_ts"] = df["close_time"].map(to_unix)
    df = df[df["close_ts"].fillna(0) >= start_ts]
    logger.info("  총 %d건 → 중복 제거 후 %d건 → 표본기간 내 %d건 (이벤트 %d개)",
                before, after_dedup, len(df), df["event_ticker"].nunique())
    out = RAW_DIR / "kalshi" / "markets" / f"{series}_markets.parquet"
    df.assign(collected_at_utc=utc_now_iso()).to_parquet(out, index=False)
    logger.info("  저장 → %s", out)
    return df


def step_candles(base: str, series: str, markets: pd.DataFrame, interval: int,
                 logger, sleep: float, limit: int | None) -> None:
    logger.info("[4/4] %s 일별 캔들 수집", series)
    out_dir = RAW_DIR / "kalshi" / "candles" / series
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = markets if limit is None else markets.head(limit)
    n = len(targets)
    ok = skipped = failed = empty = 0
    for i, m in enumerate(targets.itertuples(index=False), 1):
        out = out_dir / f"{m.ticker}.parquet"
        if out.exists():
            skipped += 1
            continue
        start = to_unix(m.open_time) or (to_unix(m.close_time) - 90 * 86400)
        end = to_unix(m.close_time)
        if m.source == "historical":
            url = f"{base}/historical/markets/{m.ticker}/candlesticks"
        else:
            url = f"{base}/series/{series}/markets/{m.ticker}/candlesticks"
        resp = http_get(url, logger, headers=HEADERS, sleep_sec=sleep,
                        params={"start_ts": start, "end_ts": end, "period_interval": interval})
        if resp is None:
            failed += 1
            continue
        candles = resp.json().get("candlesticks", []) or []
        if not candles:
            empty += 1
        # 필드 구조가 엔드포인트마다 다를 수 있어서 그대로 펼쳐 저장하고, 단위 변환은 clean 단계에서 해
        df = pd.json_normalize(candles)
        df.insert(0, "ticker", m.ticker)
        df.insert(1, "event_ticker", m.event_ticker)
        df["source"] = m.source
        df["collected_at_utc"] = utc_now_iso()
        df.to_parquet(out, index=False)
        ok += 1
        if i % 25 == 0 or i == n:
            logger.info("  진행 %d/%d | 저장 %d, 기존파일 건너뜀 %d, 빈 응답 %d, 실패 %d",
                        i, n, ok, skipped, empty, failed)
    logger.info("  %s 완료 | 저장 %d, 건너뜀 %d, 빈 응답 %d, 실패 %d", series, ok, skipped, empty, failed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", help="특정 시리즈만 수집 (기본: settings.yaml 전체)")
    ap.add_argument("--limit", type=int, default=None, help="시리즈당 캔들 수집 계약 수 제한 (테스트용)")
    args = ap.parse_args()

    cfg = load_settings()
    kcfg = cfg["kalshi"]
    base, sleep = kcfg["base_url"], kcfg["request_sleep_sec"]
    start_ts = int(datetime.fromisoformat(cfg["sample"]["start_date"]).replace(tzinfo=timezone.utc).timestamp())
    logger = get_logger("01_kalshi")
    series_list = args.series or kcfg["series"]
    logger.info("Kalshi 수집 시작 | 시리즈 %s | 표본 시작 %s", series_list, cfg["sample"]["start_date"])

    step_cutoff(base, logger, sleep)
    step_series_list(base, logger, sleep)
    for s in series_list:
        markets = step_markets(base, s, start_ts, logger, sleep)
        if not markets.empty:
            step_candles(base, s, markets, kcfg["period_interval"], logger, sleep, args.limit)
    logger.info("Kalshi 수집 종료")


if __name__ == "__main__":
    main()
