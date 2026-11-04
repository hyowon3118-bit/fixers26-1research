"""한국은행 ECOS Open API에서 국고채 2년 일별 금리 수집 (requests로 직접 호출).

필요: ECOS API 키 (무료, https://ecos.bok.or.kr/api/) → .env에 ECOS_API_KEY=...
동작: (1) 통계표의 항목 목록을 받아 이름에 '국고채(2년)'이 들어간 항목 코드를 자동으로 찾고
      (2) 그 코드로 일별 데이터를 받아 저장해. 못 찾으면 항목 목록을 저장하고 멈춰.
참고: 바탕화면 research/ecos_rates_daily.csv 에 이미 받은 금리가 있으면 그걸 써도 돼.

저장: data/raw/rates/ecos_ktb2y.csv, data/raw/rates/ecos_{stat_code}_items.csv
실행: python scraping/05_ecos_ktb.py
"""
from __future__ import annotations

import os

import pandas as pd

from common import RAW_DIR, get_logger, http_get, load_env_file, load_settings, utc_now_iso


def main() -> None:
    load_env_file()
    logger = get_logger("05_ecos_ktb")
    key = os.environ.get("ECOS_API_KEY")
    if not key:
        logger.error("ECOS_API_KEY가 없어. .env 파일이나 환경변수로 설정해")
        return
    cfg = load_settings()
    ecfg = cfg["ecos"]
    base, stat = ecfg["base_url"], ecfg["stat_code"]
    out_dir = RAW_DIR / "rates"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/2] 통계표 %s 항목 목록 조회", stat)
    resp = http_get(f"{base}/StatisticItemList/{key}/json/kr/1/1000/{stat}", logger)
    if resp is None:
        return
    rows = resp.json().get("StatisticItemList", {}).get("row", [])
    items = pd.DataFrame(rows)
    items.to_csv(out_dir / f"ecos_{stat}_items.csv", index=False, encoding="utf-8-sig")
    logger.info("  항목 %d개 저장 → ecos_%s_items.csv", len(items), stat)
    if items.empty:
        logger.error("  항목이 없어. 응답: %s", resp.text[:300])
        return
    match = items[items["ITEM_NAME"].str.contains(ecfg["item_name_contains"], regex=False, na=False)]
    if match.empty:
        logger.error("  '%s' 항목을 못 찾았어. 항목 목록 파일을 보고 settings.yaml을 수정해",
                     ecfg["item_name_contains"])
        return
    item_code = match.iloc[0]["ITEM_CODE"]
    logger.info("  사용 항목: %s (%s)", match.iloc[0]["ITEM_NAME"], item_code)

    logger.info("[2/2] 일별 데이터 조회")
    start = (pd.Timestamp(cfg["sample"]["start_date"]) - pd.DateOffset(days=30)).strftime("%Y%m%d")
    end = pd.Timestamp(cfg["sample"]["end_date"]).strftime("%Y%m%d")
    data, page, size = [], 0, 1000
    while True:
        s, e = page * size + 1, (page + 1) * size
        url = f"{base}/StatisticSearch/{key}/json/kr/{s}/{e}/{stat}/D/{start}/{end}/{item_code}"
        resp = http_get(url, logger)
        if resp is None:
            break
        block = resp.json().get("StatisticSearch", {})
        rows = block.get("row", [])
        data.extend(rows)
        total = int(block.get("list_total_count", 0) or 0)
        logger.info("  %d~%d행 수신 (누적 %d / 전체 %d)", s, e, len(data), total)
        page += 1
        if not rows or len(data) >= total:
            break
    df = pd.DataFrame(data)
    if df.empty:
        logger.error("  데이터 0건")
        return
    df = df.rename(columns={"TIME": "date", "DATA_VALUE": "value"})[["date", "value", "ITEM_NAME1", "UNIT_NAME"]]
    df["collected_at_utc"] = utc_now_iso()
    df.to_csv(out_dir / "ecos_ktb2y.csv", index=False, encoding="utf-8-sig")
    logger.info("  저장 %d행 → ecos_ktb2y.csv", len(df))


if __name__ == "__main__":
    main()
