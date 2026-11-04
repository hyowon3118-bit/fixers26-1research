"""Diercks, Katz & Wright (2026)가 공개한 Kalshi 분포·통계량 파일 내려받기 (비교 기준용).

출처: 원본 저장소의 웹사이트 코드(docs/index.html)가 읽는 공개 S3 버킷
      https://kalshi-and-the-rise-of-macro-markets.s3.amazonaws.com
원본 저장소 자체에는 data/ 폴더가 .gitignore로 빠져 있어서 원자료가 없음.

동작
  [1] 버킷 목록을 받아 어떤 파일이 있는지 저장 (체결 단위 원자료 trade_level_data가 있는지 확인)
  [2] 이 연구와 겹치는 시리즈의 일별 분포·통계량 파일을 내려받음
      - headline CPI 전월비(KXCPI): *_headline_cpi_releases_mom.csv
      - 실업률(KXU3): *_unemployment_releases.csv
      (원본의 core CPI는 전년비 계약(KXCPICOREYOY)이라 이 연구의 KXCPICORE와 다름)
  [3] 버킷에 체결 단위 원자료가 있으면 같은 두 시리즈 것도 내려받음

저장: data/raw/dkw_reference/
실행: python scraping/06_dkw_reference.py
"""
from __future__ import annotations

import re

import pandas as pd

from common import RAW_DIR, get_logger, http_get

BUCKET = "https://kalshi-and-the-rise-of-macro-markets.s3.amazonaws.com"
TARGETS = [
    "daily_moments_data/daily_moments_headline_cpi_releases_mom.csv",
    "daily_distribution_data/daily_distributions_headline_cpi_releases_mom.csv",
    "daily_moments_data/daily_moments_unemployment_releases.csv",
    "daily_distribution_data/daily_distributions_unemployment_releases.csv",
]
TRADE_TARGETS = ["trade_level_data/trade_level_data_headline_cpi_releases_mom.csv",
                 "trade_level_data/trade_level_data_unemployment.csv"]


def list_bucket(logger) -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2"}
        if token:
            params["continuation-token"] = token
        resp = http_get(BUCKET + "/", logger, params=params)
        if resp is None:
            logger.warning("  버킷 목록을 볼 수 없어 (목록 권한이 막혀 있을 수 있음). 알려진 파일만 받을게")
            return []
        keys += re.findall(r"<Key>([^<]+)</Key>", resp.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", resp.text)
        if not m:
            return keys
        token = m.group(1)


def main() -> None:
    logger = get_logger("06_dkw_reference")
    out = RAW_DIR / "dkw_reference"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[1/3] 공개 버킷 목록 확인")
    keys = list_bucket(logger)
    if keys:
        pd.Series(keys, name="key").to_csv(out / "bucket_keys.csv", index=False)
        folders = pd.Series([k.split("/")[0] for k in keys]).value_counts().to_dict()
        logger.info("  파일 %d개, 폴더별 %s", len(keys), folders)
        logger.info("  체결 단위 원자료(trade_level_data) %s", "있음" if any(k.startswith("trade_level_data/") for k in keys) else "없음")

    logger.info("[2/3] 비교용 분포·통계량 내려받기")
    for key in TARGETS:
        resp = http_get(f"{BUCKET}/{key}", logger)
        if resp is None:
            logger.error("  실패: %s", key)
            continue
        (out / key.split("/")[-1]).write_bytes(resp.content)
        logger.info("  저장 %s (%.0f KB)", key.split("/")[-1], len(resp.content) / 1024)

    logger.info("[3/3] 체결 단위 원자료 내려받기 (있을 때만)")
    for key in TRADE_TARGETS:
        if keys and key not in keys:
            logger.info("  버킷에 없음: %s", key)
            continue
        resp = http_get(f"{BUCKET}/{key}", logger, max_retries=2)
        if resp is None:
            logger.info("  받을 수 없음: %s", key)
            continue
        (out / key.split("/")[-1]).write_bytes(resp.content)
        logger.info("  저장 %s (%.0f KB)", key.split("/")[-1], len(resp.content) / 1024)
    logger.info("종료")


if __name__ == "__main__":
    main()
