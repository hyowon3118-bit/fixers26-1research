"""수집 스크립트 공통 유틸: 경로, 설정, 로그, HTTP 요청."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


def load_settings() -> dict:
    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_file() -> None:
    """프로젝트 루트의 .env 파일(KEY=VALUE 형식)을 환경변수로 읽어. 없으면 건너뜀."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_logger(name: str) -> logging.Logger:
    """콘솔과 outputs/logs/{name}_{시각}.log 에 동시에 기록하는 로거."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"{name}_{stamp}.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.info("로그 파일: %s", LOG_DIR / f"{name}_{stamp}.log")
    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(
    url: str,
    logger: logging.Logger,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 5,
    sleep_sec: float = 0.0,
    timeout: int = 30,
) -> requests.Response | None:
    """재시도(429·5xx·네트워크 오류)와 로그를 포함한 GET. 실패하면 None."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning("네트워크 오류 (%d/%d): %s → %ds 후 재시도", attempt, max_retries, e, wait)
            time.sleep(wait)
            continue
        if sleep_sec:
            time.sleep(sleep_sec)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            logger.warning("HTTP %d (%d/%d): %s → %ds 후 재시도",
                           resp.status_code, attempt, max_retries, url, wait)
            time.sleep(wait)
            continue
        logger.error("HTTP %d: %s | 응답 일부: %s", resp.status_code, url, resp.text[:300])
        return None
    logger.error("최대 재시도 초과: %s", url)
    return None


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
