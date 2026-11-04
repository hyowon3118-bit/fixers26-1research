"""ForexFactory 경제 캘린더에서 미국 지표의 actual / forecast / previous 수집.

주의
  - 사이트 이용약관을 먼저 확인하고, 요청 간격(settings.yaml)을 줄이지 마.
  - 이 사이트는 봇 차단(HTTP 403)이 걸릴 수 있어. 차단되면 스크립트가 멈추고 안내를 남겨.
    그때는 Playwright 기반 수집(GitHub: tchala120/forex-factory-scarping 참고)이나
    유료 Apify actor 사용을 검토해.
  - 페이지 구조가 바뀌었을 수 있어서 두 가지 방식으로 파싱해:
      (1) 페이지에 들어있는 JSON 데이터, (2) HTML 표 (GitHub gist pohzipohzi 방식).
  - 표시 시각은 사이트의 시간대 설정을 따라. 시간대는 로그에 경고로 남기고 clean 단계에서 검증해.
  - 과거 forecast가 발표 당시 값인지는 이 스크립트로 보장할 수 없어. 수집 시각을 scrape_log.csv에 남겨.

저장 위치:
  data/raw/forexfactory/calendar_{YYYYMM}.json   (해당 월에 시작하는 주 단위로 누적)
  data/raw/forexfactory/scrape_log.csv
  data/raw/forexfactory/html/{week}.html         (파싱 실패 시 확인용 원본)

실행: python scraping/02_forexfactory.py
      python scraping/02_forexfactory.py --start 2025-10-01 --end 2025-10-31   (테스트)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, timedelta

from bs4 import BeautifulSoup

from common import RAW_DIR, get_logger, http_get, load_settings, utc_now_iso

BASE = "https://www.forexfactory.com/calendar"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
OUT_DIR = RAW_DIR / "forexfactory"


def week_param(d: date) -> str:
    """ForexFactory 주 단위 URL 파라미터, 예: oct6.2025"""
    return f"{d.strftime('%b').lower()}{d.day}.{d.year}"


def parse_embedded_json(html: str) -> list[dict]:
    """페이지 스크립트에 들어있는 캘린더 JSON에서 이벤트 추출 (구조가 있을 때만)."""
    events = []
    for m in re.finditer(r"calendarComponentStates\[\d+\]\s*=\s*(\{.*?\});\s*\n", html, re.S):
        raw = m.group(1)
        try:
            # JS 객체 표기라 순수 JSON이 아닐 수 있음 → 실패하면 HTML 파싱으로 넘어감
            state = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for day in state.get("days", []) or []:
            for ev in day.get("events", []) or []:
                events.append({
                    "event_id": ev.get("id"),
                    "dateline_unix": ev.get("dateline"),
                    "date_text": day.get("date"),
                    "time_text": ev.get("timeLabel"),
                    "currency": ev.get("currency"),
                    "impact": ev.get("impactName") or ev.get("impactTitle"),
                    "event": ev.get("name"),
                    "actual": ev.get("actual"),
                    "forecast": ev.get("forecast"),
                    "previous": ev.get("previous"),
                    "revision": ev.get("revision"),
                    "parse_method": "embedded_json",
                })
    return events


def parse_html_table(html: str, year: int) -> list[dict]:
    """HTML 표에서 이벤트 추출. 날짜·시간 칸이 비면 위 행의 값을 이어 씀."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile("calendar__table"))
    if table is None:
        return []
    events, cur_date, cur_time = [], "", ""

    def cell(tr, name):
        td = tr.find("td", class_=re.compile(rf"calendar__{name}\b"))
        return td.get_text(" ", strip=True) if td else ""

    for tr in table.find_all("tr", class_=re.compile("calendar__row")):
        d, t = cell(tr, "date"), cell(tr, "time")
        cur_date = d or cur_date
        cur_time = t or cur_time
        name = cell(tr, "event")
        if not name:
            continue
        impact_td = tr.find("td", class_=re.compile(r"calendar__impact\b"))
        impact_span = impact_td.find("span") if impact_td else None
        events.append({
            "event_id": tr.get("data-event-id"),
            "dateline_unix": None,
            "date_text": f"{cur_date} {year}",
            "time_text": cur_time,
            "currency": cell(tr, "currency"),
            "impact": impact_span.get("title") if impact_span else None,
            "event": name,
            "actual": cell(tr, "actual"),
            "forecast": cell(tr, "forecast"),
            "previous": cell(tr, "previous"),
            "revision": None,
            "parse_method": "html_table",
        })
    return events


def main() -> None:
    cfg = load_settings()
    fcfg = cfg["forexfactory"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg["sample"]["start_date"])
    ap.add_argument("--end", default=cfg["sample"]["end_date"])
    args = ap.parse_args()

    logger = get_logger("02_forexfactory")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "html").mkdir(exist_ok=True)
    keywords = [k.lower() for k in fcfg["event_keywords"]]

    start = date.fromisoformat(args.start)
    start -= timedelta(days=start.weekday() + 1)  # ForexFactory 주는 일요일 시작
    end = date.fromisoformat(args.end)
    weeks = []
    d = start
    while d <= end:
        weeks.append(d)
        d += timedelta(days=7)
    logger.info("ForexFactory 수집 시작 | %s ~ %s | %d주 | 통화 %s | 키워드 %s",
                args.start, args.end, len(weeks), fcfg["currency"], fcfg["event_keywords"])
    logger.warning("시간 표시는 사이트 시간대 설정을 따름. clean 단계에서 BLS/BEA 발표 시각과 대조해")

    log_path = OUT_DIR / "scrape_log.csv"
    new_log = not log_path.exists()
    by_month: dict[str, list] = {}
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        if new_log:
            writer.writerow(["week", "url", "status", "parse_method", "n_all", "n_kept", "scraped_at_utc"])
        for i, wk in enumerate(weeks, 1):
            wp = week_param(wk)
            url = f"{BASE}?week={wp}"
            resp = http_get(url, logger, headers=HEADERS, sleep_sec=fcfg["request_sleep_sec"], max_retries=3)
            scraped_at = utc_now_iso()
            if resp is None:
                writer.writerow([wp, url, "failed", "", 0, 0, scraped_at])
                lf.flush()
                logger.error("[%d/%d] %s 실패. 403이면 봇 차단이야. 수집을 멈출게. "
                             "docstring의 대안(Playwright, Apify)을 확인해", i, len(weeks), wp)
                break
            html = resp.text
            events = parse_embedded_json(html)
            method = "embedded_json"
            if not events:
                events = parse_html_table(html, wk.year)
                method = "html_table"
            if not events:
                (OUT_DIR / "html" / f"{wp}.html").write_text(html, encoding="utf-8")
                logger.warning("[%d/%d] %s 파싱 결과 0건. 원본 HTML 저장 → html/%s.html", i, len(weeks), wp, wp)
            kept = [e for e in events
                    if (e.get("currency") or "").upper() == fcfg["currency"]
                    and any(k in (e.get("event") or "").lower() for k in keywords)]
            for e in kept:
                e.update(week=wp, source_url=url, scraped_at_utc=scraped_at)
            by_month.setdefault(wk.strftime("%Y%m"), []).extend(kept)
            writer.writerow([wp, url, "ok", method, len(events), len(kept), scraped_at])
            lf.flush()
            logger.info("[%d/%d] %s | 전체 %d건, 대상 %d건 (%s)", i, len(weeks), wp, len(events), len(kept), method)

    for ym, rows in by_month.items():
        out = OUT_DIR / f"calendar_{ym}.json"
        existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(existing + rows, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("저장 %s (+%d건)", out.name, len(rows))
    logger.info("ForexFactory 수집 종료")


if __name__ == "__main__":
    main()
