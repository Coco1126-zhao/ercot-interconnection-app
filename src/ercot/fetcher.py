"""
ercot/fetcher.py
================
Download the most recent monthly ERCOT GIS Report (reportTypeId=15933)
from the public MIS endpoint. No authentication required.

The endpoint returns an HTML page listing every monthly archive with
direct .xlsx links. We pick the newest file and write it to:
    data/ercot/latest.xlsx
    data/ercot/snapshots/YYYY-MM.xlsx

Run manually:
    python -m src.ercot.fetcher

Or via the GitHub Actions workflow `.github/workflows/monthly_refresh.yml`.
"""
from __future__ import annotations
import re
import sys
import shutil
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

LIST_URL = (
    "https://mis.ercot.com/misapp/GetReports.do"
    "?reportTypeId=15933&reportTitle=GIS%20Report&showHTMLView=&mimicKey="
)
BASE_URL = "https://mis.ercot.com"

DATA_DIR      = Path("data/ercot")
SNAPSHOT_DIR  = DATA_DIR / "snapshots"
LATEST_PATH   = DATA_DIR / "latest.xlsx"

# Filename pattern: RPT.00015933.0000000000000000.YYYYMMDD.HHMMSSXXX.GIS_Report_MonthYYYY.xlsx
DATE_RE = re.compile(r"\.(\d{8})\.\d+\.GIS_Report", re.IGNORECASE)


def _list_files() -> list[tuple[datetime, str]]:
    """Return [(timestamp, absolute_url), ...] sorted newest first."""
    r = requests.get(LIST_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".xlsx"):
            continue
        m = DATE_RE.search(href)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d")
        url = href if href.startswith("http") else BASE_URL + href
        out.append((ts, url))

    out.sort(key=lambda x: x[0], reverse=True)
    return out


def fetch_latest() -> Path:
    """Download newest GIS report. Returns path to the saved snapshot."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    files = _list_files()
    if not files:
        raise RuntimeError(
            "No GIS Report .xlsx links found at MIS endpoint. "
            "ERCOT may have changed the page layout — inspect "
            f"{LIST_URL} manually."
        )

    ts, url = files[0]
    # Snapshot name uses the report's month (file released early in month
    # for previous month's data — we tag by report-stamp month).
    snap_name = f"{ts.year:04d}-{ts.month:02d}.xlsx"
    snap_path = SNAPSHOT_DIR / snap_name

    if snap_path.exists():
        print(f"[skip] snapshot already present: {snap_path}")
    else:
        print(f"[get] {url} -> {snap_path}")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(snap_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 14):
                    f.write(chunk)

    # Update `latest.xlsx` pointer (plain copy — works on every OS / CI)
    shutil.copy2(snap_path, LATEST_PATH)
    print(f"[ok]  latest pointer updated: {LATEST_PATH}")
    return snap_path


if __name__ == "__main__":
    try:
        fetch_latest()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
