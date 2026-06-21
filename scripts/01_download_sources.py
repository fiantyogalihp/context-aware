
#!/usr/bin/env python3
"""Download official source documents listed in dataset/source_catalog.csv.
Uses only Python standard library. Some government sites may throttle requests;
if a direct PDF download fails, open the landing page manually from the catalog.
"""
from __future__ import annotations
import csv, time, urllib.request, urllib.parse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "dataset" / "source_catalog.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 official-health-rag-research/1.0"}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def find_first_pdf_or_download_link(html: str, base_url: str) -> str | None:
    # Try direct PDF links first, then WordPress download manager links.
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    candidates = []
    for h in hrefs:
        u = urllib.parse.urljoin(base_url, h)
        if ".pdf" in u.lower() or "wpdmdl=" in u.lower() or "drive.google.com" in u.lower():
            candidates.append(u)
    return candidates[0] if candidates else None


def main() -> int:
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8")))
    ok = 0; failed = 0
    for row in rows:
        local = ROOT / row["local_path"]
        local.parent.mkdir(parents=True, exist_ok=True)
        url = row["source_url"]
        # Skip if already exists and not empty.
        if local.exists() and local.stat().st_size > 0:
            print(f"SKIP existing: {local}")
            ok += 1
            continue
        try:
            print(f"Downloading {row['doc_id']} -> {local}")
            data = fetch(url)
            ctype = ""
            if local.suffix.lower() == ".pdf" and not data[:8].startswith(b"%PDF"):
                # Source URL is a landing page; try to find PDF/download link.
                html = data.decode("utf-8", errors="ignore")
                candidate = find_first_pdf_or_download_link(html, url)
                if candidate:
                    print(f"  found nested download link: {candidate}")
                    data = fetch(candidate)
            local.write_bytes(data)
            print(f"  saved {local.stat().st_size:,} bytes")
            ok += 1
            time.sleep(1.0)
        except Exception as e:
            failed += 1
            print(f"FAILED {row['doc_id']}: {e}", file=sys.stderr)
    print(f"Download completed: ok={ok}, failed={failed}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
