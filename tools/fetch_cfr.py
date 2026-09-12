"""Re-extract the 38 CFR 4.25 Table I fixture from the official eCFR API.

The fixture at fixtures/cfr425_table1_points.json is committed so the test
suite runs offline. This script exists so a reviewer can independently
regenerate it and diff, rather than trusting a committed blob.

    python tools/fetch_cfr.py            # writes .refs/ and prints a diff summary
    python tools/fetch_cfr.py --write    # also overwrites the committed fixture

Note: www.ecfr.gov serves a CAPTCHA to scrapers. This uses the documented
public API instead, which requires an Accept-Encoding header permitting
compression. No API key, no account, no rate limit encountered.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "cfr425_table1_points.json"
API = (
    "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-38.xml"
    "?subtitle=A&part=4&section={section}"
)
# Table I's column headings.
COLUMNS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
EXPECTED_POINTS = 684


def fetch_section(section: str, date: str) -> str:
    req = urllib.request.Request(
        API.format(date=date, section=section),
        headers={
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "recheck-hackathon-verification/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        import gzip

        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_table1(xml: str) -> list[tuple[int, int, int]]:
    """Extract (running_value, next_rating, published_combined) triples."""
    body = xml[xml.find("<TBODY>") : xml.find("</TBODY>")]
    points: list[tuple[int, int, int]] = []
    for row in re.findall(r"<TR>(.*?)</TR>", body, re.S):
        cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<TD[^>]*>(.*?)</TD>", row, re.S)
        ]
        cells = [c for c in cells if c]
        if not cells:
            continue
        try:
            running = int(cells[0])
        except ValueError:
            continue
        for index, text in enumerate(cells[1:]):
            if index >= len(COLUMNS):
                break
            try:
                points.append((running, COLUMNS[index], int(text)))
            except ValueError:
                continue
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-01", help="eCFR point-in-time date")
    ap.add_argument("--write", action="store_true", help="overwrite the committed fixture")
    args = ap.parse_args()

    xml = fetch_section("4.25", args.date)
    refs = ROOT / ".refs"
    refs.mkdir(exist_ok=True)
    (refs / "cfr425.xml").write_text(xml, encoding="utf-8")

    fresh = parse_table1(xml)
    print(f"extracted {len(fresh)} points from eCFR ({args.date})")
    if len(fresh) != EXPECTED_POINTS:
        print(f"  WARNING: expected {EXPECTED_POINTS}; the table may have changed", file=sys.stderr)

    if FIXTURE.exists():
        committed = [tuple(p) for p in json.loads(FIXTURE.read_text())]
        if committed == fresh:
            print("committed fixture is IDENTICAL to the live regulation")
        else:
            only_live = set(fresh) - set(committed)
            only_committed = set(committed) - set(fresh)
            print(f"  DIFFERS: {len(only_live)} new, {len(only_committed)} removed")
            for p in list(only_live)[:10]:
                print(f"    live only: {p}")
            for p in list(only_committed)[:10]:
                print(f"    committed only: {p}")

    if args.write:
        FIXTURE.write_text(json.dumps(fresh))
        print(f"wrote {FIXTURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
