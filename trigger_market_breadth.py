#!/usr/bin/env python3
"""
Market Breadth data fetcher and workflow trigger.

Checks freshness of GitHub Pages CSV data. If stale, triggers the
daily-market-breadth workflow via GitHub API.

Usage:
    python trigger_market_breadth.py                  # Auto: fetch if fresh, trigger if stale
    python trigger_market_breadth.py --trigger-only   # Force trigger workflow
    python trigger_market_breadth.py --fetch-only     # Fetch CSV only (no trigger)
    python trigger_market_breadth.py --max-age 6      # Staleness threshold in hours
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://tradermonty.github.io/market-breadth-analysis"
DATA_CSV_URL = f"{BASE_URL}/market_breadth_data.csv"
SUMMARY_CSV_URL = f"{BASE_URL}/market_breadth_summary.csv"

GITHUB_API_URL = (
    "https://api.github.com/repos/tradermonty/market-breadth-analysis"
    "/actions/workflows/daily-market-breadth.yml/dispatches"
)


def _get_github_token():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN not set. Add it to .env or export it."
        )
    return token


def _parse_last_modified(url):
    """Send HEAD request and return Last-Modified as aware datetime, or None."""
    resp = requests.head(url, timeout=10, allow_redirects=True)
    resp.raise_for_status()
    lm = resp.headers.get("Last-Modified")
    if not lm:
        return None
    from email.utils import parsedate_to_datetime
    return parsedate_to_datetime(lm)


def fetch_csv():
    """Download the market breadth CSV from GitHub Pages.

    Returns:
        dict with keys: csv_text, summary_text, last_modified
    """
    data_resp = requests.get(DATA_CSV_URL, timeout=30)
    data_resp.raise_for_status()

    summary_resp = requests.get(SUMMARY_CSV_URL, timeout=30)
    summary_resp.raise_for_status()

    lm = _parse_last_modified(DATA_CSV_URL)

    return {
        "csv_text": data_resp.text,
        "summary_text": summary_resp.text,
        "last_modified": lm,
    }


def trigger_workflow():
    """Trigger the daily-market-breadth workflow via GitHub API.

    Returns:
        dict with status info
    """
    token = _get_github_token()
    resp = requests.post(
        GITHUB_API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": "main"},
        timeout=15,
    )
    if resp.status_code == 204:
        return {
            "status": "triggered",
            "message": "Workflow triggered. Data will be ready in ~5 minutes.",
            "runs_url": (
                "https://github.com/tradermonty/market-breadth-analysis"
                "/actions/workflows/daily-market-breadth.yml"
            ),
        }
    resp.raise_for_status()


def fetch_market_breadth(max_age_hours=12):
    """Fetch market breadth data. Trigger workflow if data is stale.

    Args:
        max_age_hours: Maximum acceptable age of data in hours.

    Returns:
        dict with keys:
          - status: "fresh" | "triggered" | "error"
          - data_url / summary_url: CSV URLs
          - last_modified: Last-Modified datetime (if available)
          - csv_text: CSV content (only when status=="fresh")
          - summary_text: Summary CSV content (only when status=="fresh")
          - message: Human-readable status description
    """
    base = {
        "data_url": DATA_CSV_URL,
        "summary_url": SUMMARY_CSV_URL,
    }

    try:
        last_modified = _parse_last_modified(DATA_CSV_URL)
    except requests.RequestException as e:
        return {**base, "status": "error", "last_modified": None,
                "message": f"Failed to check data freshness: {e}"}

    age_hours = None
    if last_modified:
        age = datetime.now(timezone.utc) - last_modified
        age_hours = age.total_seconds() / 3600

    if age_hours is not None and age_hours <= max_age_hours:
        # Data is fresh — download it
        try:
            result = fetch_csv()
        except requests.RequestException as e:
            return {**base, "status": "error", "last_modified": last_modified,
                    "message": f"Data is fresh but download failed: {e}"}
        return {
            **base,
            "status": "fresh",
            "last_modified": last_modified,
            "csv_text": result["csv_text"],
            "summary_text": result["summary_text"],
            "message": f"Data is fresh (age: {age_hours:.1f}h).",
        }

    # Data is stale or age unknown — trigger workflow
    try:
        trigger_result = trigger_workflow()
    except EnvironmentError as e:
        return {**base, "status": "error", "last_modified": last_modified,
                "message": str(e)}
    except requests.RequestException as e:
        return {**base, "status": "error", "last_modified": last_modified,
                "message": f"Failed to trigger workflow: {e}"}

    age_msg = f" (age: {age_hours:.1f}h)" if age_hours is not None else ""
    return {
        **base,
        **trigger_result,
        "last_modified": last_modified,
        "message": f"Data is stale{age_msg}. {trigger_result['message']}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch market breadth data or trigger workflow"
    )
    parser.add_argument(
        "--trigger-only", action="store_true",
        help="Force trigger the workflow without checking freshness"
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Fetch CSV data only (no workflow trigger)"
    )
    parser.add_argument(
        "--max-age", type=float, default=12,
        help="Max data age in hours before triggering (default: 12)"
    )
    args = parser.parse_args()

    if args.trigger_only and args.fetch_only:
        print("Error: --trigger-only and --fetch-only are mutually exclusive.")
        sys.exit(1)

    if args.trigger_only:
        result = trigger_workflow()
        print(result["message"])
        print(f"Monitor: {result['runs_url']}")
        return

    if args.fetch_only:
        result = fetch_csv()
        lm = result["last_modified"]
        print(f"Last-Modified: {lm.isoformat() if lm else 'unknown'}")
        print(f"Data rows: {result['csv_text'].count(chr(10)) - 1}")
        print(result["csv_text"][:500])
        return

    # Default: auto mode
    result = fetch_market_breadth(max_age_hours=args.max_age)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result.get("last_modified"):
        print(f"Last-Modified: {result['last_modified'].isoformat()}")
    if result["status"] == "fresh":
        lines = result["csv_text"].strip().split("\n")
        print(f"Data rows: {len(lines) - 1}")
        print(f"Preview:\n{chr(10).join(lines[:3])}")


if __name__ == "__main__":
    main()
