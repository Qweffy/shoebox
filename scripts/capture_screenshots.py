"""Capture dashboard screenshots for the README via headless Playwright.

Needs the dashboard running (uv run uvicorn shoebox.web.app:app --port 8138).
uv run python scripts/capture_screenshots.py --base-url http://localhost:8138
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

_PAGES = (("/", "overview.png"), ("/ledger", "ledger.png"), ("/benchmark", "benchmark.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8138")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1024}, device_scale_factor=2)
        for path, name in _PAGES:
            page.goto(f"{args.base_url}{path}", wait_until="networkidle")
            page.wait_for_timeout(900)
            page.screenshot(path=str(args.out_dir / name), full_page=True)
            print(f"wrote {args.out_dir / name}")
        browser.close()


if __name__ == "__main__":
    main()
