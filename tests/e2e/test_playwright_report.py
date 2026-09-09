from __future__ import annotations

from pathlib import Path

import pytest

REPORT = (
    Path(__file__).resolve().parents[2]
    / "e2e_output"
    / "Acoustic Chords"
    / "e2e_report.html"
)
SCREENSHOT = REPORT.with_name("playwright_highway.png")


@pytest.mark.e2e
def test_playwright_reads_real_chart_report() -> None:
    if not REPORT.is_file():
        pytest.skip(f"Missing report: {REPORT}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(REPORT.resolve().as_uri())
        assert page.title() == "Guitar H Isolation E2E"
        expert = int(page.get_by_test_id("expert-count").inner_text())
        assert expert >= 20
        highway = page.get_by_test_id("highway")
        assert highway.locator(".gem").count() == expert
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        browser.close()
