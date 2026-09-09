from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.clip_suite import CLIPS

REPORT = Path(__file__).resolve().parents[2] / "e2e_output" / "clip_suite" / "index.html"
SCREENSHOT = REPORT.with_name("playwright_suite.png")


@pytest.mark.e2e
def test_playwright_suite_shows_every_clip_passed() -> None:
    if not REPORT.is_file():
        pytest.skip(f"Missing suite report: {REPORT}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.goto(REPORT.resolve().as_uri())
        assert page.title() == "Guitar H Isolation clip suite"
        summary = page.get_by_test_id("suite-summary").inner_text()
        assert f"{len(CLIPS)} of {len(CLIPS)} clips" in summary
        for clip in CLIPS:
            status = page.get_by_test_id(f"status-{clip.clip_id}").inner_text()
            notes = int(page.get_by_test_id(f"notes-{clip.clip_id}").inner_text())
            assert status == "PASS"
            assert notes >= 1
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        browser.close()
