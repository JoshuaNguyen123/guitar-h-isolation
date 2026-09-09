from __future__ import annotations

import html
import json
from pathlib import Path

from tests.e2e.clip_suite import SUITE_DIR


def write_suite_report(results: list[dict]) -> Path:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    (SUITE_DIR / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cards = []
    for item in results:
        status = "PASS" if item.get("ok") else "FAIL"
        cards.append(
            f"""
            <article class="card" data-testid="clip-{html.escape(item['clip_id'])}" data-ok="{str(item.get('ok')).lower()}">
              <h2>{html.escape(item['title'])}</h2>
              <p class="kind">{html.escape(item['kind'])} · {html.escape(item['artist'])}</p>
              <p class="status" data-testid="status-{html.escape(item['clip_id'])}">{status}</p>
              <ul>
                <li>Expert notes: <b data-testid="notes-{html.escape(item['clip_id'])}">{item.get('expert_notes', 0)}</b></li>
                <li>BPM: {item.get('bpm') or "n/a"}</li>
                <li>Frets: {html.escape(str(item.get('unique_frets', [])))}</li>
                <li>guitar.ogg: {item.get('guitar_ogg_bytes', 0) // 1024} KB</li>
                <li>song.ogg: {item.get('song_ogg_bytes', 0) // 1024} KB</li>
              </ul>
            </article>
            """
        )
    passed = sum(1 for item in results if item.get("ok"))
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Guitar H Isolation clip suite</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #1c1914; color: #f3ead8; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ color: #e8c36a; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: #262218; border-radius: 12px; padding: 16px 18px; }}
    .status {{ font-weight: 700; color: #3ddc6c; }}
    .card[data-ok="false"] .status {{ color: #ef3b3b; }}
    .kind {{ color: #c4b8a0; }}
  </style>
</head>
<body>
  <main>
    <h1>Clip suite</h1>
    <p data-testid="suite-summary">{passed} of {len(results)} clips produced a Clone Hero folder with guitar notes.</p>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""
    out = SUITE_DIR / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
