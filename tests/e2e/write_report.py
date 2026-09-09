from __future__ import annotations

import html
import json
import re
from pathlib import Path

CHART = Path(__file__).resolve().parents[2] / "e2e_output" / "Acoustic Chords" / "notes.chart"
INI = CHART.with_name("song.ini")
OUT = CHART.with_name("e2e_report.html")
LANE_COLORS = ["#3ddc6c", "#ef3b3b", "#f2d23c", "#3b7cff", "#f08a2a"]
LANE_NAMES = ["Green", "Red", "Yellow", "Blue", "Orange"]


def _section(text: str, name: str) -> str:
    match = re.search(rf"\[{name}\]\s*\{{(.*?)\n\}}", text, re.S)
    return match.group(1) if match else ""


def _notes(block: str) -> list[tuple[int, int, int]]:
    return [
        (int(tick), int(fret), int(sustain))
        for tick, fret, sustain in re.findall(r"(\d+) = N (\d+) (\d+)", block)
        if int(fret) <= 4
    ]


def main() -> Path:
    chart = CHART.read_text(encoding="utf-8")
    ini = INI.read_text(encoding="utf-8")
    bpm = int(re.search(r"0 = B (\d+)", chart).group(1)) / 1000.0
    expert = _notes(_section(chart, "ExpertSingle"))
    hard = _notes(_section(chart, "HardSingle"))
    medium = _notes(_section(chart, "MediumSingle"))
    easy = _notes(_section(chart, "EasySingle"))
    max_tick = max((n[0] for n in expert), default=1)
    gems = []
    for tick, fret, sustain in expert:
        left = 40 + (tick / max_tick) * 920
        width = max(10, (sustain / max_tick) * 920)
        top = 28 + fret * 42
        gems.append(
            f'<div class="gem" style="left:{left:.1f}px;top:{top}px;width:{width:.1f}px;'
            f'background:{LANE_COLORS[fret]}" title="{LANE_NAMES[fret]} @ {tick}"></div>'
        )
    report = {
        "song": "Acoustic Chords",
        "artist": "Frederic Jacquot",
        "bpm": bpm,
        "song_length_ms": 11000,
        "expert_notes": len(expert),
        "hard_notes": len(hard),
        "medium_notes": len(medium),
        "easy_notes": len(easy),
        "unique_frets": sorted({n[1] for n in expert}),
        "song_ogg_bytes": CHART.with_name("song.ogg").stat().st_size,
        "guitar_ogg_bytes": CHART.with_name("guitar.ogg").stat().st_size,
    }
    CHART.with_name("e2e_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Guitar H Isolation E2E</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, sans-serif; background: #1c1914; color: #f3ead8; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1 {{ color: #e8c36a; margin-bottom: 8px; }}
    .meta {{ color: #c4b8a0; margin-bottom: 24px; }}
    .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
    .stat {{ background: #262218; padding: 14px 16px; border-radius: 10px; min-width: 120px; }}
    .stat b {{ display: block; color: #e8c36a; font-size: 22px; }}
    .highway {{ position: relative; height: 250px; background: #11100d; border-radius: 12px; overflow: hidden; }}
    .lane {{ position: absolute; left: 0; right: 0; height: 42px; border-bottom: 1px solid #3a3428; }}
    .gem {{ position: absolute; height: 18px; border-radius: 9px; opacity: 0.92; }}
    pre {{ background: #11100d; padding: 16px; border-radius: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <h1>Guitar H Isolation — real E2E</h1>
    <p class="meta">Acoustic Chords · Frédéric Jacquot · {bpm:.1f} BPM · 11.0s guitar clip</p>
    <div class="stats">
      <div class="stat"><b data-testid="expert-count">{len(expert)}</b>Expert notes</div>
      <div class="stat"><b>{len(hard)}</b>Hard</div>
      <div class="stat"><b>{len(medium)}</b>Medium</div>
      <div class="stat"><b>{len(easy)}</b>Easy</div>
      <div class="stat"><b>{report['guitar_ogg_bytes'] // 1024} KB</b>guitar.ogg</div>
      <div class="stat"><b>{report['song_ogg_bytes'] // 1024} KB</b>song.ogg</div>
    </div>
    <div class="highway" data-testid="highway">
      {"".join(f'<div class="lane" style="top:{28 + i * 42}px"></div>' for i in range(5))}
      {"".join(gems)}
    </div>
    <h2>song.ini</h2>
    <pre>{html.escape(ini)}</pre>
  </main>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()
