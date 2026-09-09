from __future__ import annotations

import json
import sys
from pathlib import Path

from src.pipeline.run import run_pipeline
from tests.e2e.clip_suite import CLIPS, FIXTURES, SUITE_DIR, summarize_clip
from tests.e2e.write_suite_report import write_suite_report


def main() -> int:
    results = []
    for clip in CLIPS:
        src = FIXTURES / clip.source_name
        if not src.is_file():
            print(f"MISSING {src}", flush=True)
            return 1
        dest = SUITE_DIR / clip.clip_id
        print(f"=== {clip.clip_id} ({clip.kind}) ===", flush=True)

        def progress(stage: str, fraction: float, clip_id: str = clip.clip_id) -> None:
            print(f"[{clip_id}] {fraction:0.2f} {stage}", flush=True)

        run_pipeline(src, dest, clip.title, clip.artist, progress=progress)
        summary = summarize_clip(clip)
        results.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not summary["ok"]:
            print(f"FAILED {clip.clip_id}", flush=True)
            return 1

    report = write_suite_report(results)
    print(f"Wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
