"""A/B shipped evidence chain vs pre-bc00cd4 on one audio file.

    python -m tests.eval.ab_compare path/to/guitar_or_mix.wav
    python -m tests.eval.ab_compare path/to/guitar.wav --drums path/to/drums.wav
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.filters import apply_evidence_filters
from src.pipeline.transcribe import transcribe_guitar, transcribe_pre_bc00cd4


def _bucket(notes, width: float = 1.0) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for note in notes:
        counts[int(note.start_s // width)] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--drums", type=Path, default=None)
    parser.add_argument("--mode", default="balanced")
    args = parser.parse_args()
    if not args.audio.is_file():
        print(f"Missing audio: {args.audio}", file=sys.stderr)
        return 2
    ref = transcribe_pre_bc00cd4(args.audio)
    result = transcribe_guitar(args.audio, sensitivity=args.mode)
    drums = args.drums if args.drums and args.drums.is_file() else None
    shipped = apply_evidence_filters(
        result.notes,
        args.audio,
        drums_wav=drums,
        model_output=result.model_output,
        mode=args.mode,
    )
    print(f"pre-bc00cd4 (0.5/0.3, no filters): {len(ref)}")
    print(f"raw Basic Pitch ({args.mode}): {len(result.notes)}")
    print(f"shipped evidence score: {len(shipped)}")
    print(f"delta vs pre-bc00cd4: {len(shipped) - len(ref):+d}")
    print("notes / second (shipped):", _bucket(shipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
