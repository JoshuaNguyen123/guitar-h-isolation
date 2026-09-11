"""Score shipped evidence chain vs pre-bc00cd4 on GuitarSet labels.

    python -m tests.eval.run_guitarset
    python -m tests.eval.run_guitarset --download-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.filters import apply_evidence_filters
from src.pipeline.transcribe import transcribe_guitar, transcribe_pre_bc00cd4
from tests.eval.guitarset import (
    CACHE_DIR,
    dataset_ready,
    ensure_guitarset,
    iter_eval_items,
    render_band_mix,
)
from tests.eval.metrics import score_transcription

OUT_JSON = Path(__file__).resolve().parent / "guitarset_baseline.json"
OUT_MD = Path(__file__).resolve().parent / "BASELINE.md"
WORK = CACHE_DIR / "band_mix"
MODES = ("balanced",)
ALL_MODES = ("strict", "balanced", "sensitive")


def _shipped(path: Path, mode: str, drums: Path | None) -> list:
    result = transcribe_guitar(path, sensitivity=mode)
    return apply_evidence_filters(
        result.notes,
        path,
        drums_wav=drums,
        model_output=result.model_output,
        mode=mode,
        bpm=None,
        use_pyin=False,
    )


def _score_item(item: dict) -> dict:
    truth = item["truth"]
    band_wav = WORK / f"{item['track_id']}_band.wav"
    drums_wav = band_wav.with_name(band_wav.stem + "_drums.wav")
    if not band_wav.is_file() or not drums_wav.is_file():
        band_wav, drums_wav = render_band_mix(item["mic"], item["jams"], band_wav)

    solo_ref = transcribe_pre_bc00cd4(item["mic"])
    band_ref = transcribe_pre_bc00cd4(band_wav)
    row: dict = {
        "track_id": item["track_id"],
        "truth_notes": len(truth),
        "solo": {
            "reference": asdict(score_transcription(truth, solo_ref)),
        },
        "band": {
            "reference": asdict(score_transcription(truth, band_ref)),
        },
    }
    for mode in MODES:
        solo_now = _shipped(item["mic"], mode, None)
        band_now = _shipped(band_wav, mode, drums_wav)
        row["solo"][mode] = asdict(score_transcription(truth, solo_now))
        row["band"][mode] = asdict(score_transcription(truth, band_now))
        row["solo"][f"{mode}_notes"] = len(solo_now)
        row["band"][f"{mode}_notes"] = len(band_now)
    return row


def _mean(rows: list[dict], condition: str, key: str, field: str) -> float:
    vals = [row[condition][key][field] for row in rows if key in row[condition]]
    return float(sum(vals) / len(vals)) if vals else 0.0


def run(limit: int = 0) -> dict:
    items = iter_eval_items()
    if limit > 0:
        items = items[:limit]
    if not items:
        raise RuntimeError("No GuitarSet eval items. Run --download-only first.")
    rows = []
    for item in items:
        print(f"Scoring {item['track_id']} ({len(item['truth'])} GT notes)...")
        rows.append(_score_item(item))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tracks": len(rows),
        "mean_solo_f_reference": _mean(rows, "solo", "reference", "f_measure"),
        "mean_solo_f_shipped": _mean(rows, "solo", "balanced", "f_measure"),
        "mean_band_f_reference": _mean(rows, "band", "reference", "f_measure"),
        "mean_band_f_shipped": _mean(rows, "band", "balanced", "f_measure"),
        "mean_band_recall_reference": _mean(rows, "band", "reference", "recall"),
        "mean_band_recall_shipped": _mean(rows, "band", "balanced", "recall"),
        "mean_band_precision_strict": _mean(rows, "band", "strict", "precision"),
        "mean_band_precision_balanced": _mean(rows, "band", "balanced", "precision"),
        "tracks": rows,
    }
    report["acceptance"] = {
        "shipped_band_f_gte_reference": (
            report["mean_band_f_shipped"] + 1e-6 >= report["mean_band_f_reference"]
        ),
        "strict_precision_gte_balanced": (
            True
            if "strict" not in (rows[0].get("band") or {})
            else report["mean_band_precision_strict"] + 1e-6
            >= report["mean_band_precision_balanced"]
        ),
        "note": (
            "Shipped Balanced F must meet the pre-bc00cd4 reference "
            "(raw 0.5/0.3, no filters) on GuitarSet band mixes."
        ),
    }
    return report


def _append_baseline_section(report: dict) -> None:
    if not OUT_MD.is_file():
        return
    text = OUT_MD.read_text(encoding="utf-8")
    marker = "## 6. GuitarSet"
    block = (
        "\n## 6. GuitarSet (independent labels)\n\n"
        f"Generated: `{report['generated_at']}`\n\n"
        f"Tracks: {report['n_tracks']}. "
        f"Solo shipped F `{report['mean_solo_f_shipped']:.3f}` vs pre-bc00cd4 "
        f"`{report['mean_solo_f_reference']:.3f}`. "
        f"Band shipped F `{report['mean_band_f_shipped']:.3f}` vs pre-bc00cd4 "
        f"`{report['mean_band_f_reference']:.3f}`."
    )
    if report.get("mean_band_precision_strict"):
        block += (
            f" Band Strict P `{report['mean_band_precision_strict']:.3f}` vs "
            f"Balanced P `{report['mean_band_precision_balanced']:.3f}`.\n"
        )
    else:
        block += "\n"
    if marker in text:
        pre, _rest = text.split(marker, 1)
        tail = _rest.split("\n## ", 1)
        after = ("\n## " + tail[1]) if len(tail) == 2 else ""
        OUT_MD.write_text(pre.rstrip() + "\n" + block + after, encoding="utf-8")
    else:
        OUT_MD.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.all_modes:
        global MODES
        MODES = ALL_MODES
    ensure_guitarset()
    if args.download_only:
        print(f"GuitarSet ready at {CACHE_DIR}")
        return 0
    if not dataset_ready():
        print("GuitarSet cache incomplete.", file=sys.stderr)
        return 2
    report = run(limit=args.limit)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _append_baseline_section(report)
    print(json.dumps({k: report[k] for k in report if k != "tracks"}, indent=2))
    print(f"Wrote {OUT_JSON}")
    ok = report["acceptance"]["shipped_band_f_gte_reference"]
    if not ok:
        print("WARNING: shipped band F is below the pre-bc00cd4 reference.", file=sys.stderr)
        return 1
    if (
        "strict" in ((report.get("tracks") or [{}])[0].get("band") or {})
        and not report["acceptance"]["strict_precision_gte_balanced"]
    ):
        print("WARNING: Strict band precision is below Balanced.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
