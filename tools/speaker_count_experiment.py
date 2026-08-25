#!/usr/bin/env python3
"""One-off experiment (not a permanent tool): measure how pyannote's
speaker-count hints (num_speakers vs min_speakers/max_speakers vs none)
affect real diarization output on a real NOTSOFAR-1 meeting with a KNOWN
ground-truth speaker count, to inform whether live_stt should add
min/max_speakers plumbing (currently only num_speakers is wired, see
live_stt/diarization.py).

Audio: MTG_32063 sc_meetup_0/ch0.wav -- 16kHz mono, ~366s, TRUE speaker
count = 3 (Beth, Linda, Rachel), confirmed against gt_transcription.json.
Deliberately a different meeting from the MTG_32089 (5-speaker) one already
used elsewhere in this repo's history, and deliberately NOT 2-speaker (no
2-speaker meeting exists in this eval set) -- 3 lets us test both "hint
matches truth", "hint is the current hardcoded default (2, too LOW)", and
"hint is hardcoded too HIGH (5)" against the same audio in one pass.

Runs pipeline() directly (bypassing diarize_file's kwargs restriction, since
min_speakers/max_speakers aren't wired into live_stt/diarization.py yet) so
we can pass min_speakers/max_speakers for real and see what pyannote's
actual installed VBxClustering does with them.

Scoring: same informal 100ms-frame purity/agreement metric CLAUDE.md
describes for the earlier MTG_32089 run (bijective best-match per predicted
cluster, no Hungarian optimum, sanity-check-grade only) -- reimplemented
here since the original script was never committed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live_stt.config import Settings  # noqa: E402
from live_stt.diarization import annotation_to_house_json, load_pipeline  # noqa: E402

WAV_PATH = (
    "/data/vmfs/main01a_shared/Download/NOTSOFAR-1/eval_set/"
    "240825.1_eval_full_with_GT/MTG/MTG_32063/sc_meetup_0/ch0.wav"
)
GT_PATH = (
    "/data/vmfs/main01a_shared/Download/NOTSOFAR-1/eval_set/"
    "240825.1_eval_full_with_GT/MTG/MTG_32063/gt_transcription.json"
)
TRUE_SPEAKERS = ["Beth", "Linda", "Rachel"]
FRAME_SEC = 0.1

# (label, kwargs) -- kwargs passed straight to pipeline(wav, **kwargs)
CONFIGS: list[tuple[str, dict]] = [
    ("none (no hint at all)", {}),
    ("num_speakers=2 (current hardcoded DEFAULT)", {"num_speakers": 2}),
    ("num_speakers=3 (exact, matches truth)", {"num_speakers": 3}),
    ("num_speakers=5 (hardcoded too HIGH)", {"num_speakers": 5}),
    ("min=1,max=5 (loose band bracketing truth)", {"min_speakers": 1, "max_speakers": 5}),
    ("min=4,max=6 (band EXCLUDING truth)", {"min_speakers": 4, "max_speakers": 6}),
]


def build_gt_frames(gt_path: str, frame_sec: float, total_sec: float) -> list[str | None]:
    gt = json.loads(Path(gt_path).read_text())
    n_frames = int(total_sec / frame_sec) + 1
    frames: list[str | None] = [None] * n_frames
    for seg in gt:
        start_f = int(seg["start_time"] / frame_sec)
        end_f = int(seg["end_time"] / frame_sec)
        speaker = seg["speaker_id"]
        for f in range(max(0, start_f), min(n_frames, end_f + 1)):
            frames[f] = speaker
    return frames


def pred_frames_from_segments(segments: list[dict], frame_sec: float, n_frames: int) -> list[str | None]:
    frames: list[str | None] = [None] * n_frames
    for seg in segments:
        start_f = int(seg["start"] / frame_sec)
        end_f = int(seg["end"] / frame_sec)
        label = seg["label"]
        for f in range(max(0, start_f), min(n_frames, end_f + 1)):
            frames[f] = label
    return frames


def score(gt_frames: list[str | None], pred_frames: list[str | None]) -> dict:
    """Bijective best-match per predicted cluster (greedy, not Hungarian-
    optimal -- matches the informal methodology already used elsewhere in
    this repo's history for the same kind of check)."""
    n = min(len(gt_frames), len(pred_frames))
    pred_labels = sorted({p for p in pred_frames[:n] if p is not None})

    # For each predicted label, find its dominant (mode) ground-truth match.
    cluster_to_gt: dict[str, str] = {}
    cluster_purity: dict[str, float] = {}
    for label in pred_labels:
        counts: dict[str, int] = {}
        total = 0
        for i in range(n):
            if pred_frames[i] == label and gt_frames[i] is not None:
                counts[gt_frames[i]] = counts.get(gt_frames[i], 0) + 1
                total += 1
        if not counts or total == 0:
            cluster_to_gt[label] = "?"
            cluster_purity[label] = 0.0
            continue
        best_gt, best_count = max(counts.items(), key=lambda kv: kv[1])
        cluster_to_gt[label] = best_gt
        cluster_purity[label] = best_count / total

    # Overall frame-level agreement, scored ONLY over frames where both
    # gt and prediction are non-None (matches the "overlapping scored
    # audio" framing already used elsewhere in this repo's history).
    scored = 0
    correct = 0
    for i in range(n):
        g, p = gt_frames[i], pred_frames[i]
        if g is None or p is None:
            continue
        scored += 1
        if cluster_to_gt.get(p) == g:
            correct += 1
    agreement = correct / scored if scored else 0.0

    return {
        "predicted_num_clusters": len(pred_labels),
        "cluster_to_gt": cluster_to_gt,
        "cluster_purity": cluster_purity,
        "frame_agreement": agreement,
        "scored_frames": scored,
        "scored_sec": scored * FRAME_SEC,
    }


def main() -> None:
    settings = Settings(_env_file=None)
    print(f"loading pipeline ({settings.diarization_model!r})...", flush=True)
    t0 = time.time()
    pipeline = load_pipeline(settings)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    total_sec = 366.0  # from wave inspection; only used to size the gt frame array
    gt_frames = build_gt_frames(GT_PATH, FRAME_SEC, total_sec)
    n_frames = len(gt_frames)

    results = []
    for label, kwargs in CONFIGS:
        print(f"\n=== running config: {label} kwargs={kwargs} ===", flush=True)
        t0 = time.time()
        try:
            annotation = pipeline(WAV_PATH, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"CONFIG FAILED: {exc}", flush=True)
            results.append({"label": label, "kwargs": kwargs, "error": str(exc)})
            continue
        wall = time.time() - t0
        house_json = annotation_to_house_json(annotation, model=settings.diarization_model)
        pred_frames = pred_frames_from_segments(house_json["segments"], FRAME_SEC, n_frames)
        s = score(gt_frames, pred_frames)
        row = {
            "label": label,
            "kwargs": kwargs,
            "wall_sec": round(wall, 1),
            "predicted_num_speakers": house_json["num_speakers"],
            "num_segments": len(house_json["segments"]),
            **s,
        }
        results.append(row)
        print(json.dumps(row, indent=2, default=str), flush=True)

    print("\n\n=== SUMMARY ===")
    print(f"{'config':45s} {'wall_s':>7s} {'pred_n':>7s} {'true_n':>7s} {'agreement':>10s}")
    for r in results:
        if "error" in r:
            print(f"{r['label']:45s} FAILED: {r['error']}")
            continue
        print(
            f"{r['label']:45s} {r['wall_sec']:>7.1f} {r['predicted_num_speakers']:>7d} "
            f"{len(TRUE_SPEAKERS):>7d} {r['frame_agreement']:>10.3f}"
        )

    out_path = REPO_ROOT / "tools" / "speaker_count_experiment_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
