"""Build a BladeVision training dataset from a Phase-0 capture session (Phase 1).

Joins a recorder session (frames + input) with the ground-truth mod log on wall-clock time,
derives the egocentric state features perception will learn to predict, segments rounds, and
writes:
    data/dataset/<name>/examples.jsonl   one aligned frame per line (truth + features + input)
    data/dataset/<name>/rounds.json      round boundaries + outcomes
    data/dataset/<name>/stats.json       summary + alignment quality

Usage:
    python -m dataset.build --session duel2
    python -m dataset.build --session duel2 --truth data/truth/truth_20260922_061603.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import List, Optional, Tuple

from dataset.align import load_frames, load_truth, nearest_join, reduce_input_to_frames, _read_jsonl
from dataset.rounds import segment_rounds
from dataset.state import egocentric_features

ALIGN_TOLERANCE_S = 0.08  # a frame farther than this from any tick is considered unaligned


def _truth_span(path: str) -> Optional[Tuple[float, float]]:
    """Cheaply read a truth file's [t_lo, t_hi] in seconds, or None if empty/bad."""
    rows = _read_jsonl(path)
    if not rows:
        return None
    ts = [r["t_wall"] / 1000.0 for r in rows]
    return min(ts), max(ts)


def auto_select_truth(truth_dir: str, frame_lo: float, frame_hi: float) -> Optional[str]:
    """Pick the truth file with the most wall-clock overlap with the frame window."""
    best_path, best_overlap = None, 0.0
    for path in glob.glob(os.path.join(truth_dir, "*.jsonl")):
        span = _truth_span(path)
        if span is None:
            continue
        overlap = min(frame_hi, span[1]) - max(frame_lo, span[0])
        if overlap > best_overlap:
            best_overlap, best_path = overlap, path
    return best_path


def build(session_dir: str, truth_path: Optional[str], out_dir: str,
          truth_dir: str = "data/truth") -> dict:
    frames = load_frames(os.path.join(session_dir, "frames.jsonl"))
    if not frames:
        raise SystemExit(f"No frames in {session_dir}/frames.jsonl")
    inputs = _read_jsonl(os.path.join(session_dir, "input.jsonl"))
    frame_lo, frame_hi = frames[0]["t_wall"], frames[-1]["t_wall"]

    if truth_path is None:
        truth_path = auto_select_truth(truth_dir, frame_lo, frame_hi)
        if truth_path is None:
            raise SystemExit(f"No overlapping truth file found in {truth_dir}")
        print(f"Auto-selected truth: {truth_path}")
    truth = load_truth(truth_path)

    joined = nearest_join(frames, truth)
    input_states = reduce_input_to_frames(inputs, [f["t_wall"] for f in frames])

    examples: List[dict] = []
    dts: List[float] = []
    dropped = 0
    for (frame, tick, dt), inp in zip(joined, input_states):
        dts.append(dt)
        if tick is None or dt > ALIGN_TOLERANCE_S:
            dropped += 1
            continue
        opp = tick.get("opponent")
        examples.append({
            "frame": frame["idx"],
            "t_wall": frame["t_wall"],
            "align_dt": round(dt, 4),
            "tick": tick.get("tick"),
            "self": tick["self"],
            "opponent": opp,
            "features": egocentric_features(tick["self"], opp),
            "input": inp,
        })

    rounds = segment_rounds(examples)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "examples.jsonl"), "w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    with open(os.path.join(out_dir, "rounds.json"), "w") as fh:
        json.dump(rounds, fh, indent=2)

    with_opp = sum(1 for e in examples if e["opponent"] is not None)
    stats = {
        "session": os.path.basename(os.path.normpath(session_dir)),
        "truth_file": os.path.basename(truth_path),
        "frames_total": len(frames),
        "examples_aligned": len(examples),
        "dropped_unaligned": dropped,
        "align_dt_ms": {
            "median": round(statistics.median(dts) * 1000, 1) if dts else None,
            "p95": round(sorted(dts)[int(len(dts) * 0.95)] * 1000, 1) if dts else None,
            "max": round(max(dts) * 1000, 1) if dts else None,
        },
        "examples_with_opponent": with_opp,
        "opponent_coverage": round(with_opp / len(examples), 3) if examples else 0,
        "rounds": len(rounds),
        "input_frames_any_action": sum(1 for e in examples if any(e["input"].values())),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    return {"stats": stats, "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description="BladeVision dataset builder (Phase 1)")
    parser.add_argument("--session", required=True, help="session name under --sessions-dir")
    parser.add_argument("--sessions-dir", default="data/sessions")
    parser.add_argument("--truth", default=None, help="truth jsonl (default: auto-select by overlap)")
    parser.add_argument("--truth-dir", default="data/truth")
    parser.add_argument("--out", default=None, help="output dir (default: data/dataset/<session>)")
    args = parser.parse_args()

    session_dir = os.path.join(args.sessions_dir, args.session)
    out_dir = args.out or os.path.join("data", "dataset", args.session)
    result = build(session_dir, args.truth, out_dir, truth_dir=args.truth_dir)

    s = result["stats"]
    print(f"\n=== dataset: {s['session']} -> {out_dir} ===")
    print(f"  aligned examples : {s['examples_aligned']}/{s['frames_total']} "
          f"(dropped {s['dropped_unaligned']})")
    print(f"  alignment dt (ms): median {s['align_dt_ms']['median']}, "
          f"p95 {s['align_dt_ms']['p95']}, max {s['align_dt_ms']['max']}")
    print(f"  opponent coverage: {s['opponent_coverage']:.1%} "
          f"({s['examples_with_opponent']} frames)")
    print(f"  frames w/ input  : {s['input_frames_any_action']}")
    print(f"  rounds           : {s['rounds']}")
    for r in result["rounds"]:
        print(f"    round {r['round']}: frames {r['start_frame']}-{r['end_frame']} "
              f"({r['n_frames']} ex, {r['duration_s']}s) -> {r['outcome']}")


if __name__ == "__main__":
    main()
