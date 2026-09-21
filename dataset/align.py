"""Time-alignment of Phase-0 streams (Phase 1).

Frames (30 Hz) and ground-truth ticks (~20 Hz) are captured by separate processes and share
the machine wall clock. Here we (a) nearest-join each frame to a truth tick, and (b) reduce the
event-based input log to a per-frame held-state snapshot.

Units note: the recorder writes `t_wall` in seconds (time.time); the mod writes it in
milliseconds (System.currentTimeMillis). `load_truth` normalizes truth to seconds.
"""

from __future__ import annotations

import bisect
import json
from typing import Dict, List, Optional, Tuple

from dataset.state import INPUT_MAP, empty_input_state


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_frames(path: str) -> List[dict]:
    frames = _read_jsonl(path)
    frames.sort(key=lambda f: f["t_wall"])
    return frames


def load_truth(path: str) -> List[dict]:
    """Load truth ticks, normalizing t_wall (ms) -> seconds into `t` for alignment."""
    truth = _read_jsonl(path)
    for r in truth:
        r["t"] = r["t_wall"] / 1000.0
    truth.sort(key=lambda r: r["t"])
    return truth


def nearest_join(frames: List[dict], truth: List[dict]) -> List[Tuple[dict, dict, float]]:
    """For each frame, find the truth tick with the closest wall-clock time.

    Returns (frame, truth_tick, dt) triples, dt = |t_frame - t_truth| in seconds.
    Uses binary search over the sorted truth times — O(n log m).
    """
    if not truth:
        return [(f, None, float("inf")) for f in frames]
    times = [r["t"] for r in truth]
    joined = []
    for f in frames:
        tf = f["t_wall"]
        i = bisect.bisect_left(times, tf)
        best_j, best_dt = None, float("inf")
        for j in (i - 1, i):  # neighbours around the insertion point
            if 0 <= j < len(times):
                dt = abs(times[j] - tf)
                if dt < best_dt:
                    best_dt, best_j = dt, j
        joined.append((f, truth[best_j], best_dt))
    return joined


def reduce_input_to_frames(
    input_events: List[dict], frame_times: List[float]
) -> List[Dict[str, bool]]:
    """Replay press/release events to produce the held input state at each frame time.

    frame_times must be ascending. Events are applied in wall-clock order; each frame snapshots
    the currently-held channels.
    """
    events = sorted(input_events, key=lambda e: e["t_wall"])
    held = empty_input_state()
    states: List[Dict[str, bool]] = []
    ei = 0
    n = len(events)
    for tf in frame_times:
        while ei < n and events[ei]["t_wall"] <= tf:
            _apply_event(events[ei], held)
            ei += 1
        states.append(dict(held))
    return states


def _apply_event(event: dict, held: Dict[str, bool]) -> None:
    etype = event.get("type")
    if etype == "key":
        channel = INPUT_MAP.get(event.get("key"))
        if channel is not None:
            held[channel] = bool(event.get("pressed"))
    elif etype == "click":
        channel = INPUT_MAP.get(event.get("button"))
        if channel is not None:
            held[channel] = bool(event.get("pressed"))
    # scroll events don't map to a held channel
