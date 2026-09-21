"""Round segmentation for BladeVision captures (Phase 1).

A "round" is a continuous stretch of fighting: an opponent is present, and neither a large
time gap nor a health reset (respawn / new round / heal to full) interrupts it. Rounds are the
natural training/eval unit and let us label outcomes (win/loss).

Input is the list of per-frame example dicts produced by build.py (each has `t_wall`, `self`,
and optional `opponent`). Output is a list of round descriptors.
"""

from __future__ import annotations

from typing import List, Optional

# Heuristics (kept explicit rather than magic numbers scattered in the loop).
MAX_GAP_SECONDS = 3.0        # a gap longer than this splits rounds
HEALTH_RESET_JUMP = 6.0      # an upward health jump this large = respawn/new round
LOW_HEALTH_EPS = 0.5         # <= this counts as "dead"
MIN_ROUND_FRAMES = 10        # ignore blips shorter than this


def segment_rounds(examples: List[dict]) -> List[dict]:
    """Split aligned examples (with opponents) into rounds. Returns round descriptors."""
    active = [(i, e) for i, e in enumerate(examples) if e.get("opponent") is not None]
    if not active:
        return []

    rounds: List[dict] = []
    start_pos = 0
    for k in range(1, len(active)):
        prev_i, prev_e = active[k - 1]
        cur_i, cur_e = active[k]
        if _is_boundary(prev_e, cur_e):
            rounds.append(_make_round(active[start_pos:k]))
            start_pos = k
    rounds.append(_make_round(active[start_pos:]))

    # Filter out too-short blips and renumber.
    rounds = [r for r in rounds if r["n_frames"] >= MIN_ROUND_FRAMES]
    for idx, r in enumerate(rounds):
        r["round"] = idx
    return rounds


def _is_boundary(prev_e: dict, cur_e: dict) -> bool:
    if cur_e["t_wall"] - prev_e["t_wall"] > MAX_GAP_SECONDS:
        return True
    for who in ("self", "opponent"):
        prev_h = prev_e[who]["health"]
        cur_h = cur_e.get(who, {}).get("health")
        if cur_h is not None and cur_h - prev_h > HEALTH_RESET_JUMP:
            return True  # health jumped up = new round
    return False


def _make_round(active_slice: List) -> dict:
    idxs = [i for i, _ in active_slice]
    es = [e for _, e in active_slice]
    return {
        "start_frame": es[0]["frame"],
        "end_frame": es[-1]["frame"],
        "start_example": idxs[0],
        "end_example": idxs[-1],
        "n_frames": len(es),
        "duration_s": round(es[-1]["t_wall"] - es[0]["t_wall"], 2),
        "outcome": _outcome(es),
    }


def _outcome(es: List[dict]) -> str:
    last = es[-1]
    self_h = last["self"]["health"]
    opp_h = last["opponent"]["health"] if last.get("opponent") else None
    if opp_h is not None and opp_h <= LOW_HEALTH_EPS:
        return "win"
    if self_h <= LOW_HEALTH_EPS:
        return "loss"
    return "undetermined"
