"""Dependency-free unit tests for the Phase-1 dataset builder.

Exercises nearest-join, event->held-state reduction, egocentric features, and round
segmentation on synthetic data (no numpy/opencv/game needed).
Run: `python tests/test_dataset.py` or `pytest tests/test_dataset.py`.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.align import nearest_join, reduce_input_to_frames
from dataset.rounds import segment_rounds
from dataset.state import egocentric_features


def test_nearest_join_picks_closest_tick():
    frames = [{"idx": 0, "t_wall": 1.02}, {"idx": 1, "t_wall": 1.06}]
    truth = [{"t": 0.95}, {"t": 1.05}, {"t": 1.20}]
    joined = nearest_join(frames, truth)
    assert joined[0][1]["t"] == 1.05 and abs(joined[0][2] - 0.03) < 1e-9  # 1.05 (0.03) beats 0.95 (0.07)
    assert joined[1][1]["t"] == 1.05 and abs(joined[1][2] - 0.01) < 1e-9


def test_nearest_join_empty_truth():
    joined = nearest_join([{"idx": 0, "t_wall": 1.0}], [])
    assert joined[0][1] is None and joined[0][2] == float("inf")


def test_input_reduction_tracks_held_state():
    events = [
        {"type": "key", "key": "w", "pressed": True, "t_wall": 1.0},
        {"type": "click", "button": "Button.left", "pressed": True, "t_wall": 1.2},
        {"type": "click", "button": "Button.left", "pressed": False, "t_wall": 1.3},
        {"type": "key", "key": "w", "pressed": False, "t_wall": 2.0},
    ]
    states = reduce_input_to_frames(events, [0.5, 1.1, 1.25, 1.5, 2.5])
    assert states[0]["forward"] is False           # before any press
    assert states[1]["forward"] is True            # w held
    assert states[2]["forward"] and states[2]["attack"]   # w + click held
    assert states[3]["forward"] and not states[3]["attack"]  # click released
    assert not states[4]["forward"]                # w released


def test_input_reduction_ignores_unmapped_and_stray_release():
    events = [
        {"type": "key", "key": "<63>", "pressed": True, "t_wall": 1.0},  # unmapped
        {"type": "key", "key": "d", "pressed": False, "t_wall": 1.1},    # release w/o press
        {"type": "scroll", "dx": 0, "dy": 1, "t_wall": 1.2},
    ]
    states = reduce_input_to_frames(events, [1.5])
    assert not any(states[0].values())  # nothing held


def test_egocentric_features_directions():
    # Self at origin facing south (yaw 0 -> +Z). Opponent 3 blocks south = dead ahead.
    f = egocentric_features({"x": 0, "y": 0, "z": 0, "yaw": 0.0},
                            {"x": 0, "y": 0, "z": 3.0})
    assert abs(f["fwd"] - 3.0) < 1e-6
    assert abs(f["right"]) < 1e-6
    assert abs(f["dist"] - 3.0) < 1e-6
    assert f["bearing_cos"] > 0.99  # dead ahead

    # Opponent to the player's right (west, -X) when facing south.
    f2 = egocentric_features({"x": 0, "y": 0, "z": 0, "yaw": 0.0},
                             {"x": -3.0, "y": 0, "z": 0})
    assert f2["right"] > 2.9 and f2["bearing_sin"] > 0.99


def test_egocentric_features_none_without_opponent():
    assert egocentric_features({"x": 0, "y": 0, "z": 0, "yaw": 0}, None) is None


def _ex(frame, t, self_h, opp_h):
    return {"frame": frame, "t_wall": t, "self": {"health": self_h},
            "opponent": {"health": opp_h}}


def test_segment_rounds_splits_on_gap_and_reset():
    exs = []
    # Round 0: frames 0..14, opponent dying at the end (health reaches 0).
    for i in range(15):
        exs.append(_ex(i, 100.0 + i * 0.05, 20 - i * 0.1, max(0.0, 20 - i * 1.6)))
    # Big time gap, then Round 1 with full health (reset).
    for i in range(15):
        exs.append(_ex(100 + i, 130.0 + i * 0.05, 20 - i * 0.2, 20 - i * 0.5))
    rounds = segment_rounds(exs)
    assert len(rounds) == 2
    assert rounds[0]["outcome"] == "win"   # opponent health ~0 at end of round 0
    assert rounds[0]["round"] == 0 and rounds[1]["round"] == 1


def test_segment_rounds_ignores_frames_without_opponent():
    exs = [{"frame": i, "t_wall": i * 0.05, "self": {"health": 20}, "opponent": None}
           for i in range(20)]
    assert segment_rounds(exs) == []


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
