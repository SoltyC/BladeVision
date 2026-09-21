"""Unit tests for Phase-2 perception (targets + model forward shapes).

Uses numpy + torch (installed for training). Run: `python tests/test_perception.py`.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from perception.dataset import DIST_NORM, FOV_HALF_DEG, build_targets


def _example(health, bearing_deg=None, dist=None):
    if bearing_deg is None:
        return {"self": {"health": health}, "features": None, "opponent": None}
    b = math.radians(bearing_deg)
    return {
        "self": {"health": health},
        "opponent": {"health": 20},
        "features": {"bearing_sin": math.sin(b), "bearing_cos": math.cos(b), "dist": dist},
    }


def test_in_view_true_when_within_cone():
    t = build_targets([_example(20, bearing_deg=10.0, dist=4.0)])
    assert t["in_view"][0] == 1.0
    assert abs(t["dist"][0] - 4.0 / DIST_NORM) < 1e-6
    # bearing recovered
    ang = math.degrees(math.atan2(t["bearing"][0, 0], t["bearing"][0, 1]))
    assert abs(ang - 10.0) < 1e-4


def test_out_of_view_when_beyond_cone():
    t = build_targets([_example(20, bearing_deg=FOV_HALF_DEG + 20, dist=4.0)])
    assert t["in_view"][0] == 0.0
    # bearing/dist targets stay zero (masked out of the regression loss)
    assert t["dist"][0] == 0.0 and t["bearing"][0, 0] == 0.0


def test_no_opponent_gives_zero_in_view_but_keeps_health():
    t = build_targets([_example(10, bearing_deg=None)])
    assert t["in_view"][0] == 0.0
    assert abs(t["self_health"][0] - 0.5) < 1e-6  # 10/20


def test_model_forward_shapes():
    import torch
    from perception.model import PerceptionNet

    m = PerceptionNet()
    out = m(torch.zeros(2, 3, 90, 160))
    assert out["in_view_logit"].shape == (2,)
    assert out["bearing"].shape == (2, 2)
    assert out["dist"].shape == (2,)
    assert out["self_health"].shape == (2,)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
