"""Perception dataset (Phase 2): frames -> ground-truth state targets.

Reads a Phase-1 dataset (`examples.jsonl`) plus the session video, extracts the referenced
frames (cached to disk), and builds regression/classification targets:

  - in_view      : is the opponent present AND within the horizontal view cone (BCE target)
  - bearing      : (sin, cos) of the bearing to the opponent (regressed only when in view)
  - dist         : horizontal distance, normalized (regressed only when in view)
  - self_health  : own health / 20 (always regressed — it's on the HUD)

Bearing/distance come straight from the Phase-1 egocentric features. "in view" is derived from
the bearing magnitude because the model can only see an opponent that's actually on screen — an
opponent directly behind the player leaves no pixels to learn from.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Target normalization / geometry constants.
FOV_HALF_DEG = 55.0     # opponent counts as "in view" within +/- this bearing
DIST_NORM = 32.0        # distance normalizer (blocks); clamps far opponents to ~1
MAX_HEALTH = 20.0


def load_examples(dataset_dir: str) -> List[dict]:
    with open(os.path.join(dataset_dir, "examples.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_frames(video_path: str, indices: List[int], size: Tuple[int, int],
                   cache_path: Optional[str] = None) -> Dict[int, np.ndarray]:
    """Decode the requested frame indices from the video, resized to (H, W). Cached to .npz.

    Returns {frame_idx: uint8 HxWx3 RGB}. Sequential decode (fast, no random seeks).
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        idxs = data["idxs"]
        frames = data["frames"]
        return {int(i): frames[k] for k, i in enumerate(idxs)}

    import cv2

    wanted = set(indices)
    H, W = size
    out: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {video_path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                out[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()

    if cache_path and out:
        idxs = np.array(sorted(out.keys()), dtype=np.int32)
        frames = np.stack([out[int(k)] for k in idxs])
        np.savez_compressed(cache_path, idxs=idxs, frames=frames)
    return out


def build_targets(examples: List[dict], fov_half_deg: float = FOV_HALF_DEG) -> Dict[str, np.ndarray]:
    """Vectorize per-example targets + a visibility mask. Pure numpy (no torch/cv2)."""
    n = len(examples)
    in_view = np.zeros(n, dtype=np.float32)
    bearing = np.zeros((n, 2), dtype=np.float32)   # (sin, cos)
    dist = np.zeros(n, dtype=np.float32)
    self_health = np.zeros(n, dtype=np.float32)

    for i, e in enumerate(examples):
        self_health[i] = e["self"]["health"] / MAX_HEALTH
        feat = e.get("features")
        if feat is None:
            continue
        bearing_deg = abs(math.degrees(math.atan2(feat["bearing_sin"], feat["bearing_cos"])))
        visible = bearing_deg <= fov_half_deg
        if visible:
            in_view[i] = 1.0
            bearing[i, 0] = feat["bearing_sin"]
            bearing[i, 1] = feat["bearing_cos"]
            dist[i] = min(feat["dist"] / DIST_NORM, 1.5)

    return {"in_view": in_view, "bearing": bearing, "dist": dist, "self_health": self_health}


def make_arrays(session_dir: str, dataset_dir: str, size: Tuple[int, int]):
    """Load examples, extract aligned frames, and align X (images) with Y (targets).

    Returns (X uint8 [N,H,W,3], targets dict, examples) with examples that had a decodable frame.
    """
    examples = load_examples(dataset_dir)
    indices = [e["frame"] for e in examples]
    H, W = size
    cache = os.path.join(dataset_dir, f"frames_{H}x{W}.npz")
    frames = extract_frames(os.path.join(session_dir, "video.mp4"), indices, size, cache_path=cache)

    kept = [e for e in examples if e["frame"] in frames]
    X = np.stack([frames[e["frame"]] for e in kept])
    targets = build_targets(kept)
    return X, targets, kept
