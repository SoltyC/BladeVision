"""Perception inference (Phase 2): a trained model -> state vector for one frame.

Used later by the Milestone-B live loop (frame -> state -> policy). Keeps the same output
semantics as the training targets.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from perception.model import PerceptionNet


class Perceiver:
    """Loads a checkpoint and turns a single RGB frame into a state dict."""

    def __init__(self, ckpt_path: str, device: str = None) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.size: Tuple[int, int] = tuple(ckpt["size"])
        self.dist_norm = ckpt["dist_norm"]
        self.max_health = ckpt["max_health"]
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = PerceptionNet().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, frame_rgb: np.ndarray) -> dict:
        """frame_rgb: HxWx3 uint8 (any size; resized to the model's input). Returns state."""
        import cv2

        H, W = self.size
        img = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        bearing = out["bearing"][0].cpu().numpy()
        return {
            "in_view": float(torch.sigmoid(out["in_view_logit"])[0].item()),
            "bearing_deg": math.degrees(math.atan2(float(bearing[0]), float(bearing[1]))),
            "dist_blocks": float(out["dist"][0].item()) * self.dist_norm,
            "self_health": float(out["self_health"][0].item()) * self.max_health,
        }
