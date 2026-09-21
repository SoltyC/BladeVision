"""Perception model (Phase 2): a compact CNN, frame -> state heads.

Deliberately small — the initial datasets are tiny, so a large backbone would only overfit.
Outputs the state signals the policy/executor need: opponent visibility, bearing, distance,
and own health. The same egocentric quantities the sim policy already consumes (§4.3), so
perception plugs in ahead of the Milestone-A brain.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    # stride-2 conv downsamples; BN + ReLU. One block per resolution halving.
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PerceptionNet(nn.Module):
    """Frame (B,3,H,W) -> dict of heads."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _block(3, width),          # /2
            _block(width, width * 2),  # /4
            _block(width * 2, width * 4),  # /8
            _block(width * 4, width * 4),  # /16
            nn.AdaptiveAvgPool2d(1),
        )
        feat = width * 4
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(feat, feat), nn.ReLU(inplace=True))
        self.head_in_view = nn.Linear(feat, 1)
        self.head_bearing = nn.Linear(feat, 2)
        self.head_dist = nn.Linear(feat, 1)
        self.head_health = nn.Linear(feat, 1)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.shared(self.backbone(x))
        return {
            "in_view_logit": self.head_in_view(h).squeeze(-1),
            "bearing": self.head_bearing(h),          # (B, 2) raw sin/cos
            "dist": self.head_dist(h).squeeze(-1),
            "self_health": self.head_health(h).squeeze(-1),
        }
