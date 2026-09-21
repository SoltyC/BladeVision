"""Train the BladeVision perception model (Phase 2).

Trains frame -> {in_view, bearing, dist, self_health} on a Phase-1 dataset, with a temporal
train/val split (val = the tail of the session, to reduce leakage from near-identical adjacent
frames). Reports honest, denormalized metrics and saves the best-val checkpoint.

Usage:
    python -m perception.train --session duel2 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.dataset import DIST_NORM, MAX_HEALTH, make_arrays
from perception.model import PerceptionNet


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_tensors(X: np.ndarray, t: dict):
    # X uint8 [N,H,W,3] -> float [N,3,H,W] in [0,1]
    img = torch.from_numpy(X).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    return (
        img,
        torch.from_numpy(t["in_view"]),
        torch.from_numpy(t["bearing"]),
        torch.from_numpy(t["dist"]),
        torch.from_numpy(t["self_health"]),
    )


def _angular_err_deg(pred_bearing: np.ndarray, true_bearing: np.ndarray) -> float:
    """Mean absolute bearing error in degrees between two (sin, cos) arrays."""
    pa = np.arctan2(pred_bearing[:, 0], pred_bearing[:, 1])
    ta = np.arctan2(true_bearing[:, 0], true_bearing[:, 1])
    d = np.abs((pa - ta + math.pi) % (2 * math.pi) - math.pi)
    return float(np.degrees(d).mean())


def evaluate(model, loader, device) -> dict:
    model.eval()
    n = 0
    in_view_correct = 0
    health_abs = 0.0
    vis_pred_bear, vis_true_bear, vis_dist_abs = [], [], []
    with torch.no_grad():
        for img, iv, bear, dist, hp in loader:
            img = img.to(device)
            out = model(img)
            iv_pred = (torch.sigmoid(out["in_view_logit"]).cpu() > 0.5).float()
            in_view_correct += (iv_pred == iv).sum().item()
            health_abs += (out["self_health"].cpu() - hp).abs().sum().item() * MAX_HEALTH
            n += len(iv)
            mask = iv.bool().numpy()
            if mask.any():
                vis_pred_bear.append(out["bearing"].cpu().numpy()[mask])
                vis_true_bear.append(bear.numpy()[mask])
                vis_dist_abs.append(
                    np.abs(out["dist"].cpu().numpy()[mask] - dist.numpy()[mask]) * DIST_NORM
                )
    metrics = {
        "in_view_acc": in_view_correct / n,
        "health_mae": health_abs / n,
    }
    if vis_pred_bear:
        pb = np.concatenate(vis_pred_bear)
        tb = np.concatenate(vis_true_bear)
        metrics["bearing_err_deg"] = _angular_err_deg(pb, tb)
        metrics["dist_mae_blocks"] = float(np.concatenate(vis_dist_abs).mean())
    return metrics


def train(session: str, sessions_dir: str, dataset_dir: str, out_dir: str,
          epochs: int, batch: int, size, val_frac: float, lr: float) -> dict:
    session_path = os.path.join(sessions_dir, session)
    ds_path = os.path.join(dataset_dir, session)
    X, targets, kept = make_arrays(session_path, ds_path, size)
    print(f"loaded {len(kept)} frames  (visible/in-view: {int(targets['in_view'].sum())})")

    n = len(kept)
    n_val = max(1, int(n * val_frac))
    split = n - n_val  # temporal: last n_val are validation
    img, iv, bear, dist, hp = _to_tensors(X, targets)

    tr = TensorDataset(img[:split], iv[:split], bear[:split], dist[:split], hp[:split])
    va = TensorDataset(img[split:], iv[split:], bear[split:], dist[split:], hp[split:])
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch, shuffle=False)

    device = _device()
    model = PerceptionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "model.pt")
    best = math.inf
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        for img_b, iv_b, bear_b, dist_b, hp_b in tr_loader:
            img_b = img_b.to(device)
            iv_b, bear_b, dist_b, hp_b = iv_b.to(device), bear_b.to(device), dist_b.to(device), hp_b.to(device)
            out = model(img_b)
            loss = bce(out["in_view_logit"], iv_b)
            loss = loss + ((out["self_health"] - hp_b) ** 2).mean()
            mask = iv_b.bool()
            if mask.any():
                loss = loss + ((out["bearing"][mask] - bear_b[mask]) ** 2).mean()
                loss = loss + ((out["dist"][mask] - dist_b[mask]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        m = evaluate(model, va_loader, device)
        # Rank checkpoints by bearing error when available, else by health MAE.
        score = m.get("bearing_err_deg", m["health_mae"] * 10)
        tag = ""
        if score < best:
            best, best_metrics = score, m
            torch.save({"state_dict": model.state_dict(), "size": list(size),
                        "dist_norm": DIST_NORM, "max_health": MAX_HEALTH}, ckpt_path)
            tag = "  *saved"
        print(f"epoch {epoch:2d} | in_view {m['in_view_acc']:.2f} | "
              f"bearing {m.get('bearing_err_deg', float('nan')):5.1f}deg | "
              f"dist {m.get('dist_mae_blocks', float('nan')):4.1f}b | "
              f"hp_mae {m['health_mae']:.2f}{tag}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"best": best_metrics, "n_frames": n, "n_val": n_val}, fh, indent=2)
    print(f"\nbest val: {best_metrics}\nsaved -> {ckpt_path}")
    return best_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="BladeVision perception trainer (Phase 2)")
    p.add_argument("--session", required=True)
    p.add_argument("--sessions-dir", default="data/sessions")
    p.add_argument("--dataset-dir", default="data/dataset")
    p.add_argument("--out", default=None, help="default: perception/runs/<session>")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--height", type=int, default=90)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    out = args.out or os.path.join("perception", "runs", args.session)
    train(args.session, args.sessions_dir, args.dataset_dir, out,
          args.epochs, args.batch, (args.height, args.width), args.val_frac, args.lr)


if __name__ == "__main__":
    main()
