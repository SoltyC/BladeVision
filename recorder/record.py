"""BladeVision session recorder (Phase 0) — CLI entry point.

Runs frame capture (main thread) + input capture (background threads) into one timestamped
session directory, alongside a meta.json describing the capture (sensitivity/FOV/GUI-scale
matter for the pixel->rotation mapping, see docs/DESIGN.md §2.3).

Usage:
    python -m recorder.record --out data/sessions --name duel1 --fps 30 --scale 0.5 \
        --sensitivity 100 --fov 90 --gui-scale 2
    # Stop with the F8 hotkey or Ctrl-C.

macOS permissions (System Settings > Privacy & Security):
    - Screen Recording  -> your terminal/Python (for frame capture)
    - Input Monitoring / Accessibility -> your terminal/Python (for input hooks)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

from recorder.capture import FrameRecorder
from recorder.input_hooks import InputRecorder


def _parse_region(text):
    if not text:
        return None
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be 'left,top,width,height'")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="BladeVision session recorder (Phase 0)")
    parser.add_argument("--out", default="data/sessions", help="parent directory for sessions")
    parser.add_argument("--name", default=None, help="session name (default: timestamp)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=0.5, help="downscale factor for frames")
    parser.add_argument("--region", type=_parse_region, default=None,
                        help="capture region 'left,top,width,height' (default: primary monitor)")
    # Capture-config metadata (affects pixel<->rotation mapping; record it).
    parser.add_argument("--sensitivity", type=float, default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--gui-scale", type=int, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    name = args.name or time.strftime("session_%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.out, name)
    os.makedirs(session_dir, exist_ok=True)

    meta = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": args.fps,
        "scale": args.scale,
        "region": args.region,
        "sensitivity": args.sensitivity,
        "fov": args.fov,
        "gui_scale": args.gui_scale,
        "notes": args.notes,
        "clock": "t_wall aligns with the ground-truth mod's currentTimeMillis (same machine)",
    }
    with open(os.path.join(session_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    stop_event = threading.Event()
    inputs = InputRecorder(session_dir, on_stop_hotkey=stop_event.set)
    frames = FrameRecorder(session_dir, region=args.region, fps=args.fps, scale=args.scale)

    print(f"Recording -> {session_dir}")
    print("Press F8 or Ctrl-C to stop.")
    inputs.start()
    t0 = time.time()
    try:
        frames.run(stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        inputs.stop()

    dur = time.time() - t0
    print(f"\nDone. {frames.frames_written} frames ({frames.frames_written / max(dur, 1e-6):.1f} fps), "
          f"{inputs.events_written} input events, {dur:.1f}s.")
    print(f"Session: {session_dir}")


if __name__ == "__main__":
    main()
