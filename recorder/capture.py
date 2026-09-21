"""Screen-frame capture for the BladeVision recorder (Phase 0).

Grabs the game region at a fixed cadence and writes it to an H.264/mp4v video plus a
frame index (`frames.jsonl`) carrying both a monotonic and a wall-clock timestamp per frame.
The wall-clock stamp is what aligns frames to the Fabric ground-truth mod's per-tick log,
since both processes share the machine clock (see docs/DESIGN.md §2.2).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple


class FrameRecorder:
    """Captures frames on the calling thread until a stop flag is set."""

    def __init__(
        self,
        session_dir: str,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps: int = 30,
        scale: float = 1.0,
    ) -> None:
        self._session_dir = session_dir
        self._region = region  # (left, top, width, height) or None for full primary monitor
        self._fps = fps
        self._scale = scale
        self.frames_written = 0

    def _monitor(self, sct):
        if self._region is not None:
            left, top, width, height = self._region
            return {"left": left, "top": top, "width": width, "height": height}
        return sct.monitors[1]  # primary monitor (index 0 is the union of all monitors)

    def run(self, stop_event) -> None:
        """Capture loop. Blocks until `stop_event` is set. Requires mss + opencv + numpy."""
        try:
            import cv2
            import numpy as np
            import mss
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                f"Missing capture dependency ({exc.name}). "
                f"Install with: pip install -r recorder/requirements.txt"
            )

        video_path = os.path.join(self._session_dir, "video.mp4")
        index_path = os.path.join(self._session_dir, "frames.jsonl")
        frame_interval = 1.0 / self._fps

        with mss.mss() as sct, open(index_path, "w") as index:
            mon = self._monitor(sct)
            out_w = max(1, int(mon["width"] * self._scale))
            out_h = max(1, int(mon["height"] * self._scale))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, self._fps, (out_w, out_h))
            if not writer.isOpened():
                raise SystemExit(f"Could not open video writer at {video_path}")

            next_t = time.perf_counter()
            try:
                while not stop_event.is_set():
                    grab = np.asarray(sct.grab(mon))  # BGRA
                    frame = cv2.cvtColor(grab, cv2.COLOR_BGRA2BGR)
                    if self._scale != 1.0:
                        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    index.write(json.dumps({
                        "idx": self.frames_written,
                        "t_perf": time.perf_counter(),
                        "t_wall": time.time(),
                    }) + "\n")
                    index.flush()  # survive a hard kill; the video writer already flushes
                    self.frames_written += 1

                    # Maintain cadence without drift; if we fall behind, resync.
                    next_t += frame_interval
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.perf_counter()
            finally:
                writer.release()
