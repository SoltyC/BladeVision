"""OS-level input capture for the BladeVision recorder (Phase 0).

Records keyboard presses/releases and mouse buttons/scroll with monotonic + wall-clock
timestamps to `input.jsonl`.

IMPORTANT: when Minecraft grabs the cursor for in-game look, the OS cursor stops moving, so
mouse-*motion* (aim) is NOT observable here — that signal comes from the ground-truth mod's
per-tick yaw/pitch instead (docs/DESIGN.md §2.1). This hook captures the discrete inputs
(movement keys, jump/sprint/sneak, attack/use clicks, hotbar) that the mod cannot see.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional


class InputRecorder:
    """Runs pynput listeners on background threads until stopped."""

    def __init__(self, session_dir: str, on_stop_hotkey: Optional[Callable[[], None]] = None) -> None:
        import os

        self._path = os.path.join(session_dir, "input.jsonl")
        self._file = open(self._path, "w")
        self._lock = threading.Lock()
        self._kb = None
        self._ms = None
        self._on_stop_hotkey = on_stop_hotkey  # called when the stop hotkey (F8) is pressed
        self.events_written = 0

    def _write(self, event: dict) -> None:
        event["t_perf"] = time.perf_counter()
        event["t_wall"] = time.time()
        with self._lock:
            self._file.write(json.dumps(event) + "\n")
            self.events_written += 1

    # --- keyboard ---------------------------------------------------------
    def _on_press(self, key) -> None:
        name = getattr(key, "char", None) or str(key)
        if name == "Key.f8" and self._on_stop_hotkey is not None:
            self._on_stop_hotkey()
            return
        self._write({"type": "key", "key": name, "pressed": True})

    def _on_release(self, key) -> None:
        name = getattr(key, "char", None) or str(key)
        self._write({"type": "key", "key": name, "pressed": False})

    # --- mouse ------------------------------------------------------------
    def _on_click(self, x, y, button, pressed) -> None:
        self._write({"type": "click", "button": str(button), "pressed": pressed})

    def _on_scroll(self, x, y, dx, dy) -> None:
        self._write({"type": "scroll", "dx": dx, "dy": dy})

    def start(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                f"Missing input dependency ({exc.name}). "
                f"Install with: pip install -r recorder/requirements.txt"
            )
        self._kb = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._ms = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
        self._kb.start()
        self._ms.start()

    def stop(self) -> None:
        if self._kb is not None:
            self._kb.stop()
        if self._ms is not None:
            self._ms.stop()
        with self._lock:
            self._file.close()
