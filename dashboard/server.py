"""Tiny stdlib web dashboard for the BladeVision self-play trainer.

Serves a single self-contained page plus a JSON endpoint that reads the trainer's
`progress.jsonl` / `status.json` from a run directory. No external dependencies.

Usage:
    python -m dashboard.server --runs runs/selfplay --port 8765
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "index.html")


def _read_progress(runs_dir: str) -> dict:
    """Collect the history (progress.jsonl) and latest status (status.json) for the UI."""
    history = []
    progress_path = os.path.join(runs_dir, "progress.jsonl")
    if os.path.exists(progress_path):
        with open(progress_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a half-written trailing line

    status = None
    status_path = os.path.join(runs_dir, "status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path) as fh:
                status = json.load(fh)
        except json.JSONDecodeError:
            status = history[-1] if history else None

    last_update = os.path.getmtime(progress_path) if os.path.exists(progress_path) else None
    return {"history": history, "status": status, "last_update": last_update, "runs_dir": runs_dir}


def make_handler(runs_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.startswith("/api/progress"):
                payload = json.dumps(_read_progress(runs_dir)).encode("utf-8")
                self._send(200, payload, "application/json")
                return
            if self.path in ("/", "/index.html"):
                try:
                    with open(_INDEX, "rb") as fh:
                        self._send(200, fh.read(), "text/html; charset=utf-8")
                except FileNotFoundError:
                    self._send(500, b"index.html missing", "text/plain")
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, *_args):  # silence per-request stderr spam
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="BladeVision training dashboard")
    parser.add_argument("--runs", default="runs/selfplay", help="run directory to read")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runs_dir))
    print(f"BladeVision dashboard: http://{args.host}:{args.port}  (reading {runs_dir})")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        server.server_close()


if __name__ == "__main__":
    main()
