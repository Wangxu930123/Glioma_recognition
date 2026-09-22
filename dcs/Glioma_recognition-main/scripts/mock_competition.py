from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nibabel as nib
import numpy as np


class CallbackHandler(BaseHTTPRequestHandler):
    event = threading.Event()
    payload: dict[str, object] | None = None

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        type(self).payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()
        type(self).event.set()

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete mock competition")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--workspace", default=None,
        help="复用指定 workspace（默认用一次性临时目录）。"
             "用于**带真实权重**的完整演练：workspace 下需有规范 §5.2 的 "
             "checkpoint/<goal>/…；否则服务端只在临时 workspace 里找权重，"
             "必然 FileNotFoundError。",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        _make_dataset(dataset)
        callback_server = ThreadingHTTPServer(("127.0.0.1", 0), CallbackHandler)
        callback_thread = threading.Thread(
            target=callback_server.serve_forever,
            daemon=True,
        )
        callback_thread.start()
        port = _free_port()
        env = os.environ.copy()
        ws = (Path(args.workspace).expanduser().resolve() if args.workspace
              else root / "workspace")
        env.update(
            {
                "COMPETITION_WORKSPACE": str(ws),
                "COMPETITION_CALLBACK_URL": (
                    f"http://127.0.0.1:{callback_server.server_port}/callback"
                ),
            }
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
        )
        try:
            _wait_for_health(port, args.timeout)
            print("PASS Health")
            started = time.perf_counter()
            response = _post_json(
                f"http://127.0.0.1:{port}/call",
                {
                    "request_id": "mock-request",
                    "team_id": "mock-team",
                    "track_code": "mock-track",
                    "input": {
                        "evaluation_id": "mock-evaluation",
                        "dataset_path": str(dataset),
                    },
                },
            )
            elapsed = time.perf_counter() - started
            assert response["status"] == "accepted"
            assert elapsed < 5.0
            print("PASS Call response time")
            if not CallbackHandler.event.wait(args.timeout):
                raise TimeoutError("callback was not received")
            payload = CallbackHandler.payload or {}
            output = Path(str(payload["predPath"]))
            assert output.is_dir()
            assert (output / "duplicate_pairs.jsonl").is_file()
            assert len(list(output.glob("*/prediction.json"))) == 2
            assert len(list(output.glob("*/*/*.nii.gz"))) == 4
            print("PASS Background execution")
            print("PASS Output and NIfTI validation")
            print("PASS Callback")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            callback_server.shutdown()
            callback_server.server_close()


def _make_dataset(root: Path) -> None:
    for accession in ("ACC001", "ACC002"):
        for uid in ("T1CE", "FLAIR"):
            directory = root / accession / uid
            directory.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)),
                str(directory / f"{uid}.nii.gz"),
            )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("service did not become healthy")


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    main()

