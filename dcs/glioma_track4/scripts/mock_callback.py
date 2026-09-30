#!/usr/bin/env python3
"""本地 mock 回调接收器：代替平台接收推理服务的回调，验证「回调闭环」。

为什么需要它：``06_platform_serve.sh`` 推理完成后会把 ``{request_id, evaluationId,
predPath}`` POST 到 ``CALLBACK_URL``。本地测试容器里没有平台 —— 不设地址则
服务只打日志"未解析到回调地址"，**闭环链路从未被真正验证过**。
把它当 ``CALLBACK_URL`` 指过来，就能完整演练"推理 → 回调 → 平台收到 predPath"。

用法::

    # ① 起接收器（后台）
    nohup python3 scripts/mock_callback.py --port 9000 > logs/mock_callback.log 2>&1 &

    # ② 带着它启动推理服务
    CALLBACK_URL="http://127.0.0.1:9000/api/competition/inference/callback/" \
      bash scripts/06_platform_serve.sh

    # ③ 推理完成后，查看收到的回调
    curl -s http://127.0.0.1:9000/received

输出：每个回调一行 JSON（含 request_id / evaluationId / predPath / 时间），
同时原样保存到 ``$WORKSPACE/logs/mock_callback_received.jsonl``。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_RECEIVED: list[dict] = []
_STORE: Path | None = None


def _store_path() -> Path:
    global _STORE
    if _STORE is None:
        ws = os.environ.get("WORKSPACE") or "/2026aicompetition/workspace"
        d = Path(ws) / "logs"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            d = Path("logs")
            d.mkdir(parents=True, exist_ok=True)
        _STORE = d / "mock_callback_received.jsonl"
    return _STORE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                             # noqa: ANN102
        print(f"[mock-callback] {self.command} {self.path}", flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                             # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {"_raw": raw.decode("utf-8", "replace")}
        record = {"received_at": time.strftime("%H:%M:%S"), "path": self.path, "body": data}
        _RECEIVED.append(record)
        try:
            with _store_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        print(f"[mock-callback] ✓ 收到回调: {json.dumps(data, ensure_ascii=False)}",
              flush=True)
        # 平台应答格式不重要，本地演练回 200 即可
        self._json(200, {"code": 0, "msg": "received", "n": len(_RECEIVED)})

    def do_GET(self):                                              # noqa: N802
        if self.path.rstrip("/") in ("/received", "/received/"):
            self._json(200, {"count": len(_RECEIVED), "callbacks": _RECEIVED})
        elif self.path.rstrip("/") in ("/health", "/health/"):
            self._json(200, {"status": "mock-callback up", "count": len(_RECEIVED)})
        else:
            self._json(404, {"msg": "GET /received 查看收到的回调；GET /health 探活"})


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 mock 回调接收器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    print(f"[mock-callback] 监听 http://{a.host}:{a.port}"
          f"（POST 任意路径=收回调；GET /received=查看）", flush=True)
    print(f"[mock-callback] 回调会同时追加到 {_store_path()}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
