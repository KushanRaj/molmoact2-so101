"""Serve a MolmoAct-compatible policy over HTTP.

This is intentionally small and dependency-light so it can run on EC2 without
introducing a web framework. The robot laptop calls this server with current
camera frames, joint state in model frame, and prompt; the server returns an
action chunk.

Example:
    PYTHONPATH=. python tools/policy_server.py \
        --backend molmo --host 0.0.0.0 --port 8008 \
        --device cuda --dtype bfloat16
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from molmoact_so101.model.policy import DTYPES, MolmoActPolicy, REPO_ID
from molmoact_so101.model.policy_api import DummyPolicy


def parse_args():
    p = argparse.ArgumentParser(description="MolmoAct policy HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--backend", default="molmo", choices=["molmo", "dummy"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--dummy-mode", default="gripper-open-close")
    p.add_argument("--dummy-chunk-len", type=int, default=30)
    p.add_argument("--dummy-amplitude", type=float, default=12.0)
    return p.parse_args()


def build_policy(args):
    if args.backend == "dummy":
        return DummyPolicy(
            mode=args.dummy_mode,
            chunk_len=args.dummy_chunk_len,
            amplitude=args.dummy_amplitude,
        )
    return MolmoActPolicy.from_pretrained(
        REPO_ID,
        dtype=args.dtype,
        device=args.device,
        apply_patches=False,
    )


def decode_image(item):
    raw = base64.b64decode(item["data"])
    return Image.open(io.BytesIO(raw)).convert("RGB")


class Handler(BaseHTTPRequestHandler):
    policy = None

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/predict_chunk":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            images = [decode_image(x) for x in req["images"]]
            state = np.asarray(req["state"], dtype=np.float32)
            actions = self.policy.predict_chunk(
                images=images,
                state=state,
                prompt=req["prompt"],
                num_steps=int(req.get("num_steps", 10)),
                cuda_graph=False,
            )
            self._send_json({"actions": np.asarray(actions, dtype=np.float32).tolist()})
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": type(e).__name__,
                "message": str(e),
            }).encode("utf-8"))

    def log_message(self, fmt, *args):
        print(f"[policy-server] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    args = parse_args()
    Handler.policy = build_policy(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[policy-server] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[policy-server] interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
