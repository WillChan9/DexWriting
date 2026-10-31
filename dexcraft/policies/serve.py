"""Serve a local Policy over HTTP so the evaluator can drive it remotely:

    python -m dexcraft.policies.serve --policy dexcraft.policies.scripted:make_policy --port 8100
    python -m dexcraft.evaluate --policy-url http://localhost:8100 ...
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def _unpng(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))), dtype=np.uint8)

from .base import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    policy = load_policy(args.policy)

    class H(BaseHTTPRequestHandler):
        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"control_mode": policy.control_mode, "obs_mode": policy.obs_mode})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode()) if n else {}
            if self.path == "/reset":
                policy.reset(payload.get("task_info", {}))
                self._send({"ok": True})
            elif self.path == "/act":
                obs = {"position": np.asarray(payload["position"], dtype=np.float32),
                       "tactile": np.asarray(payload["tactile"], dtype=np.float32),
                       "task": {"vec": np.asarray(payload["task"]["vec"], dtype=np.float32),
                                "map": _unpng(payload["task"]["map"])},
                       "external": {"vec": np.asarray(payload["external"]["vec"], dtype=np.float32),
                                    "map": _unpng(payload["external"]["map"])},
                       "instruction": payload.get("instruction", "")}
                if "images" in payload:
                    from PIL import Image
                    obs["images"] = {k: np.asarray(Image.open(io.BytesIO(base64.b64decode(v))))
                                     for k, v in payload["images"].items()}
                self._send({"action": np.asarray(policy.act(obs), dtype=float).tolist()})
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    print(f"serving {args.policy} on port {args.port}")
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
