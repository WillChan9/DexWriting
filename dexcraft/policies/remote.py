"""Remote policy: the environment talks to a policy server over HTTP/JSON.

Protocol (also implemented by serve.py, which wraps any local Policy):
    POST /reset   {"task_info": {...}}                      -> {"ok": true}
    POST /act     {"instruction": "...",
                   "position": [75 floats], "tactile": [34 floats],
                   "task":     {"vec": [10 floats], "map": base64 PNG 400x300},
                   "external": {"vec": [41 floats], "map": base64 PNG 400x300},
                   "images": {cam: base64 PNG} (optional)}  -> {"action": [34 floats] or [[34], ...]}
    GET  /info                                             -> {"control_mode": "...", "obs_mode": "..."}
"""
import base64
import io
import json
import urllib.request

import numpy as np

from .base import Policy


def _png(m):
    """uint8 map -> base64 PNG; a sparse map compresses to a few kB."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(m, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


class RemotePolicy(Policy):
    def __init__(self, base_url):
        self.url = base_url.rstrip("/")
        with urllib.request.urlopen(self.url + "/info", timeout=10) as r:
            info = json.loads(r.read().decode())
        self.control_mode = info.get("control_mode", "delta_pos")
        self.obs_mode = info.get("obs_mode", "state")

    def reset(self, task_info):
        _post(self.url + "/reset", {"task_info": task_info})

    def act(self, obs):
        payload = {"instruction": obs["instruction"],
                   "position": np.asarray(obs["position"]).tolist(),
                   "tactile": np.asarray(obs["tactile"]).tolist(),
                   "task": {"vec": np.asarray(obs["task"]["vec"]).tolist(),
                            "map": _png(obs["task"]["map"])},
                   "external": {"vec": np.asarray(obs["external"]["vec"]).tolist(),
                                "map": _png(obs["external"]["map"])}}
        if "images" in obs:
            from PIL import Image
            enc = {}
            for k, im in obs["images"].items():
                buf = io.BytesIO()
                Image.fromarray(im).save(buf, format="PNG")
                enc[k] = base64.b64encode(buf.getvalue()).decode()
            payload["images"] = enc
        return np.asarray(_post(self.url + "/act", payload)["action"], dtype=np.float32)
