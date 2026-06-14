"""Policy backends for SO-101 runtime.

Every backend exposes the MolmoActPolicy shape:

    predict_chunk(images, state, prompt, num_steps, cuda_graph) -> (T, 6)

`images` are PIL images in [scene, wrist/top] order and `state` is already in
model joint frame. The runtime owns all hardware, safety clamps, camera capture,
and frame conversion.
"""
import base64
import io
import json
import urllib.request

import numpy as np

from .policy import ACTION_FPS


class DummyPolicy:
    """Hardcoded policy for hardware-loop smoke tests."""

    def __init__(
        self,
        *,
        mode: str = "gripper-open-close",
        chunk_len: int = 30,
        amplitude: float = 12.0,
    ):
        self.mode = mode
        self.chunk_len = int(chunk_len)
        self.amplitude = float(amplitude)
        self._call_idx = 0

    def predict_chunk(
        self,
        images,
        state: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 10,
        cuda_graph: bool = False,
    ) -> np.ndarray:
        del images, prompt, num_steps, cuda_graph
        state = np.asarray(state, dtype=np.float32)
        target = state.copy()
        sign = 1.0 if self._call_idx % 2 == 0 else -1.0
        self._call_idx += 1

        if self.mode == "hold":
            pass
        elif self.mode == "gripper-open-close":
            target[5] = 100.0 if state[5] < 50.0 else 0.0
        elif self.mode == "shoulder-pan-wave":
            target[0] = state[0] + sign * self.amplitude
        elif self.mode == "shoulder-lift-wave":
            target[1] = state[1] + sign * self.amplitude
        elif self.mode == "wrist-flex-wave":
            target[3] = state[3] + sign * self.amplitude
        else:
            raise ValueError(f"Unknown dummy mode {self.mode!r}")

        alpha = np.linspace(0.0, 1.0, self.chunk_len, dtype=np.float32)[:, None]
        return state[None, :] + alpha * (target[None, :] - state[None, :])


class RemoteHttpPolicy:
    """HTTP client for running the brain on another machine.

    The server is expected to accept POST JSON at `url`:

        {
          "prompt": str,
          "state": [float x 6],
          "images": [{"format": "jpeg", "data": base64}, ...],
          "num_steps": int
        }

    and return:

        {"actions": [[float x 6], ...]}
    """

    def __init__(self, url: str, *, timeout: float = 30.0, jpeg_quality: int = 90):
        self.url = url
        self.timeout = float(timeout)
        self.jpeg_quality = int(jpeg_quality)
        self.ACTION_FPS = ACTION_FPS

    def predict_chunk(
        self,
        images,
        state: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 10,
        cuda_graph: bool = False,
    ) -> np.ndarray:
        del cuda_graph
        payload = {
            "prompt": prompt,
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "num_steps": int(num_steps),
            "images": [self._encode_image(im) for im in images],
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise RuntimeError(f"Remote actions must have shape (T, 6), got {actions.shape}")
        return actions

    def _encode_image(self, image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.jpeg_quality)
        return {
            "format": "jpeg",
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
