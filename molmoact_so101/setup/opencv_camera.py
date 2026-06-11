"""Generic OpenCV camera source for non-RealSense RGB inputs.

This is meant for laptop webcams, phone-as-webcam devices, USB webcams, or
URL streams that OpenCV can open. It exposes both `read()` and
`get_latest_color()` so the existing runtime can use it as either the top/wrist
camera or the side/scene camera.
"""
import sys
import threading
import time

import cv2


FLIP_CHOICES = ["v", "h", "180", "none"]
_FLIP_CODES = {"v": 0, "h": 1, "180": -1}


def _parse_source(source):
    try:
        return int(source)
    except (TypeError, ValueError):
        return source


def _default_backend():
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class OpenCVCamera:
    """Continuously captures latest BGR frame from any OpenCV camera source."""

    def __init__(
        self,
        source,
        *,
        name="camera",
        width=640,
        height=480,
        fps=30,
        flip="none",
        backend=None,
    ):
        self.source = _parse_source(source)
        self.name = name
        self.flip_code = _FLIP_CODES.get(flip)
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()

        cap_backend = _default_backend() if backend is None else backend
        self._cap = cv2.VideoCapture(self.source, cap_backend)
        if not self._cap.isOpened() and cap_backend != cv2.CAP_ANY:
            self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open {name} camera source {source!r}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[OpenCVCamera] {name}: source={source!r} target={width}x{height}@{fps}")

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_latest_color(self):
        return self.read()

    def update_observations(self, *, wrist_bgr=None, scene_bgr=None):
        return None

    def close(self):
        self.release()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        print(f"[OpenCVCamera] {self.name}: released")
