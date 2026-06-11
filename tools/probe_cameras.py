"""Probe OpenCV camera indexes and save one frame from each working device."""
import argparse
import os
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=10)
    parser.add_argument("--out-dir", default="runs/camera_probe")
    parser.add_argument("--warmup-frames", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY

    for idx in range(args.max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        frame = None
        for _ in range(args.warmup_frames):
            ok, maybe = cap.read()
            if ok:
                frame = maybe
            time.sleep(0.03)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frame is None:
            print(f"[probe] index={idx}: opened but produced no frame")
            continue

        path = os.path.join(args.out_dir, f"camera_{idx}.jpg")
        cv2.imwrite(path, frame)
        print(f"[probe] index={idx}: {width:.0f}x{height:.0f}@{fps:.1f} -> {path}")


if __name__ == "__main__":
    main()
