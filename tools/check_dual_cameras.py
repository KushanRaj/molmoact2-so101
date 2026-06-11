"""Preview and save paired side/top camera frames without loading MolmoAct2.

Example:
    python tools/check_dual_cameras.py \
        --side-source 0 \
        --top-source 1 \
        --out-dir runs/camera_check \
        --show
"""
import argparse
import os
import time

import cv2

from molmoact_so101.setup.opencv_camera import FLIP_CHOICES, OpenCVCamera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-source", required=True)
    parser.add_argument("--top-source", required=True)
    parser.add_argument("--side-flip", choices=FLIP_CHOICES, default="none")
    parser.add_argument("--top-flip", choices=FLIP_CHOICES, default="none")
    parser.add_argument("--out-dir", default="runs/camera_check")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--save-every", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    side = OpenCVCamera(args.side_source, name="side", flip=args.side_flip)
    top = OpenCVCamera(args.top_source, name="top", flip=args.top_flip)

    start = time.time()
    next_save = start
    saved = 0
    try:
        while time.time() - start < args.seconds:
            side_bgr = side.read()
            top_bgr = top.read()
            if side_bgr is None or top_bgr is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now >= next_save:
                stamp = int(now * 1000)
                cv2.imwrite(os.path.join(args.out_dir, f"{stamp}_side.jpg"), side_bgr)
                cv2.imwrite(os.path.join(args.out_dir, f"{stamp}_top.jpg"), top_bgr)
                print(f"[camera-check] saved pair {saved} at {stamp}")
                saved += 1
                next_save = now + args.save_every

            if args.show:
                cv2.imshow("side", side_bgr)
                cv2.imshow("top", top_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.03)
    finally:
        side.release()
        top.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
