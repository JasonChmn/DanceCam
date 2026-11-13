"""Diagnostic webcam : résolution/fps réellement obtenus.  python tools/diag_webcam.py [--seconds 10]"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance.capture import Camera  # noqa: E402
from webcamdance.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cfg = Config(cam_index=a.camera)
    t = time.perf_counter()
    cam = Camera(cfg)
    print(f"ouverture : {(time.perf_counter() - t) * 1e3:.0f} ms, backend {cam.cap.getBackendName()}")
    fcc = int(cam.cap.get(cv2.CAP_PROP_FOURCC))
    print(f"demandé {cfg.cam_w}x{cfg.cam_h}@{cfg.cam_fps} MJPG | obtenu {cam.w}x{cam.h} "
          f"fps annoncé {cam.cap.get(cv2.CAP_PROP_FPS):.1f} fourcc {fcc.to_bytes(4, 'little').decode(errors='replace')}")
    cam.start()
    ts, last = [], -1
    t_end = time.perf_counter() + a.seconds
    while time.perf_counter() < t_end:
        fid, f, tc = cam.latest()
        if fid != last:
            ts.append(tc)
            last = fid
        time.sleep(0.001)
    cam.stop()
    cam.join(timeout=2)
    d = np.diff(ts) * 1e3
    print(f"frames {len(ts)} en {a.seconds:.0f}s -> {len(ts) / a.seconds:.1f} fps mesurés")
    print(f"intervalle ms : moy {d.mean():.1f} p50 {np.median(d):.1f} p95 {np.percentile(d, 95):.1f} max {d.max():.1f}")


if __name__ == "__main__":
    main()
