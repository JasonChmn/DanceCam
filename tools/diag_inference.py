"""Diagnostic inférence temps réel sur le flux webcam (boucle réelle Camera -> PlayerWorker).

    python tools/diag_inference.py [--players 1] [--seconds 30] [--model yolo26s-pose.pt] [--no-window]

Mesure : latence d'inférence (moy/p95), latence capture->pose publiée, débit, VRAM, personnes vues.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance.capture import Camera, PlayerWorker  # noqa: E402
from webcamdance.config import Config  # noqa: E402
from webcamdance.pose import PoseEstimator  # noqa: E402
from webcamdance.skeleton import EDGES  # noqa: E402

COLORS = [(255, 229, 0), (160, 64, 255), (0, 214, 255), (3, 255, 118)]


class Timed:
    """Enveloppe PoseEstimator : chronomètre chaque appel (synchro GPU incluse via .cpu())."""

    def __init__(self, pose):
        self.pose, self.ms = pose, []

    def __call__(self, f):
        t = time.perf_counter()
        r = self.pose(f)
        self.ms.append((time.perf_counter() - t) * 1e3)
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--model")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()
    import torch
    cfg = Config(cam_index=a.camera)
    if a.model:
        cfg.rt_model = a.model
    cam = Camera(cfg)
    pose = Timed(PoseEstimator(cfg.rt_model, cfg))
    pose.ms.clear()  # sans le warmup
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    worker = PlayerWorker(cam, pose, cfg, a.players)
    cam.start(); worker.start()
    worker.lock()  # verrouille les slots dès que n joueurs sont visibles
    lat, vis, t_end = [], [], time.perf_counter() + a.seconds
    while time.perf_counter() < t_end:
        while not worker.q.empty():
            t_cap, ps = worker.q.get()
            lat.append((time.perf_counter() - t_cap) * 1e3)
            vis.append(sum(p is not None for p in ps))
        if not a.no_window:
            frame, poses = worker.view
            if frame is not None:
                img = frame.copy()
                for i, kp in enumerate(poses):
                    if kp is None:
                        continue
                    for p, q in EDGES:
                        if kp[p, 2] >= cfg.kpt_conf and kp[q, 2] >= cfg.kpt_conf:
                            cv2.line(img, tuple(kp[p, :2].astype(int)), tuple(kp[q, :2].astype(int)), COLORS[i % 4], 4)
                    cv2.putText(img, f"P{i + 1}", tuple(kp[0, :2].astype(int) - [0, 40]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[i % 4], 3)
                left = t_end - time.perf_counter()
                cv2.putText(img, f"{left:4.1f}s  locked={worker.locked}  pose {worker.infer_ms:.1f} ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("diag_inference (Echap = stop)", cv2.resize(img, (960, 540)))
            if cv2.waitKey(1) == 27:
                break
        else:
            time.sleep(0.005)
    worker.stop(); cam.stop()
    worker.join(timeout=2); cam.join(timeout=2)
    cv2.destroyAllWindows()
    ms = np.array(pose.ms)
    dur = a.seconds
    print(f"modèle {cfg.rt_model} imgsz {cfg.imgsz} device {pose.pose.device} prec {pose.pose.prec}")
    print(f"inférences {len(ms)} -> {len(ms) / dur:.1f}/s | inférence ms : moy {ms.mean():.1f} p50 {np.median(ms):.1f} "
          f"p95 {np.percentile(ms, 95):.1f} max {ms.max():.1f}")
    if lat:
        lat = np.array(lat)
        print(f"poses publiées (slots verrouillés) {len(lat)} | capture->publication ms : moy {lat.mean():.1f} "
              f"p95 {np.percentile(lat, 95):.1f}")
        vis = np.array(vis)
        print(f"joueurs suivis par frame : " + " ".join(f"{k}:{(vis == k).mean() * 100:.0f}%" for k in range(a.players + 1)))
    else:
        print("slots jamais verrouillés (personne / pas assez de joueurs visibles)")
    if torch.cuda.is_available():
        print(f"VRAM pic (torch) {torch.cuda.max_memory_allocated() / 1e6:.0f} Mo, réservée {torch.cuda.max_memory_reserved() / 1e6:.0f} Mo")


if __name__ == "__main__":
    main()
