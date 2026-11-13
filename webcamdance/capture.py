"""Thread caméra (dernière frame seulement) + thread inférence/tracking joueurs."""
import queue
import sys
import threading
import time

import cv2
import numpy as np

from .appearance import signatures
from .coach_id import overlap_frac
from .tracking import OneEuro, PlayerTracker, centers_scales, select_people


class Camera(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.cam_index, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # sinon YUY2 bridé à ~10 fps en 720p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
        # 1 seul tampon => V4L2 tombe à 15 fps (mesuré) ; le thread lit en continu, 2 suffit
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError(f"Webcam {cfg.cam_index} inaccessible")
        self.h, self.w = f.shape[:2]
        self._latest = (0, cv2.flip(f, 1), time.perf_counter())
        self._halt = False

    def latest(self):
        return self._latest

    def stop(self):
        self._halt = True

    def run(self):
        fid = 0
        while not self._halt:
            ok, f = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            fid += 1
            # miroir AVANT inférence : géométrie et labels G/D directement comparables au coach
            self._latest = (fid, cv2.flip(f, 1), t)
        self.cap.release()


class PlayerWorker(threading.Thread):
    """Inférence sur la dernière frame caméra, affectation aux slots, lissage, publication."""

    def __init__(self, cam, pose, cfg, n_players):
        super().__init__(daemon=True)
        self.cam, self.pose, self.cfg, self.n = cam, pose, cfg, n_players
        pose(np.zeros((cam.h, cam.w, 3), np.uint8))  # warmup à la résolution caméra (sinon ~1 s sur la 1re frame)
        self.tracker = PlayerTracker(n_players, cam.w)
        self.filters = [OneEuro() for _ in range(n_players)]
        self.q = queue.SimpleQueue()           # (t_capture, [kp(17,3) | None] * n)
        self.view = (None, [None] * n_players)  # pour l'affichage
        self.n_visible = 0
        self.infer_ms = 0.0
        self._lock_req = False
        self._halt = False

    def lock(self):
        """Fige les slots (gauche->droite) et mémorise la couleur de chaque joueur à la prochaine
        frame où n joueurs sont visibles et mesurables."""
        self._lock_req = True

    @property
    def locked(self):
        return self.tracker.ready

    def stop(self):
        self._halt = True

    def run(self):
        last, thr, n = -1, self.cfg.kpt_conf, self.n
        while not self._halt:
            fid, frame, t = self.cam.latest()
            if fid == last:
                time.sleep(0.001)
                continue
            last = fid
            t0 = time.perf_counter()
            kpts, boxes, _ = self.pose(frame)[0]
            self.infer_ms += 0.1 * ((time.perf_counter() - t0) * 1e3 - self.infer_ms)
            idx = select_people(boxes, n + 1)
            kpts, boxes = kpts[idx], boxes[idx]
            centers, scales = centers_scales(kpts, boxes, thr)
            sigs = signatures(frame, kpts, thr)          # couleur de tenue (~1 ms)
            self.n_visible = min(len(idx), n)

            if not self.tracker.ready:  # calibration : ordre provisoire gauche->droite
                top = np.arange(min(n, len(idx)))
                order = top[np.argsort(centers[top, 0])] if len(top) else top
                poses = [kpts[j] for j in order] + [None] * (n - len(order))
                if self._lock_req and len(order) == n and not np.isnan(sigs[order]).any():
                    self.tracker.init(centers[order], scales[order], sigs[order], t)
                self.view = (frame, poses)
                continue

            poses = []
            for i, j in enumerate(self.tracker.update(centers, scales, sigs, t, overlap_frac(boxes))):
                if j is None:
                    if t - self.tracker.seen[i] > 0.5:
                        self.filters[i] = OneEuro()
                    poses.append(None)
                    continue
                kp = kpts[j].copy()
                kp[:, :2] = self.filters[i](kp[:, :2], t, mask=kp[:, 2:3] >= thr)
                poses.append(kp)
            self.view = (frame, poses)
            self.q.put((t, poses))
