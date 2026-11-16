"""Thread caméra (dernière frame seulement) + thread inférence/tracking joueurs."""
import queue
import sys
import threading
import time
import traceback

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
    """Inférence sur la dernière frame caméra, affectation aux slots, lissage, publication.

    Mode groupe (group=True) : nombre de joueurs inconnu au départ. Le worker publie toutes les
    personnes vues (`raw`) jusqu'à lock_people() (photo bras levés), puis suit ces personnes ;
    keep() réduit ensuite aux joueurs retenus."""

    FREE_MAX = 12      # personnes publiées en mode libre (photo de groupe)

    def __init__(self, cam, pose, cfg, n_players, group=False):
        super().__init__(daemon=True)
        self.cam, self.pose, self.cfg, self.n, self.group = cam, pose, cfg, n_players, group
        self.imgsz = cfg.group_imgsz if group else None
        self._infer(np.zeros((cam.h, cam.w, 3), np.uint8))  # warmup à la résolution caméra (sinon ~1 s sur la 1re frame)
        self.tracker = None if group else PlayerTracker(n_players, cam.w)
        self.filters = [OneEuro() for _ in range(n_players)]
        self.q = queue.SimpleQueue()           # (t_capture, [kp(17,3) | None] * n)
        self.view = (None, [None] * n_players)  # pour l'affichage
        self.raw = None         # mode libre : (frame, t, kpts, boxes, sigs, centers, scales), personnes triées par x
        self.n_visible = 0
        self.infer_ms = 0.0
        self._lock_req = False
        self._cmds = []         # commandes du mode groupe, exécutées dans le thread
        self.errors = 0         # erreurs rattrapées (journalisées) : le suivi continue
        self._last = -1
        self._halt = False

    def _infer(self, frame):
        return self.pose(frame, self.imgsz) if self.imgsz else self.pose(frame)

    def lock(self):
        """Fige les slots (gauche->droite) et mémorise la couleur de chaque joueur à la prochaine
        frame où n joueurs sont visibles et mesurables."""
        self._lock_req = True

    def lock_people(self, raw, idx):
        """Groupe : suit les personnes idx (triées par x) de la photo `raw`."""
        self._cmds.append(("lock", raw, list(idx)))

    def keep(self, slots):
        """Groupe : ne garde que ces slots (ordre = P1, P2…)."""
        self._cmds.append(("keep", list(slots)))

    def unlock(self):
        """Groupe : retour au mode libre (refaire le comptage)."""
        self._cmds.append(("unlock",))

    @property
    def locked(self):
        return self.tracker is not None and self.tracker.ready

    def stop(self):
        self._halt = True

    def _run_cmds(self):
        while self._cmds:
            c = self._cmds.pop(0)
            if c[0] == "lock":
                _, (_, t, _, _, sigs, centers, scales), idx = c
                self.n = len(idx)
                self.tracker = PlayerTracker(max(1, self.n), self.cam.w)
                self.tracker.init(centers[idx], scales[idx], sigs[idx], t)
            elif c[0] == "keep" and self.tracker is not None:
                self.tracker.subset(c[1])
                self.n = len(c[1])
            elif c[0] == "unlock":
                self.tracker, self.n = None, 0
            self.filters = [OneEuro() for _ in range(self.n)]
            self.view = (self.view[0], [None] * self.n)

    def run(self):
        while not self._halt:
            try:
                self._step()
            except Exception:       # une frame en erreur ne doit pas arrêter le suivi (sinon jeu figé)
                self.errors += 1
                if self.errors <= 3:
                    print("[capture] erreur dans le suivi des joueurs :")
                    traceback.print_exc()
                time.sleep(0.05)

    def _step(self):
        thr = self.cfg.kpt_conf
        if self._cmds:
            self._run_cmds()
        fid, frame, t = self.cam.latest()
        if fid == self._last:
            time.sleep(0.001)
            return
        self._last = fid
        n = self.n
        t0 = time.perf_counter()
        kpts, boxes, _ = self._infer(frame)[0]
        self.infer_ms += 0.1 * ((time.perf_counter() - t0) * 1e3 - self.infer_ms)
        if self.group:
            # rangées : les joueurs du fond paraissent petits (jambes cachées) -> seuil de taille bas
            idx = select_people(boxes, self.FREE_MAX if self.tracker is None else n + 4, min_rel_h=0.2)
        else:
            idx = select_people(boxes, n + 1)
        kpts, boxes = kpts[idx], boxes[idx]
        centers, scales = centers_scales(kpts, boxes, thr)
        sigs = signatures(frame, kpts, thr)          # couleur de tenue (~1 ms)
        self.n_visible = min(len(idx), n) if n else len(idx)

        if self.group and self.tracker is None:     # mode libre : tout le monde, gauche->droite
            o = np.argsort(centers[:, 0]) if len(idx) else np.zeros(0, int)
            self.raw = (frame, t, kpts[o], boxes[o], sigs[o], centers[o], scales[o])
            self.view = (frame, list(kpts[o]))
            return

        if not self.tracker.ready:  # calibration : ordre provisoire gauche->droite
            top = np.arange(min(n, len(idx)))
            order = top[np.argsort(centers[top, 0])] if len(top) else top
            poses = [kpts[j] for j in order] + [None] * (n - len(order))
            if self._lock_req and len(order) == n and not np.isnan(sigs[order]).any():
                self.tracker.init(centers[order], scales[order], sigs[order], t)
            self.view = (frame, poses)
            return

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
