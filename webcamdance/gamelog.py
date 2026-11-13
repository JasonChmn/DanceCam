"""Journal de partie JSONL : perfs périodiques + jugements par move + résumé."""
import json
import time
from pathlib import Path


class GameLog:
    def __init__(self, path, **header):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8", buffering=1)  # ligne par ligne : survit à un crash
        self._seen = {}           # joueur -> nb de moves déjà écrits
        self._t_perf = 0.0
        self._cam_fid = None
        self.write("header", **header)

    def write(self, kind, **d):
        self.f.write(json.dumps({"type": kind, "wall": round(time.time(), 3), **d}, ensure_ascii=False) + "\n")

    def perf(self, now, t, tick, worker, cam, video_lag, period=1.0):
        """Toutes les `period` s : fps rendu, latence inférence, fps caméra réel, retard vidéo."""
        if now - self._t_perf < period:
            return
        fid = cam.latest()[0]
        dt = now - self._t_perf if self._t_perf else None
        cam_fps = (fid - self._cam_fid) / dt if dt and self._cam_fid is not None else None
        self._t_perf, self._cam_fid = now, fid
        self.write("perf", t=round(t, 3), render_fps=round(tick.get_fps(), 1), infer_ms=round(worker.infer_ms, 2),
                   cam_fps=None if cam_fps is None else round(cam_fps, 1), video_lag_frames=video_lag,
                   visible=worker.n_visible)

    def track(self, t, poses, thr):
        """Position x (milieu des épaules) de chaque slot : repère les échanges d'identité."""
        xs = [None if kp is None or (kp[5:7, 2] < thr).any() else round(float(kp[5:7, 0].mean()), 1)
              for kp in poses]
        self.write("track", t=round(t, 3), x=xs)

    def moves(self, scores):
        """Écrit les nouveaux jugements de chaque joueur."""
        for p, sc in enumerate(scores):
            k = self._seen.get(p, 0)
            for h in sc.history[k:]:
                self.write("move", player=p, **h)
            self._seen[p] = len(sc.history)

    def close(self, scores):
        self.moves(scores)
        for p, sc in enumerate(scores):
            self.write("result", player=p, total=sc.total, ratio=round(sc.ratio, 4), stars=sc.stars,
                       counts=sc.counts, best_combo=sc.best_combo)
        self.f.close()
