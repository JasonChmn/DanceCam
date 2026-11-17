"""Banc de test « soirée » : enchaîne des parties jouées par un bot dans un même processus et mesure
ce qu'on ne voit pas en une partie : rythme image par image (saccades), fuites (RAM, VRAM, threads,
handles), temps d'inférence, pauses du ramasse-miettes. À lancer sur la machine qui joue (3060).

    python tools/qa_soak.py                    # 2 tours, plein écran (~8 min)
    python tools/qa_soak.py --rounds 3 --window 1920x1080

La « webcam » rejoue la vidéo de la chanson (les coachs servent de joueurs) : rien à faire devant
l'écran. Scénarios : solo complet avec pause / reprise / redimensionnement, 2 joueurs avec
« Recommencer », 3 joueurs quittés en calibration, 4 joueurs quittés via le menu pause, groupe à 6.
Rapport : logs/qa_<date>/rapport.json + résumé à l'écran.
"""
import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pygame  # noqa: E402

from webcamdance import library, play  # noqa: E402
from webcamdance.config import Config  # noqa: E402

CUR = {"song": None, "calib": 0}
EV = {"play_t0": None}


def calib_frame(song, n):
    """Première image où n coachs sont visibles en entier (pour que la calibration se verrouille)."""
    k = np.load(song / "poses.npz")["kpts"]
    ok = k[..., 2] >= 0.35
    full = (ok[..., 5] & ok[..., 6] & (ok[..., 15] | ok[..., 16])).sum(1)
    idx = np.flatnonzero(full >= min(n, k.shape[1]))
    return int(idx[0]) if len(idx) else 0


def letterbox(f, W=1280, H=720):
    s = min(W / f.shape[1], H / f.shape[0])
    r = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class BotCamera(threading.Thread):
    """« Webcam » = vidéo de la chanson en cours, calée sur la position audio."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.h, self.w = 720, 1280
        self._latest = (0, np.zeros((720, 1280, 3), np.uint8), time.perf_counter())
        self._halt, self.src, self.cap, self.idx = False, None, None, 0

    def latest(self):
        return self._latest

    def stop(self):
        self._halt = True

    def run(self):
        fid = 0
        while not self._halt:
            time.sleep(1 / 30)
            song = CUR["song"]
            if song is None:
                continue
            if song != self.src:
                if self.cap:
                    self.cap.release()
                self.cap, self.src = cv2.VideoCapture(str(song / "video.mp4")), song
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, CUR["calib"])
                ok, f = self.cap.read()
                self.first = letterbox(f) if ok else np.zeros((720, 1280, 3), np.uint8)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.idx = 0
            pos = pygame.mixer.music.get_pos() / 1000 if EV["play_t0"] else -1
            f = self.first
            if pos >= 0:
                tgt = max(self.idx, int(pos * self.fps))
                while self.idx < tgt and self.cap.grab():
                    self.idx += 1
                ok, g = self.cap.retrieve()
                if ok:
                    f = letterbox(g)
            fid += 1
            self._latest = (fid, f, time.perf_counter())


play.Camera = BotCamera
play.arms_up = lambda kp, thr: True       # groupe : tout le monde « lève les bras »

FLIPS, GC, _gc_t = [], [], {}
_flip = pygame.display.flip


def _flip_rec():
    FLIPS.append(time.perf_counter())
    _flip()


pygame.display.flip = _flip_rec
_start = play.SongClock.start


def _start_rec(self):
    _start(self)
    if EV["play_t0"] is None:
        EV["play_t0"] = time.perf_counter()


play.SongClock.start = _start_rec


def _gc_cb(phase, info):
    if phase == "start":
        _gc_t["t"] = time.perf_counter()
    elif "t" in _gc_t:
        GC.append((info["generation"], (time.perf_counter() - _gc_t["t"]) * 1e3, time.perf_counter()))


gc.callbacks.append(_gc_cb)


def key(k):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))


def wait_play(timeout=60):
    t = time.perf_counter()
    while EV["play_t0"] is None and time.perf_counter() - t < timeout:
        time.sleep(0.05)
    return EV["play_t0"] is not None


def at_play(dt):
    time.sleep(max(0.0, EV["play_t0"] + dt - time.perf_counter()))


def driver(name, meta, size):
    """Joue le scénario en postant des événements clavier / fenêtre."""
    if name == "3P_quitte_calibration":
        time.sleep(3)
        key(pygame.K_ESCAPE)
        return
    if name == "groupe6":
        time.sleep(4); key(pygame.K_RETURN)             # lobby -> comptage (5 s)
        time.sleep(8); key(pygame.K_RETURN)             # photo -> commencer
    if not wait_play():
        print(f"[qa] {name} : la partie n'a jamais démarré (calibration ?)", flush=True)
        key(pygame.K_ESCAPE)
        return
    if name == "solo_complet":
        at_play(15); key(pygame.K_SPACE)                # pause (menu)
        at_play(18); key(pygame.K_RETURN)               # Reprendre -> 3 s de compte à rebours
        if size:
            at_play(26); pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=size[0] * 5 // 6, h=size[1] * 5 // 6))
            at_play(31); pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
        at_play(meta["duration"] + 8); key(pygame.K_RETURN)          # résultats -> highlights
        time.sleep(4); key(pygame.K_ESCAPE)                          # highlights -> menu
    elif name == "2P_recommence":
        at_play(20); key(pygame.K_ESCAPE)               # menu pause
        time.sleep(1); key(pygame.K_DOWN); key(pygame.K_RETURN)               # Recommencer
    elif name == "4P_quitte_pause":
        at_play(30); key(pygame.K_ESCAPE)
        time.sleep(1); key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_RETURN)   # Retour au menu
    else:
        at_play(30); key(pygame.K_ESCAPE)
        time.sleep(1); key(pygame.K_UP); key(pygame.K_RETURN)                 # (haut = dernier) Retour au menu


def frame_stats(t_from):
    ts = np.array([t for t in FLIPS if t >= t_from])
    if len(ts) < 10:
        return {}
    d = np.diff(ts) * 1e3
    return {"fps": round(len(d) / (ts[-1] - ts[0]), 1), "p50_ms": round(float(np.percentile(d, 50)), 1),
            "p99_ms": round(float(np.percentile(d, 99)), 1), "max_ms": round(float(d.max()), 1),
            "saccades_25ms": int((d > 25).sum()), "saccades_50ms": int((d > 50).sum()),
            "pires_a_s": [round(float(ts[i + 1] - EV["play_t0"]), 1) for i in np.argsort(-d)[:3]]}


def log_stats(path):
    """Inférence et caméra d'après le journal de partie."""
    try:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    except OSError:
        return {}
    perf = [r for r in rows if r.get("type") == "perf"]
    inf = [r["infer_ms"] for r in perf if r.get("infer_ms")]
    return {"inference_ms_moy": round(float(np.mean(inf)), 1) if inf else None,
            "inference_ms_max": round(float(np.max(inf)), 1) if inf else None,
            "erreurs_suivi": max((r.get("errors", 0) for r in perf), default=0)}


def snapshot():
    p = psutil.Process()
    out = {"ram_mo": round(p.memory_info().rss / 2**20), "threads": threading.active_count(),
           "handles": p.num_handles() if sys.platform == "win32" else p.num_fds()}
    try:
        import torch
        if torch.cuda.is_available():
            out["vram_mo"] = round(torch.cuda.memory_reserved() / 2**20)
    except ImportError:
        pass
    return out


def pick_songs(root):
    """Une chanson par scénario selon le nombre de coachs (la plus courte, pour aller vite)."""
    songs = sorted(library.list_songs(root), key=lambda s: s["meta"].get("duration", 1e9))
    by = lambda k: next((s["path"] for s in songs if s["meta"].get("n_coaches") == k), None)  # noqa: E731
    many = next((s["path"] for s in songs if s["meta"].get("n_coaches", 1) >= 3), None)
    scen = [("solo_complet", by(1), 1), ("2P_recommence", by(2), 2), ("3P_quitte_calibration", by(3), 3),
            ("4P_quitte_pause", by(4), 4), ("groupe6", many, 6)]
    return [s for s in scen if s[1] is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs-dir", default=str(ROOT / "songs"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--window", help="fenêtré (ex. 1920x1080) ; défaut : plein écran comme en jeu")
    ap.add_argument("--model")
    a = ap.parse_args()
    out = ROOT / "logs" / f"qa_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    scen = pick_songs(a.songs_dir)
    print("[qa] scénarios :", [(n, p.name, k) for n, p, k in scen], flush=True)

    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    pygame.key.set_repeat(300, 40)
    cfg = Config()
    cfg.pause_on_focus_loss = False             # test sans surveillance
    if a.model:
        cfg.rt_model = a.model
    size = tuple(int(v) for v in a.window.lower().split("x")) if a.window else None
    t = time.perf_counter()
    disp = play.Display(size is None, size or (1280, 720), vsync=cfg.vsync)
    sess = play.Session(cfg)
    sess.ensure(disp)
    res = {"demarrage_s": round(time.perf_counter() - t, 2), "vsync": disp.vsync_on,
           "ecran": list(disp.screen.get_size()), "debut": snapshot(), "parties": []}
    try:
        import torch
        res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        pass
    print("[qa] démarrage", res["demarrage_s"], "s", res.get("gpu"), res["debut"], flush=True)
    try:
        for r in range(a.rounds):
            for name, song, n in scen:
                meta = json.loads((song / "meta.json").read_text(encoding="utf-8"))
                CUR["song"], CUR["calib"] = song, calib_frame(song, min(n, 4))
                EV["play_t0"] = None
                FLIPS.clear(); GC.clear()
                threading.Thread(target=driver, args=(name, meta, size), daemon=True).start()
                log_path = out / f"{r}_{name}.jsonl"
                err = None
                try:
                    outcome, scores, _ = play.run_game(disp, sess, song, n, log_path=log_path)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    outcome, scores, err = "erreur", [], repr(e)
                pygame.key.set_repeat(300, 40)
                CUR["song"] = None
                time.sleep(1)
                g = {"tour": r, "scenario": name, "chanson": song.name, "issue": outcome, "erreur": err,
                     "scores": [s.total for s in scores],
                     "gc_max_ms_en_jeu": round(max((x[1] for x in GC if EV["play_t0"] and x[2] > EV["play_t0"]),
                                                   default=0.0), 1),
                     **(frame_stats(EV["play_t0"] + 2) if EV["play_t0"] else {}), **log_stats(log_path),
                     **snapshot()}
                res["parties"].append(g)
                print("[qa]", json.dumps(g, ensure_ascii=False), flush=True)
                (out / "rapport.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        sess.close()
        pygame.quit()
    summary(res, out)


def summary(res, out):
    P = res["parties"]
    if not P:
        return
    ram = [g["ram_mo"] for g in P]
    # fuite = croissance d'un tour à l'autre (le 1er tour initialise tout : modèle à 960, caches…)
    ends = [g["ram_mo"] for i, g in enumerate(P) if i + 1 == len(P) or P[i + 1]["tour"] != g["tour"]]
    per_round = len(P) / max(1, len(ends))
    growth = (ends[-1] - ends[0]) / (per_round * (len(ends) - 1)) if len(ends) > 1 else float("nan")
    played = [g for g in P if "fps" in g]
    print("\n==================== RÉSUMÉ ====================")
    print(f"GPU {res.get('gpu')} · écran {res['ecran']} · vsync {res['vsync']} · démarrage {res['demarrage_s']} s")
    print(f"{'scénario':24s} {'issue':9s} {'fps':>5s} {'p99':>6s} {'max':>6s} {'>25ms':>6s} {'inf.':>6s} {'RAM':>6s}")
    v = lambda g, k: "-" if g.get(k) is None else str(g[k])  # noqa: E731
    for g in P:
        print(f"{g['scenario']:24s} {g['issue']:9s} {v(g, 'fps'):>5} {v(g, 'p99_ms'):>6} {v(g, 'max_ms'):>6} "
              f"{v(g, 'saccades_25ms'):>6} {v(g, 'inference_ms_moy'):>6} {g['ram_mo']:>6}")
    print(f"RAM : {ram[0]} -> {ram[-1]} Mo ; d'un tour à l'autre {growth:+.0f} Mo par partie "
          f"({'OK' if growth < 15 else 'À SURVEILLER'} ; --rounds 2 minimum pour le mesurer)")
    gcm = max(g["gc_max_ms_en_jeu"] for g in P)
    print(f"Pire pause du ramasse-miettes en jeu : {gcm} ms ({'OK' if gcm < 20 else 'saccade possible'})")
    errs = [g for g in P if g["erreur"]]
    print(f"Erreurs : {len(errs)}" + (f" ({', '.join(g['scenario'] for g in errs)})" if errs else ""))
    if played:
        p99 = max(g["p99_ms"] for g in played)
        inf = max((g.get("inference_ms_moy") or 0) for g in played)
        print(f"Pire p99 : {p99} ms ({'fluide' if p99 < 25 else 'saccades'}) · pire inférence moyenne : {inf} ms "
              f"({'OK' if inf < 25 else 'trop lent : baisser group_imgsz ou passer en TensorRT'})")
    print(f"Rapport complet : {out / 'rapport.json'}")


if __name__ == "__main__":
    main()
