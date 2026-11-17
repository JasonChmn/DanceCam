"""Highlights de fin de partie : la caméra est enregistrée pendant la partie (fichier temporaire),
puis on montre le meilleur et le pire passage (fenêtre glissante sur les notes des moves), avec la
musique. Chaque passage, ou la partie entière, peut être gardé en MP4 compact (caméra brute + musique)
dans clips/. La vidéo temporaire est effacée à la fin de la partie dans tous les cas."""
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from . import hud
from .media import ffmpeg_exe

REC_W = 960                 # largeur de l'enregistrement (16:9 -> 960x540)
SHOW, FADE = 0.6, 0.3       # durée d'affichage d'un jugement incrusté, puis fondu (s)


def tmp_dir(song):
    return Path(song).parent.parent / "clips" / ".tmp"


def clean_tmp(song):
    """Restes d'une partie précédente (plantage, fermeture brutale) : effacés."""
    for f in tmp_dir(song).glob("*"):
        try:
            f.unlink()
        except OSError:
            pass


class Recorder(threading.Thread):
    """Enregistre chaque nouvelle image caméra (960 px de large, MPEG-4) dans un fichier temporaire.
    Lit cam.latest() en continu : ne touche pas au thread caméra ni à la boucle de jeu."""

    def __init__(self, cam, song):
        super().__init__(daemon=True, name="recorder")
        self.cam = cam
        d = tmp_dir(song)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"partie_{os.getpid()}_{int(time.time())}.mp4"
        self.size = (REC_W, int(round(REC_W * cam.h / cam.w / 2)) * 2)
        self.scale = REC_W / cam.w          # px caméra -> px enregistrement
        self.tcap = []                      # horodatage de capture de chaque image écrite
        self.marks = ([], [])               # (perf_counter, temps chanson) relevés par la boucle de jeu
        self.heads = []                     # (temps chanson, [(x, y) haut de tête | None] par joueur)
        self.savers = []                    # threads d'export en cours
        self.readers = []                   # lecteurs ouverts sur le fichier (fermés avant de l'effacer)
        self._halt = False
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30, self.size)
        if not self._writer.isOpened():
            raise RuntimeError(f"écriture vidéo impossible ({self.path})")

    def run(self):
        last = None
        try:
            while not self._halt:
                fid, f, t = self.cam.latest()
                if fid == last or f is None:
                    time.sleep(0.004)
                    continue
                last = fid
                self._writer.write(cv2.resize(f, self.size, interpolation=cv2.INTER_AREA))
                self.tcap.append(t)
        except Exception as e:      # disque plein… : pas de highlights, la partie continue
            print(f"[highlights] enregistrement interrompu : {e}")
        finally:
            self._writer.release()

    def stop(self):
        self._halt = True

    def mark(self, now, t):
        self.marks[0].append(now)
        self.marks[1].append(t)

    def add_heads(self, ts, poses, thr):
        hs = []
        for kp in poses:
            ok = None if kp is None else kp[:, 2] >= thr
            hs.append(None if ok is None or not ok.any() else
                      (float(np.median(kp[ok, 0])) * self.scale, float(kp[ok, 1].min()) * self.scale))
        self.heads.append((ts, hs))

    def discard(self, disp=None):
        """Attend les enregistrements en cours puis efface la vidéo temporaire."""
        self.stop()
        if self.is_alive():
            self.join(timeout=3)
        if any(s.is_alive() for s in self.savers) and disp is not None:
            disp.message("Enregistrement de la vidéo…")
        while any(s.is_alive() for s in self.savers):
            pygame.event.pump()             # fenêtre toujours réactive (sinon « ne répond pas »)
            time.sleep(0.05)
        for r in self.readers:              # Windows : un fichier ouvert ne peut pas être effacé
            r.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError as e:                # effacé au début de la prochaine partie
            print(f"[highlights] vidéo temporaire non effacée : {e}")


def _label(cfg, s):
    return next(j[1] for j in cfg.judgements if s >= j[0])


def pick_windows(scores, cfg, seconds, t_min, t_max):
    """Fenêtre glissante (pas = 1 move) sur la note moyenne des joueurs à chaque move.
    -> [meilleure, pire] (dicts a, b, mean, label ; la pire peut manquer) ou [] si trop peu de données."""
    p = cfg.move_period
    n = len(scores)
    M = max((h["move"] for sc in scores for h in sc.history), default=-1) + 1
    vals = np.full((M, n), np.nan)
    for i, sc in enumerate(scores):
        for h in sc.history:
            if h["score"] is not None:
                vals[h["move"], i] = h["score"]
    cnt = np.isfinite(vals).sum(1)
    # move retenu si au moins la moitié des joueurs y ont été jugés (sinon : quelqu'un hors champ)
    team = np.where(cnt >= max(1, (n + 1) // 2), np.nansum(vals, 1) / np.maximum(cnt, 1), np.nan)
    k = max(1, int(round(seconds / p)))
    wins = []
    for w in range(0, M - k + 1):
        seg = team[w:w + k]
        a, b = w * p, (w + k) * p
        if np.isfinite(seg).mean() >= 0.7 and a >= t_min and b <= t_max:
            wins.append((float(np.nanmean(seg)), w))
    if not wins:
        return []
    best = max(wins)
    out = [best]
    rest = [x for x in wins if abs(x[1] - best[1]) >= k]
    if rest:
        out.append(min(rest))
    res = []
    for m, w in out:
        res.append({"a": w * p, "b": (w + k) * p, "mean": m, "label": _label(cfg, m)})
    return res


class Highlights:
    """Après la partie : passages retenus, lecture synchronisée avec la musique, export MP4."""

    def __init__(self, rec, scores, cfg, song, audio, title):
        self.rec, self.cfg, self.song, self.audio, self.title = rec, cfg, Path(song), Path(audio), title
        n = len(rec.tcap)
        if n < 30 or len(rec.marks[0]) < 2:
            self.clips, self.ok = [], False
            return
        perf, songt = np.array(rec.marks[0]), np.array(rec.marks[1])
        # temps chanson de chaque image enregistrée (pause : temps figé, images ignorées à la lecture)
        st = np.interp(np.array(rec.tcap[:n]), perf, songt) - cfg.cam_latency
        self.st = np.maximum.accumulate(st)
        self.t0, self.t1 = float(max(0.0, self.st[0])), float(self.st[-1])
        self.clips = pick_windows(scores, cfg, cfg.highlight_seconds, self.t0, self.t1)
        self.ok = bool(self.clips)
        self.size = rec.size
        # jugements par joueur : (instant d'affichage = fin du move, libellé)
        p = cfg.move_period
        self.judg = [[((h["move"] + 1) * p, h["label"]) for h in sc.history if h["label"] != "SKIP"]
                     for sc in scores]
        self.judg_t = [np.array([t for t, _ in j]) for j in self.judg]
        self.head_t = np.array([t for t, _ in rec.heads]) if rec.heads else np.zeros(0)
        self.saved = {}         # clé -> "saving" | "saved" | "error"
        for c in self.clips:
            c["poster"] = self._frame_at(c["a"] + 0.5)

    # ---------------------------------------------------------------- images
    def index(self, s):
        """Image enregistrée correspondant au temps chanson s (la dernière capturée avant s)."""
        return int(np.clip(np.searchsorted(self.st, s, side="right") - 1, 0, len(self.st) - 1))

    def _frame_at(self, s):
        cap = cv2.VideoCapture(str(self.rec.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.index(s))
        ok, f = cap.read()
        cap.release()
        return f if ok else None

    def reader(self, a):
        r = _SeqReader(self.rec.path, self.index(a))
        self.rec.readers.append(r)
        return r

    # ---------------------------------------------------------------- incrustations
    def overlays(self, surf, s, rect):
        """Jugements récents de chaque joueur, petits, au-dessus de sa tête, sans se chevaucher."""
        if not len(self.head_t):
            return
        i = int(np.clip(np.searchsorted(self.head_t, s), 0, len(self.head_t) - 1))
        if abs(self.head_t[i] - s) > 0.4:
            return
        heads = self.rec.heads[i][1]
        k = rect.w / self.size[0]
        placed = []
        for p, (jt, jl) in enumerate(zip(self.judg_t, self.judg)):
            j = int(np.searchsorted(jt, s, side="right")) - 1
            if j < 0 or p >= len(heads) or heads[p] is None:
                continue
            age = s - jt[j]
            if age > SHOW + FADE:
                continue
            alpha = 255 if age < SHOW else int(255 * (1 - (age - SHOW) / FADE))
            label = jl[j][1]
            img = hud.tag_surface(f"P{p + 1} {label}", 15, hud.JUDGE_COL.get(label, (255, 255, 255)))
            x, y = rect.x + heads[p][0] * k, rect.y + heads[p][1] * k
            r = img.get_rect(midbottom=(int(x), int(y - hud.px(4))))
            if r.top < rect.top:                   # tête en haut de l'image : à côté, pas sur le visage
                r = img.get_rect(midleft=(int(x + hud.px(30)), int(y + r.h)))
            for _ in range(len(placed) + 1):       # chevauchement : on remonte d'un cran
                hit = next((q for q in placed if r.colliderect(q)), None)
                if hit is None:
                    break
                r.bottom = hit.top - 1
            r.clamp_ip(rect)
            placed.append(r)
            if alpha < 255:
                img = img.copy()
                img.set_alpha(alpha)
            surf.blit(img, r)

    # ---------------------------------------------------------------- export
    def save(self, key, a, b):
        """Encode [a, b] (temps chanson) en MP4 H.264 + musique, en arrière-plan."""
        out_dir = self.song.parent.parent / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = {"best": "meilleur", "worst": "pire", "full": "partie"}[key]
        out = out_dir / f"{self.song.name}_{time.strftime('%Y-%m-%d_%Hh%M-%S')}_{name}.mp4"
        self.saved[key] = "saving"
        th = threading.Thread(target=self._encode, args=(key, a, b, out), daemon=True)
        self.rec.savers.append(th)
        th.start()

    def _encode(self, key, a, b, out):
        part = out.with_suffix(".part.mp4")
        w, h = self.size
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30", "-i", "-",
               "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(self.audio),
               "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
               str(part)]
        proc = rd = None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            rd = _SeqReader(self.rec.path, self.index(a))
            for s in np.arange(a, b, 1 / 30):
                f = rd.get(self.index(s))
                if f is not None:
                    proc.stdin.write(f.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg a échoué ({proc.returncode})")
            os.replace(part, out)
            self.saved[key] = "saved"
            print(f"[highlights] vidéo enregistrée : {out}")
        except Exception as e:
            self.saved[key] = "error"
            print(f"[highlights] enregistrement impossible : {e}")
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            if rd is not None:
                rd.close()


class _SeqReader:
    """Lecture séquentielle de la vidéo temporaire (saut en avant par grab, jamais en arrière)."""

    def __init__(self, path, start):
        self.cap = cv2.VideoCapture(str(path))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.pos, self.frame = start - 1, None

    def get(self, idx):
        while self.pos < idx:
            if self.pos < idx - 1:
                ok = self.cap.grab()
                self.pos += 1
                if not ok:
                    break
                continue
            ok, f = self.cap.read()
            self.pos += 1
            if ok:
                self.frame = f
            break
        return self.frame

    def close(self):
        self.cap.release()


# -------------------------------------------------------------------- écran
def layout(W, H, aspect):
    """Deux fenêtres côte à côte + boutons. -> (panes, save_btns, full_btn)."""
    px = hud.px
    gap = px(30)
    pw = (W - 3 * gap) // 2
    ph = int(pw / aspect)
    max_h = H - px(120) - px(250)
    if ph > max_h:
        ph = max_h
        pw = int(ph * aspect)
    x0 = (W - 2 * pw - gap) // 2
    top = px(120)
    panes = [pygame.Rect(x0 + i * (pw + gap), top, pw, ph) for i in range(2)]
    save = [pygame.Rect(r.centerx - px(130), r.bottom + px(52), px(260), px(46)) for r in panes]
    full = pygame.Rect(W // 2 - px(210), save[0].bottom + px(28), px(420), px(50))
    return panes, save, full


SAVE_TXT = {None: "Enregistrer", "saving": "Enregistrement…", "saved": "Enregistré !", "error": "Échec"}
