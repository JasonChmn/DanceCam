"""Test de bout en bout sans danseur : la « webcam » renvoie la vidéo du coach (miroir brut,
comme une vraie caméra filmant quelqu'un qui imite le coach), calée sur la position audio.

    python tools/bot_play.py songs/<nom> [--delay-ms 0] [--still] [args de play.py...]

--delay-ms : retard simulé du « joueur » ; --still : joueur immobile (doit scorer bas).
Un bon joueur parfait doit obtenir ~100 % de PERFECT ; un immobile, quasi que des MISS.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance import play  # noqa: E402

bot = argparse.ArgumentParser(add_help=False)
bot.add_argument("--delay-ms", type=float, default=0.0)
bot.add_argument("--still", action="store_true")
bot.add_argument("--cam-song", help="la « webcam » joue la vidéo de cette chanson (danse fausse)")
bot.add_argument("--isolate", action="store_true", help="ne garder que les coachs suivis (1 personne par joueur)")
bot.add_argument("--calib-at", type=float, default=0.0, help="image (s) montrée pendant la calibration")
bot.add_argument("--seconds", type=float, help="quitte le jeu après N s (Échap simulé)")
bot.add_argument("--shots", nargs="*", type=float, default=[], help="captures d'écran à ces instants (s)")
bot.add_argument("--shot-dir", default="logs")
bot.add_argument("--resize-at", nargs=3, metavar=("T", "W", "H"), help="simule un redimensionnement de fenêtre")
B, rest = bot.parse_known_args()
SONG = Path(rest[0])


def _isolator(cfg):
    """-> f(frame, idx) qui noircit tout sauf les coachs associés aux joueurs (même règle
    d'association que le jeu), pour simuler n vraies personnes devant la caméra."""
    import numpy as np
    from webcamdance.media import spotlight
    from webcamdance.scoring import CoachTrack, assign_coaches
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--players", type=int, default=1)
    n = ap.parse_known_args(rest)[0].players
    src = Path(B.cam_song or SONG)                  # poses de la vidéo réellement montrée
    coach = CoachTrack(src / "poses.npz", cfg)
    w = cv2.VideoCapture(str(src / "video.mp4")).get(cv2.CAP_PROP_FRAME_WIDTH)
    keep = sorted(set(assign_coaches(n, coach.formation(w)[0], coach.present.mean(0))))

    def f(frame, idx):
        idx = min(max(idx, 0), coach.T - 1)
        return spotlight(frame.copy(), coach.kpts[idx][keep], 1.0, 0, cfg.kpt_conf, 0.0)
    return f


class BotCamera(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.iso = _isolator(cfg) if B.isolate else None
        self.cap = cv2.VideoCapture(str(Path(B.cam_song or SONG) / "video.mp4"))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(B.calib_at * self.fps))
        ok, f = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.h, self.w = f.shape[:2]
        self.first = self.iso(f, int(B.calib_at * self.fps)) if self.iso else f
        self.idx = 0
        self._latest = (0, f, time.perf_counter())  # après flip Camera == image coach
        self._halt = False

    def latest(self):
        return self._latest

    def stop(self):
        self._halt = True

    def run(self):
        fid = 0
        while not self._halt:
            time.sleep(1 / 30)
            t = time.perf_counter()
            pos = pygame.mixer.music.get_pos() / 1000 if pygame.mixer.get_init() else -1
            f = self.first
            if pos >= 0 and not B.still:
                tgt = int((pos - B.delay_ms / 1000) * self.fps)
                if tgt < self.idx:
                    tgt = self.idx
                while self.idx < tgt and self.cap.grab():
                    self.idx += 1
                ok, f = self.cap.retrieve()
                if not ok:
                    continue
                if self.iso:
                    f = self.iso(f, self.idx)
            fid += 1
            # la vraie Camera publie cv2.flip(brut) ; brut = flip(coach) => publié = coach
            self._latest = (fid, f, t)


play.Camera = BotCamera
_t0, _flip, _todo = time.perf_counter(), pygame.display.flip, sorted(B.shots)


def _flip_shot():
    if _todo and time.perf_counter() - _t0 >= _todo[0]:
        out = Path(B.shot_dir) / f"shot_{_todo.pop(0):05.1f}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(pygame.display.get_surface(), str(out))
    _flip()


pygame.display.flip = _flip_shot
if B.resize_at:
    def _resize():
        time.sleep(float(B.resize_at[0]))
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=int(B.resize_at[1]), h=int(B.resize_at[2])))
    threading.Thread(target=_resize, daemon=True).start()
if B.seconds:
    def _quit():
        time.sleep(B.seconds)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    threading.Thread(target=_quit, daemon=True).start()
sys.argv = ["play"] + rest
play.main()
