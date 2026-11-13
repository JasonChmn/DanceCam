"""Diagnostic lecture vidéo + audio comme en jeu (SongClock + VideoPlayer + pygame).

    python tools/diag_av.py songs/<nom> [--seconds 30] [--fullscreen]

Mesure : fps de rendu, coût du blit, écart (frame affichée - frame attendue par l'horloge),
dérive horloge perf_counter vs position audio rapportée par pygame.mixer.music.get_pos().
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance import hud  # noqa: E402
from webcamdance.media import SongClock, VideoPlayer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fullscreen", action="store_true")
    a = ap.parse_args()
    song = Path(a.song)
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    flags = pygame.SCALED | (pygame.FULLSCREEN if a.fullscreen else 0)
    screen = pygame.display.set_mode((1280, 720), flags, vsync=1)
    print(f"pygame-ce {pygame.version.ver}, SDL {pygame.version.SDL}, driver {pygame.display.get_driver()}, "
          f"fenêtre {screen.get_size()}, mixer {pygame.mixer.get_init()}")
    W, H = screen.get_size()
    video = VideoPlayer(song / "video.mp4", (W, H)); video.start()
    pygame.mixer.music.load(str(song / "audio.ogg"))
    clock, tick = SongClock(), pygame.time.Clock()
    clock.start()
    vsurf, vid_id = None, -1
    frame_dt, blit_ms, lag, drift, conv_ms = [], [], [], [], []
    last = time.perf_counter()
    running = True
    while running and clock.time() < a.seconds:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        t = clock.time()
        video.set_time(t)
        if video.frame_id != vid_id and video.frame is not None:
            t0 = time.perf_counter()
            vid_id, vsurf = video.frame_id, hud.bgr_surface(video.frame)
            conv_ms.append((time.perf_counter() - t0) * 1e3)
        screen.fill((0, 0, 0))
        t0 = time.perf_counter()
        if vsurf:
            screen.blit(vsurf, vsurf.get_rect(center=(W // 2, H // 2)))
        blit_ms.append((time.perf_counter() - t0) * 1e3)
        hud.text(screen, f"t={t:6.2f}  frame={vid_id}  attendu={int(t * video.fps)}  {tick.get_fps():.0f} fps",
                 18, (255, 255, 0), (10, 10))
        pygame.display.flip()
        tick.tick(60)
        now = time.perf_counter()
        frame_dt.append(now - last); last = now
        if t > 1.0:
            lag.append(int(t * video.fps) - vid_id)                     # frames de retard de l'image affichée
            pos = pygame.mixer.music.get_pos() / 1000                   # s d'audio joué selon SDL
            drift.append((t - pos) * 1e3)
    video.stop(); video.join(timeout=2)
    pygame.quit()
    fd = np.array(frame_dt[10:]) * 1e3
    lag, drift = np.array(lag), np.array(drift)
    print(f"rendu : {1000 / fd.mean():.1f} fps moy, intervalle p95 {np.percentile(fd, 95):.1f} ms, "
          f"max {fd.max():.1f} ms, frames > 25 ms : {(fd > 25).mean() * 100:.1f} %")
    print(f"conversion BGR->Surface : moy {np.mean(conv_ms):.2f} ms | blit : moy {np.mean(blit_ms):.2f} ms")
    print(f"retard image affichée vs horloge (frames vidéo) : moy {lag.mean():.2f} p95 {np.percentile(lag, 95):.0f} "
          f"max {lag.max()} | répartition " + " ".join(f"{k}:{(lag == k).mean() * 100:.0f}%" for k in range(0, 4)))
    n = len(drift)
    print(f"horloge - get_pos (ms) : début {drift[:30].mean():.1f}, fin {drift[-30:].mean():.1f}, "
          f"dérive {drift[-30:].mean() - drift[:30].mean():+.1f} ms sur {a.seconds:.0f} s (n={n})")


if __name__ == "__main__":
    main()
