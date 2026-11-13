"""Moteur de jeu temps réel (une partie). Le menu (python -m webcamdance) appelle run_game.

    python -m webcamdance.play songs/<nom> --players 2

Touches : Espace = pause, S = squelettes coach, ←/→ = changer de coach (calibration),
F11 = plein écran, Échap = retour / quitter.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import pygame

from . import hud, library, media
from .capture import Camera, PlayerWorker
from .config import Config
from .gamelog import GameLog
from .media import SongClock, VideoPlayer
from .pose import PoseEstimator
from .scoring import CoachTrack, PlayerScore, assign_coaches

CALIB, COUNTDOWN, PLAY, RESULTS = range(4)


def open_display(fullscreen, window):
    """Fenêtre redimensionnable au rendu natif (pas de SCALED : le HUD est dessiné net à la
    taille réelle). Plein écran = résolution du bureau."""
    if fullscreen:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return pygame.display.set_mode(window, pygame.RESIZABLE)


def coach_thumbnails(video_path, coach, thr, height):
    """Vignette de chaque coach : frame la plus visible de ses 10 premières secondes de
    présence, recadrée sur ses keypoints. -> liste de pygame.Surface."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for c in range(coach.n):
        idx = np.flatnonzero(coach.present[:, c])
        img = None
        if len(idx):
            win = idx[idx < idx[0] + int(10 * coach.fps)]
            best = int(win[np.argmax(coach.cover[win, c])])
            cap.set(cv2.CAP_PROP_POS_FRAMES, best)
            ok, f = cap.read()
            if ok:
                kp = coach.kpts[best, c]
                v = kp[:, 2] >= thr
                x0, y0 = kp[v, :2].min(0)
                x1, y1 = kp[v, :2].max(0)
                m = 0.25 * (y1 - y0)
                x0, x1 = int(max(0, x0 - m)), int(min(f.shape[1], x1 + m))
                y0, y1 = int(max(0, y0 - 1.2 * m)), int(min(f.shape[0], y1 + 0.5 * m))
                img = f[y0:y1, x0:x1]
        if img is None or img.size == 0:
            img = np.full((height, height // 2, 3), 60, np.uint8)
        w = max(1, int(img.shape[1] * height / img.shape[0]))
        out.append(hud.bgr_surface(np.ascontiguousarray(cv2.resize(img, (w, height)))))
    cap.release()
    return out


class Display:
    """Fenêtre partagée menu / jeu : rendu natif, redimensionnable, F11 plein écran."""

    def __init__(self, fullscreen=False, window=(1280, 720), caption="WebcamDance"):
        self.fullscreen, self.window = fullscreen, window
        self.screen = open_display(fullscreen, window)
        pygame.display.set_caption(caption)
        hud.set_scale(self.screen.get_height())

    def handle(self, e):
        """Traite redimensionnement / F11. -> True si la taille a changé (refaire la mise en page)."""
        if e.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.window = (max(640, e.w), max(360, e.h))
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_F11:
            self.fullscreen = not self.fullscreen
        else:
            return False
        self.screen = open_display(self.fullscreen, self.window)
        hud.set_scale(self.screen.get_height())
        return True

    def message(self, s):
        W, H = self.screen.get_size()
        self.screen.fill((14, 14, 28))
        hud.text(self.screen, s, 36, (255, 255, 255), (W // 2, H // 2), "center")
        pygame.display.flip()


class Session:
    """Caméra + modèle de pose, chargés une fois et gardés entre les parties."""

    def __init__(self, cfg):
        self.cfg, self.cam, self.pose = cfg, None, None
        media.limit_threads(cfg.game_threads)   # au-delà, plus de CPU (et de chauffe) sans gain de fluidité

    def ensure(self, disp):
        if self.cam is None:
            disp.message("Démarrage de la caméra et du modèle…")
            self.cam = Camera(self.cfg)
            self.cam.start()
            self.pose = PoseEstimator(self.cfg.rt_model, self.cfg)

    def close(self):
        if self.cam is not None:
            self.cam.stop()
            self.cam.join(timeout=2)


def run_game(disp, sess, song, n, log_path=None):
    """Joue une partie. -> ("menu" | "quit", scores, records) ; scores vide si abandon avant la musique."""
    cfg = sess.cfg
    song, n = Path(song), max(1, min(4, n))
    meta = json.loads((song / "meta.json").read_text(encoding="utf-8"))
    pygame.display.set_caption(f"WebcamDance — {meta['title']}")
    sess.ensure(disp)
    disp.message("Chargement de la chanson…")
    coach = CoachTrack(song / "poses.npz", cfg)
    scores, records = [], []        # scores créés au démarrage de la musique (association figée)
    log = GameLog(log_path or Path("logs") / f"{song.name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
                  song=str(song), players=n, cfg=dataclasses.asdict(cfg))
    cam = sess.cam
    worker = PlayerWorker(cam, sess.pose, cfg, n); worker.start()
    W, H = disp.screen.get_size()
    video = VideoPlayer(song / "video.mp4", (W, int(H * 0.65)), band=meta.get("band", (0.0, 1.0)),
                        coach_kpts=coach.kpts, thr=cfg.kpt_conf, dim=0.4); video.start()

    def relayout():
        """Disposition à la résolution native : bandeau coach(s) en haut (65 %), bande joueurs en bas."""
        W, H = disp.screen.get_size()
        top_h = int(H * 0.65)
        video.set_fit((W, top_h))
        return {"W": W, "H": H, "strip": pygame.Rect(0, top_h, W, H - top_h),
                "vrect": pygame.Rect((W - video.size[0]) // 2, (top_h - video.size[1]) // 2, *video.size),
                "thumbs": coach_thumbnails(song / "video.mp4", coach, cfg.kpt_conf, hud.px(150)),
                "row": pygame.Rect(0, hud.px(14), W, hud.px(150)),
                "cam": pygame.Rect(W // 2 - hud.px(320), hud.px(210), hud.px(640), hud.px(360))}

    L = relayout()
    # joueurs -> coachs par rang gauche->droite (formation de départ), ←/→ pour décaler
    fx, _ = coach.formation(video.src_w)
    offset = 0
    cmap = assign_coaches(n, fx, coach.present.mean(0), offset)
    ccolors = hud.coach_colors(coach.n, cmap)
    show_coach_skel = True
    pygame.mixer.music.load(str(song / "audio.ogg"))
    clock, tick = SongClock(), pygame.time.Clock()

    state, stable_since, countdown_t0 = CALIB, None, 0.0
    vsurf, vid_id = None, -1
    outcome = None
    while outcome is None:
        now = time.perf_counter()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                outcome = "quit"
            elif disp.handle(e):
                L = relayout()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                outcome = "menu"
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and state == RESULTS:
                outcome = "menu"
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE and state == PLAY:
                clock.toggle_pause()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                show_coach_skel = not show_coach_skel
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_LEFT, pygame.K_RIGHT) and state in (CALIB, COUNTDOWN):
                offset += 1 if e.key == pygame.K_RIGHT else -1
                cmap = assign_coaches(n, fx, coach.present.mean(0), offset)
                ccolors = hud.coach_colors(coach.n, cmap)
        screen = disp.screen
        frame, poses = worker.view
        t = clock.time()
        W, H, px = L["W"], L["H"], hud.px

        if state == CALIB:
            screen.fill((14, 14, 28))
            hud.coach_thumbs_row(screen, L["thumbs"], fx, ccolors, L["row"])
            hud.webcam(screen, frame, poses, L["cam"], cfg.kpt_conf)
            # corps entier exigé (épaules + au moins une cheville) : en jeu, un membre manquant
            # vient alors d'une occlusion ou d'une détection ratée, qui ne sont pas pénalisées
            full = sum(1 for kp in poses if kp is not None and (kp[5:7, 2] >= cfg.kpt_conf).all()
                       and (kp[15:17, 2] >= cfg.kpt_conf).any())
            ok = full >= n
            stable_since = (stable_since or now) if ok else None
            msg = (f"Placez-vous côte à côte, corps entier visible (pieds compris) — {full}/{n} joueur(s)")
            hud.text(screen, msg, 26, (118, 255, 3) if ok else (255, 255, 255), (W // 2, px(610)), "center")
            if coach.n > 1:
                hud.text(screen, "← / → : changer de coach", 20, (200, 200, 215), (W // 2, px(648)), "center", bold=False)
            if stable_since and now - stable_since > 1.5:
                worker.lock()
            if worker.locked:
                state, countdown_t0 = COUNTDOWN, now

        elif state == COUNTDOWN:
            screen.fill((14, 14, 28))
            hud.coach_thumbs_row(screen, L["thumbs"], fx, ccolors, L["row"])
            hud.webcam(screen, frame, poses, L["cam"], cfg.kpt_conf)
            if coach.n > 1:
                hud.text(screen, "← / → : changer de coach", 20, (200, 200, 215), (W // 2, px(648)), "center", bold=False)
            left = 3.0 - (now - countdown_t0)
            hud.text(screen, str(max(1, int(left) + 1)), 160, (255, 255, 255), (W // 2, H // 2), "center")
            if left <= 0:
                while not worker.q.empty():
                    worker.q.get()
                scores = [PlayerScore(cfg, coach, cmap[p]) for p in range(n)]
                hud.reset_crops()
                log.write("mapping", coach_of_player=cmap)
                clock.start()
                state = PLAY

        else:  # PLAY / RESULTS
            if state == PLAY:
                video.set_time(t)
                paused = clock.paused_at is not None
                while not worker.q.empty():
                    t_cap, ps = worker.q.get()
                    if paused:
                        continue
                    ts = t - (now - t_cap) - cfg.cam_latency  # temps chanson de la capture
                    log.track(ts, ps, cfg.kpt_conf)
                    for p, kp in enumerate(ps):
                        if kp is not None:
                            scores[p].add(ts, kp)
                for sc in scores:
                    sc.close_due(t)
                log.perf(now, t, tick, worker, cam, int(t * video.fps) - vid_id)
                log.moves(scores)
                if t >= meta["duration"] or not pygame.mixer.music.get_busy() and not paused and t > 1.0:
                    for sc in scores:
                        sc.close_due(t + 10)
                    state = RESULTS
                    log.close(scores)
                    records = library.save_result(song, scores, n)
                    print(f"[play] journal : {log.path}")
            if video.frame_id != vid_id and video.frame is not None:
                vid_id, vsurf = video.frame_id, hud.bgr_surface(video.frame)
            screen.fill((0, 0, 0))
            vrect = L["vrect"]
            if vsurf:
                screen.blit(vsurf, vsurf.get_rect(center=vrect.center))
                if 0 <= vid_id < coach.T:
                    hud.coach_overlay(screen, coach, vid_id, vrect, video.scale, video.y0,
                                      cfg.kpt_conf, ccolors, skel=show_coach_skel)
            ci = min(max(vid_id, 0), coach.T - 1)
            waiting = [not coach.present[ci, cmap[p]] for p in range(n)]
            hud.progress(screen, min(1.0, t / meta["duration"]))
            hud.player_strip(screen, L["strip"], frame, poses, scores, t, cfg.kpt_conf, waiting)
            hud.text(screen, f"{tick.get_fps():.0f} fps · pose {worker.infer_ms:.0f} ms", 14,
                     (180, 180, 190), (W - px(10), px(12)), "topright", bold=False)
            if state == RESULTS:
                hud.results(screen, scores, records)

        pygame.display.flip()
        tick.tick(cfg.render_fps)

    pygame.mixer.music.stop()
    if not log.f.closed:  # partie interrompue : on garde ce qui a été joué (scores vides si avant le jeu)
        log.close(scores)
        print(f"[play] journal : {log.path}")
    worker.stop(); video.stop()
    for th in (worker, video):
        th.join(timeout=2)
    return outcome, scores, records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", help="dossier produit par import_song")
    ap.add_argument("--players", type=int, default=1)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--latency-ms", type=float)
    ap.add_argument("--fullscreen", action="store_true", help="plein écran à la résolution native")
    ap.add_argument("--window", default="1280x720", help="taille de la fenêtre (ex. 1920x1080)")
    ap.add_argument("--fps", type=int, help="fps de rendu max (défaut 30 ; 60 = animations plus fluides)")
    ap.add_argument("--log", help="journal JSONL (défaut : logs/<chanson>_<date>.jsonl)")
    a = ap.parse_args()
    cfg = config_from_args(a)
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    disp = Display(a.fullscreen, tuple(int(v) for v in a.window.lower().split("x")))
    sess = Session(cfg)
    try:
        run_game(disp, sess, a.song, a.players, a.log)
    finally:
        sess.close()
        pygame.quit()


def config_from_args(a):
    cfg = Config(cam_index=a.camera)
    if a.model:
        cfg.rt_model = a.model
    if a.fps:
        cfg.render_fps = a.fps
    if a.latency_ms is not None:
        cfg.cam_latency = a.latency_ms / 1000
    return cfg


if __name__ == "__main__":
    main()
