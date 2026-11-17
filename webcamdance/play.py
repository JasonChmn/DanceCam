"""Moteur de jeu temps réel (une partie). Le menu (python -m webcamdance) appelle run_game.

    python -m webcamdance.play songs/<nom> --players 2

Touches : Espace ou Échap = pause (Reprendre · Recommencer · Retour au menu), S = squelettes coach,
P = pictos des mouvements à venir, ←/→ = changer de coach (calibration), F11 = plein écran.
Hors partie (calibration, résultats), Échap = retour au menu. Fenêtre quittée (Alt+Tab) = pause.
Résultats : Entrée = highlights (meilleur et pire passage, enregistrables en MP4 dans clips/).

Mode groupe (--players 5 ou plus, jusqu'à 8) : comptage bras levés (5 s), photo où chacun reçoit son
P1…Pn et son coach (clic : changer de coach, clic droit : retirer/ajouter, T : tous sur le même
coach, R : refaire), puis jeu collaboratif (jauge et combos d'équipe).
"""
import argparse
import dataclasses
import gc
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from . import highlights, hud, library, media
from .capture import Camera, PlayerWorker
from .config import Config
from .gamelog import GameLog
from .media import SongClock, VideoPlayer
from .pictos import Pictos
from .pose import PoseEstimator
from .scoring import CoachTrack, PlayerScore, TeamScore, assign_coaches, fun_titles, group_coaches, solo_coach
from .tracking import arms_up

CALIB, COUNTDOWN, PLAY, RESULTS, LOBBY, SNAP, PHOTO, HILITE = range(8)   # LOBBY/SNAP/PHOTO : mode groupe


def open_display(fullscreen, window, vsync=True):
    """Fenêtre redimensionnable au rendu natif (pas de SCALED : le HUD est dessiné net à la
    taille réelle). Plein écran = résolution du bureau. vsync : flip() attend le rafraîchissement
    de l'écran -> chaque image reste affichée la même durée (sinon saccades irrégulières)."""
    size, flags = ((0, 0), pygame.FULLSCREEN) if fullscreen else (window, pygame.RESIZABLE)
    if vsync:
        try:
            return pygame.display.set_mode(size, flags, vsync=1)
        except pygame.error:            # pilote sans vsync : affichage classique
            pass
    return pygame.display.set_mode(size, flags)


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

    def __init__(self, fullscreen=False, window=(1280, 720), caption="WebcamDance", vsync=True):
        self.fullscreen, self.window, self.vsync = fullscreen, window, vsync
        self.screen = open_display(fullscreen, window, vsync)
        self._check_vsync()
        pygame.display.set_caption(caption)
        hud.set_scale(self.screen.get_height())

    def _check_vsync(self):
        """vsync réellement obtenu ? (le pilote peut le refuser : repli silencieux sinon)"""
        is_vsync = getattr(pygame.display, "is_vsync", None)
        self.vsync_on = bool(is_vsync()) if is_vsync and self.vsync else False
        if self.vsync and not self.vsync_on:
            print("[affichage] vsync refusé par le pilote : affichage sans synchro écran")

    def handle(self, e):
        """Traite redimensionnement / F11. -> True si la taille a changé (refaire la mise en page)."""
        if e.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.window = (max(640, e.w), max(360, e.h))
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_F11:
            self.fullscreen = not self.fullscreen
        else:
            return False
        self.reopen()
        return True

    def reopen(self, vsync=None):
        if vsync is not None:
            self.vsync = vsync
        self.screen = open_display(self.fullscreen, self.window, self.vsync)
        self._check_vsync()
        hud.set_scale(self.screen.get_height())

    def message(self, s):
        W, H = self.screen.get_size()
        self.screen.fill((14, 14, 28))
        hud.text(self.screen, s, 36, (255, 255, 255), (W // 2, H // 2), "center")
        pygame.display.flip()


class Session:
    """Caméra + modèle de pose, chargés une fois et gardés entre les parties."""

    def __init__(self, cfg):
        self.cfg, self.cam, self.pose = cfg, None, None
        self.cam_error = None
        media.limit_threads(cfg.game_threads)   # au-delà, plus de CPU (et de chauffe) sans gain de fluidité
        # la caméra s'ouvre dans un thread dédié (1-2 s) : le menu reste fluide et affiche l'aperçu
        self._cond = threading.Condition()
        self._want, self._done, self._closing = (cfg.cam_index, 0), None, False
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()
        # modèle de pose chargé en fond dès le démarrage (plusieurs secondes avec CUDA)
        self._pose_thread = threading.Thread(target=self._load_pose, daemon=True)
        self._pose_thread.start()

    def _load_pose(self):
        try:
            self.pose = PoseEstimator(self.cfg.rt_model, self.cfg)
        except Exception as e:      # erreur relancée par ensure(), au lancement d'une partie
            print(f"[pose] chargement du modèle impossible : {e}")

    @property
    def cam_ready(self):
        return self._done == self._want

    def set_camera(self, index):
        """Change de caméra (non bloquant). Rechoisir la même relance l'ouverture (rebranchement)."""
        with self._cond:
            self._want = (index, self._want[1] + 1)
            self._cond.notify_all()

    def _cam_loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closing or self._want != self._done)
                if self._closing:
                    break
                want = self._want
            old, self.cam = self.cam, None
            if old is not None:
                old.stop()
                old.join(timeout=2)
            self.cfg.cam_index = want[0]
            try:
                cam, err = Camera(self.cfg), None
                cam.start()
            except Exception as e:
                cam, err = None, str(e)
            with self._cond:
                self.cam, self.cam_error, self._done = cam, err, want
                self._cond.notify_all()
        if self.cam is not None:
            self.cam.stop()
            self.cam.join(timeout=2)

    def ensure(self, disp):
        if not self.cam_ready or self.pose is None:
            disp.message("Démarrage de la caméra et du modèle…")
        with self._cond:
            self._cond.wait_for(lambda: self.cam_ready)
        if self.cam is None:
            raise RuntimeError(self.cam_error or "Webcam inaccessible")
        self._pose_thread.join()
        if self.pose is None:       # échec du préchargement : nouvel essai (l'erreur remonte ici)
            self.pose = PoseEstimator(self.cfg.rt_model, self.cfg)

    def close(self):
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._cam_thread.join(timeout=3)
        # quitter pendant le préchargement du modèle : attendre sa fin (sinon plantage natif à la sortie)
        self._pose_thread.join(timeout=60)


def photo_rect(W, H):
    """Zone de la webcam / photo en mode groupe (16:9, sous la rangée de coachs)."""
    top, bottom = hud.px(208), H - hud.px(112)
    h = bottom - top
    w = min(int(h * 16 / 9), W - hud.px(40))
    h = int(w * 9 / 16)
    return pygame.Rect((W - w) // 2, top, w, h)


def group_buttons(W, H):
    """Boutons de l'écran photo : (clé, libellé, rect)."""
    px = hud.px
    labels = [("redo", "Refaire (R)", 220), ("same", "Même coach (T)", 240), ("start", "Commencer (Entrée)", 300)]
    gap = px(20)
    total = sum(px(w) for _, _, w in labels) + gap * (len(labels) - 1)
    x, y, out = (W - total) // 2, H - px(64), []
    for k, lab, w in labels:
        out.append((k, lab, pygame.Rect(x, y, px(w), px(48))))
        x += px(w) + gap
    return out


def draw_button(surf, label, rect, color, on=True):
    pygame.draw.rect(surf, color if on else (40, 40, 60), rect, border_radius=hud.px(12))
    hud.text(surf, label, 22, (10, 10, 20) if on else (220, 220, 230), rect.center, "center", shadow=False)


PAUSE_ITEMS = [("resume", "Reprendre"), ("restart", "Recommencer"), ("menu", "Retour au menu")]


def pause_buttons(W, H):
    """Boutons du menu pause : (clé, libellé, rect), empilés au centre."""
    px = hud.px
    w, h, gap = px(380), px(58), px(16)
    y0 = H // 2 - (len(PAUSE_ITEMS) * (h + gap) - gap) // 2 + px(30)
    return [(k, lab, pygame.Rect(W // 2 - w // 2, y0 + i * (h + gap), w, h)) for i, (k, lab) in enumerate(PAUSE_ITEMS)]


def draw_pause(surf, sel):
    W, H = surf.get_size()
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 170))
    surf.blit(dim, (0, 0))
    btns = pause_buttons(W, H)
    hud.text(surf, "PAUSE", 56, (255, 255, 255), (W // 2, btns[0][2].y - hud.px(24)), "midbottom")
    for i, (_, lab, r) in enumerate(btns):
        draw_button(surf, lab, r, (0, 229, 255), on=i == sel)
    hud.text(surf, "↑ / ↓ choisir · Entrée valider · Échap reprendre", 18, (200, 200, 215),
             (W // 2, btns[-1][2].bottom + hud.px(20)), "midtop", bold=False)


def person_at(raw, rect, pos):
    """Indice de la personne de la photo sous le clic (boîte la plus petite qui le contient)."""
    frame, boxes = raw[0], raw[3]
    s = rect.w / frame.shape[1]
    x, y = (pos[0] - rect.x) / s, (pos[1] - rect.y) / s
    hit = [(float((b[2] - b[0]) * (b[3] - b[1])), i) for i, b in enumerate(boxes)
           if b[0] - 20 <= x <= b[2] + 20 and b[1] - 40 <= y <= b[3] + 20]
    return min(hit)[1] if hit else None


def draw_people(surf, frame, kpts, rect, thr, styles):
    """Squelettes sur une image affichée dans rect. styles[i] = (couleur, épaisseur, étiquette | None)."""
    s = rect.w / frame.shape[1]
    tags = []
    for kp, (col, w, label) in zip(kpts, styles):
        hud.skeleton(surf, kp, col, thr, s, rect.topleft, w)
        if label and (kp[:, 2] >= thr).any():
            ok = kp[:, 2] >= thr
            x = rect.x + float(np.median(kp[ok, 0])) * s
            y = rect.y + float(kp[ok, 1].min()) * s
            tags.append(hud.tag(surf, label, 26 if label != "+" else 22, col, (x, y - hud.px(6)), "midbottom", clip=rect))
        else:
            tags.append(None)
    return tags


def run_game(disp, sess, song, n, log_path=None):
    """Joue une partie. -> ("menu" | "restart" | "quit", scores, records) ; scores vide si abandon avant la musique.
    En cas d'erreur (chanson illisible…), les threads de la partie sont arrêtés avant de la relayer."""
    started = []        # threads lancés (suivi des joueurs, vidéo, pictos)
    try:
        return _run_game(disp, sess, song, n, log_path, started)
    finally:
        pygame.mixer.music.stop()
        for th in started:
            th.stop()
        for th in started:
            if hasattr(th, "join"):
                th.join(timeout=2)
        for th in started:
            if hasattr(th, "discard"):       # enregistrement caméra : jamais gardé s'il n'est pas sauvé
                th.discard(disp)


def _run_game(disp, sess, song, n, log_path, started):
    cfg = sess.cfg
    group = n >= 5
    if group:   # on note ce qui est visible : un joueur du fond dont seuls les bras dépassent compte
        cfg = dataclasses.replace(cfg, player_min_cover=cfg.group_min_cover)
    song, n = Path(song), (0 if group else max(1, min(4, n)))
    meta = json.loads((song / "meta.json").read_text(encoding="utf-8"))
    pygame.display.set_caption(f"WebcamDance — {meta['title']}")
    sess.ensure(disp)
    pygame.key.set_repeat()         # pas de répétition en jeu (Espace maintenu = pause/reprise en boucle)
    disp.message("Chargement de la chanson…")
    coach = CoachTrack(song / "poses.npz", cfg)
    pictos = Pictos(coach.kpts, coach.fps, cfg.kpt_conf)
    started.append(pictos)
    highlights.clean_tmp(song)
    rec, hl = None, None            # enregistrement caméra (pendant le jeu) ; highlights (après)
    hl_play, hl_reader, hl_t0, hl_posters = None, None, 0.0, {}
    scores, records = [], []        # scores créés au démarrage de la musique (association figée)
    log = GameLog(log_path or Path("logs") / f"{song.name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
                  song=str(song), players=n, group=group, vsync=disp.vsync_on, cfg=dataclasses.asdict(cfg))
    cam = sess.cam
    worker = PlayerWorker(cam, sess.pose, cfg, n, group=group); worker.start(); started.append(worker)
    W, H = disp.screen.get_size()
    video = VideoPlayer(song / "video.mp4", (W, int(H * 0.65)), band=meta.get("band", (0.0, 1.0)),
                        coach_kpts=coach.kpts, thr=cfg.kpt_conf, dim=0.4); video.start(); started.append(video)

    def relayout():
        """Disposition à la résolution native : bandeau coach(s) en haut (65 %), bande joueurs en bas."""
        W, H = disp.screen.get_size()
        top_h = int(H * 0.65)
        video.set_fit((W, top_h))
        L = {"W": W, "H": H, "strip": pygame.Rect(0, top_h, W, H - top_h),
                "vrect": pygame.Rect((W - video.size[0]) // 2, (top_h - video.size[1]) // 2, *video.size),
                "thumbs": coach_thumbnails(song / "video.mp4", coach, cfg.kpt_conf, hud.px(150)),
                "row": pygame.Rect(0, hud.px(14), W, hud.px(150)),
                "cam": pygame.Rect(W // 2 - hud.px(320), hud.px(210), hud.px(640), hud.px(360))}
        if group:
            L["photo"], L["btns"] = photo_rect(W, H), group_buttons(W, H)
            L["count_btn"] = pygame.Rect(W // 2 - hud.px(170), H - hud.px(64), hud.px(340), hud.px(48))
            L["mini"] = [pygame.transform.smoothscale(t, (max(1, t.get_width() * hud.px(48) // t.get_height()), hud.px(48)))
                         for t in L["thumbs"]]
            L["photo_surf"] = None
        pictos.layout(L["vrect"])
        return L

    L = relayout()
    # chargement fini : objets présents (modèle, chanson) figés hors du ramasse-miettes -> pas de
    # collecte complète (~100 ms, saccade visible) pendant la partie
    gc.collect()
    gc.freeze()
    # joueurs -> coachs par rang gauche->droite (formation de départ), ←/→ pour décaler
    fx, _ = coach.formation(video.src_w)
    offset, manual = 0, False      # manual : ←/→ utilisées, le choix ne suit plus le joueur seul
    cmap = assign_coaches(n, fx, coach.present.mean(0), offset)
    ccolors = hud.coach_colors(coach.n, cmap)
    show_coach_skel, show_pictos = True, cfg.show_pictos
    pygame.mixer.music.load(str(song / "audio.ogg"))
    clock, tick = SongClock(), pygame.time.Clock()

    # mode groupe : photo (personnes triées par x), sélection (indices photo -> P1…Pn), coachs
    photo, sel, over, same, snap_t0, team, titles = None, [], {}, False, 0.0, None, []
    pres = coach.present.mean(0)

    def group_cmap():
        xs = photo[5][sel, 0] if sel else np.zeros(0)
        order = list(np.argsort(fx))
        if same:
            main = order.index(int(np.argmax(pres)))
            base = [int(order[(main + offset) % coach.n])] * len(sel)
        else:
            base = group_coaches(xs, fx, pres, offset)
        return [over.get(i, c) for i, c in zip(sel, base)]

    def take_photo():
        """Fin du comptage : personnes bras levés retenues (les plus grandes si trop nombreuses)."""
        nonlocal photo, sel, over, same
        raw = worker.raw
        if raw is None or len(raw[2]) == 0:
            return False
        photo, over, same = raw, {}, False
        up = [i for i, kp in enumerate(raw[2]) if arms_up(kp, cfg.kpt_conf)]
        up = sorted(up, key=lambda i: -raw[6][i])[:cfg.group_max]
        sel = sorted(up)
        L["photo_surf"] = None
        worker.lock_people(raw, range(len(raw[2])))     # suit tout le monde dès la photo
        return True

    def hl_ok():
        return hl is not None and hl.ok

    def hl_button(W, H):
        return pygame.Rect(W // 2 - hud.px(170), H - hud.px(100), hud.px(340), hud.px(50))

    def hl_start(i, now):
        """Lit le passage i (0 meilleur, 1 pire) avec la musique à cet endroit."""
        nonlocal hl_play, hl_reader, hl_t0
        if hl_reader is not None:
            hl_reader.close()
        c = hl.clips[i]
        hl_play, hl_reader, hl_t0 = i, hl.reader(c["a"]), now
        try:
            pygame.mixer.music.play(start=c["a"])
        except pygame.error as e:
            print(f"[highlights] musique : {e}")

    def hl_stop():
        nonlocal hl_play, hl_reader
        pygame.mixer.music.stop()
        if hl_reader is not None:
            hl_reader.close()
        hl_play, hl_reader = None, None

    def draw_highlights(screen, now):
        nonlocal hl_play
        W, H, px = L["W"], L["H"], hud.px
        screen.fill((14, 14, 28))
        hud.text(screen, "Highlights", 44, (255, 255, 255), (W // 2, px(24)), "midtop")
        panes, sbtn, fbtn = highlights.layout(W, H, hl.size[0] / hl.size[1])
        if hl_play is not None and now - hl_t0 >= hl.clips[hl_play]["b"] - hl.clips[hl_play]["a"]:
            nxt = hl_play + 1 if hl_play + 1 < len(hl.clips) else None     # meilleur, puis pire
            hl_stop()
            if nxt is not None:
                hl_start(nxt, now)
        for i, r in enumerate(panes):
            good = i == 0
            col = (118, 255, 3) if good else (255, 90, 90)
            hud.text(screen, "Meilleur moment" if good else "Pire moment", 26, col, (r.centerx, r.y - px(8)), "midbottom")
            if i >= len(hl.clips):
                pygame.draw.rect(screen, (32, 32, 54), r, border_radius=px(8))
                hud.text(screen, "partie trop courte", 20, (180, 180, 195), r.center, "center", bold=False)
                continue
            c = hl.clips[i]
            if hl_play == i:
                s = c["a"] + (now - hl_t0)
                f = hl_reader.get(hl.index(s))
                if f is not None:
                    screen.blit(hud.frame_surface(f, r.size), r)
                hl.overlays(screen, s, r)
                frac = min(1.0, (s - c["a"]) / (c["b"] - c["a"]))
                pygame.draw.rect(screen, (40, 40, 60), (r.x, r.bottom + px(4), r.w, px(5)))
                pygame.draw.rect(screen, col, (r.x, r.bottom + px(4), int(r.w * frac), px(5)))
            else:
                key = (i, r.size)
                if key not in hl_posters and c["poster"] is not None:
                    img = pygame.transform.smoothscale(hud.bgr_surface(c["poster"]), r.size)
                    img.fill((110, 110, 110), special_flags=pygame.BLEND_RGB_MULT)
                    hl_posters[key] = img
                if key in hl_posters:
                    screen.blit(hl_posters[key], r)
                cx, cy, k = r.centerx, r.centery, px(28)
                pygame.draw.polygon(screen, (255, 255, 255), [(cx - k // 2, cy - k), (cx - k // 2, cy + k), (cx + k, cy)])
            pygame.draw.rect(screen, col, r, px(3), border_radius=px(4))
            a, b = c["a"], c["b"]
            hud.text(screen, f"{int(a) // 60}:{int(a) % 60:02d} – {int(b) // 60}:{int(b) % 60:02d} · {c['label']} en moyenne",
                     20, (220, 220, 235), (r.centerx, r.bottom + px(14)), "midtop", bold=False)
            st = hl.saved.get(("best", "worst")[i])
            draw_button(screen, highlights.SAVE_TXT[st], sbtn[i], (0, 229, 255), on=st in (None, "error"))
        st = hl.saved.get("full")
        lab = "Enregistrer la partie entière" if st is None else highlights.SAVE_TXT[st]
        draw_button(screen, lab, fbtn, (255, 215, 64), on=st in (None, "error"))
        hud.text(screen, "Espace : rejouer · clic sur une vidéo : la revoir · Entrée / Échap : menu", 18,
                 (200, 200, 215), (W // 2, H - px(16)), "midbottom", bold=False)

    state, stable_since, countdown_t0 = (LOBBY if group else CALIB), None, 0.0
    vsurf, vid_id = None, -1
    pause_sel, resume_at = None, None   # menu pause ouvert (bouton sélectionné) ; fin du compte à rebours de reprise
    focus_lost = (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED)
    outcome = None
    while outcome is None:
        now = time.perf_counter()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                outcome = "quit"
            elif disp.handle(e):
                L = relayout()
            elif state == PLAY and pause_sel is not None and e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN,
                                                                           pygame.MOUSEMOTION):
                act = None
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_SPACE):
                        act = "resume"
                    elif e.key in (pygame.K_UP, pygame.K_DOWN):
                        pause_sel = (pause_sel + (1 if e.key == pygame.K_DOWN else -1)) % len(PAUSE_ITEMS)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        act = PAUSE_ITEMS[pause_sel][0]
                else:
                    hit = next((i for i, (_, _, r) in enumerate(pause_buttons(L["W"], L["H"]))
                                if r.collidepoint(e.pos)), None)
                    if hit is not None:
                        pause_sel = hit
                        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                            act = PAUSE_ITEMS[hit][0]
                if act == "resume":                     # reprise après un compte à rebours de 3 s
                    pause_sel, resume_at = None, now + 3.0
                elif act:
                    outcome = act
            elif state == PLAY and (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_SPACE)
                                    or e.type in focus_lost and cfg.pause_on_focus_loss):
                if clock.paused_at is None:             # Échap par mégarde : pause, pas d'abandon
                    clock.toggle_pause()
                pause_sel, resume_at = 0, None
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and state != PLAY:
                outcome = "menu"
            elif state == RESULTS and (e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                                       or e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and hl_ok()
                                       and hl_button(W, H).collidepoint(e.pos)):
                if hl_ok():
                    state = HILITE
                    hl_start(0, now)
                else:
                    outcome = "menu"
            elif state == HILITE and e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                outcome = "menu"
            elif state == HILITE and e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                hl_start(0, now)
            elif state == HILITE and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                panes, sbtn, fbtn = highlights.layout(W, H, hl.size[0] / hl.size[1])
                for i, c in enumerate(hl.clips):
                    key = ("best", "worst")[i]
                    if panes[i].collidepoint(e.pos):
                        hl_start(i, now)
                    elif sbtn[i].collidepoint(e.pos) and hl.saved.get(key) in (None, "error"):
                        hl.save(key, c["a"], c["b"])
                if fbtn.collidepoint(e.pos) and hl.saved.get("full") in (None, "error"):
                    hl.save("full", hl.t0, hl.t1)
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                show_coach_skel = not show_coach_skel
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                show_pictos = not show_pictos
            elif group and state == LOBBY and (e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                                               or e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
                                               and L["count_btn"].collidepoint(e.pos)):
                state, snap_t0 = SNAP, now
            elif group and state == PHOTO and (e.type == pygame.KEYDOWN
                                               or e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3)):
                act = None
                if e.type == pygame.KEYDOWN:
                    act = {pygame.K_RETURN: "start", pygame.K_KP_ENTER: "start", pygame.K_r: "redo",
                           pygame.K_t: "same", pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}.get(e.key)
                else:   # boutons : clic gauche seulement (ni clic droit ni molette)
                    btn = next((k for k, _, r in L["btns"] if r.collidepoint(e.pos)), None)
                    act = btn if e.button == 1 else None
                    i = None if btn else person_at(photo, L["photo"], e.pos)
                    if i is not None and e.button == 3:                 # clic droit : retirer / ajouter
                        act = None
                        if i in sel:
                            sel.remove(i)
                        elif len(sel) < cfg.group_max:
                            sel = sorted(sel + [i])
                    elif i is not None and e.button == 1:
                        if i not in sel:                                # clic sur un non retenu : ajouté
                            if len(sel) < cfg.group_max:
                                sel = sorted(sel + [i])
                        else:                                           # clic sur un joueur : coach suivant
                            order = list(np.argsort(fx))
                            cur = group_cmap()[sel.index(i)]
                            over[i] = int(order[(order.index(cur) + 1) % coach.n])
                if act in ("left", "right"):
                    offset += 1 if act == "right" else -1
                    over = {}
                elif act == "same":
                    same, offset, over = not same, 0, {}
                elif act == "redo":
                    worker.unlock()
                    photo, sel, state = None, [], LOBBY
                elif act == "start" and sel:
                    worker.keep(sel)
                    n, cmap = len(sel), group_cmap()
                    ccolors = hud.coach_colors(coach.n, cmap)
                    state, countdown_t0 = COUNTDOWN, now
                if photo is not None and state == PHOTO:
                    cmap = group_cmap()
                    ccolors = hud.coach_colors(coach.n, cmap)
            elif not group and e.type == pygame.KEYDOWN and e.key in (pygame.K_LEFT, pygame.K_RIGHT) and state in (CALIB, COUNTDOWN):
                step = 1 if e.key == pygame.K_RIGHT else -1
                if n == 1:   # décale depuis le coach affiché (éventuellement choisi par position)
                    order = list(np.argsort(fx))
                    cmap = [int(order[(order.index(cmap[0]) + step) % coach.n])]
                else:
                    offset += step
                    cmap = assign_coaches(n, fx, coach.present.mean(0), offset)
                manual = True
                ccolors = hud.coach_colors(coach.n, cmap)
        if resume_at is not None and now >= resume_at:
            resume_at = None
            clock.toggle_pause()
        if state != PLAY:       # poses inutilisées hors jeu (écran photo, résultats) : la file ne grossit pas
            while not worker.q.empty():
                worker.q.get()
        screen = disp.screen
        frame, poses = worker.view
        t = clock.time()
        W, H, px = L["W"], L["H"], hud.px

        if state in (LOBBY, SNAP):
            screen.fill((14, 14, 28))
            hud.coach_thumbs_row(screen, L["thumbs"], fx, [(hud.NEUTRAL, "")] * coach.n, L["row"])
            pr, raw = L["photo"], worker.raw
            if raw is not None:
                screen.blit(hud.frame_surface(raw[0], pr.size), pr)
                up = [arms_up(kp, cfg.kpt_conf) for kp in raw[2]]
                draw_people(screen, raw[0], raw[2], pr, cfg.kpt_conf,
                            [((118, 255, 3), px(5), None) if u else ((150, 150, 165), px(2), None) for u in up])
                nup = sum(up)
            else:
                nup = 0
            pygame.draw.rect(screen, (255, 255, 255), pr, px(2), border_radius=px(6))
            if state == LOBBY:
                hud.text(screen, f"Groupe (jusqu'à {cfg.group_max}) : placez-vous face à la caméra, en 2-3 rangées si besoin",
                         22, (255, 255, 255), (W // 2, pr.bottom + px(10)), "midtop", bold=False)
                draw_button(screen, "Lancer le comptage (Entrée)", L["count_btn"], (118, 255, 3))
            else:
                left = 5.0 - (now - snap_t0)
                hud.tag(screen, "Levez les bras !", 48, (255, 215, 64), (W // 2, pr.y + px(20)), "midtop")
                hud.text(screen, str(max(1, int(left) + 1)), 150, (255, 255, 255), pr.center, "center")
                hud.text(screen, f"{nup} bras levé(s)", 26, (118, 255, 3), (W // 2, pr.bottom + px(10)), "midtop")
                if left <= 0:
                    state = PHOTO if take_photo() else LOBBY
                    if state == PHOTO:
                        cmap = group_cmap()
                        ccolors = hud.coach_colors(coach.n, cmap)
            hud.text(screen, f"pose {worker.infer_ms:.0f} ms", 14, (180, 180, 190), (W - px(10), px(12)),
                     "topright", bold=False)

        elif state == HILITE:
            draw_highlights(screen, now)

        elif state == PHOTO:
            screen.fill((14, 14, 28))
            hud.coach_thumbs_row(screen, L["thumbs"], fx, ccolors, L["row"])
            pr = L["photo"]
            if L["photo_surf"] is None:
                L["photo_surf"] = hud.bgr_surface(cv2.resize(photo[0], pr.size))
            screen.blit(L["photo_surf"], pr)
            styles = [(hud.pcolor(sel.index(i)), px(4), f"P{sel.index(i) + 1}") if i in sel
                      else ((120, 120, 135), px(2), "+") for i in range(len(photo[2]))]
            tags = draw_people(screen, photo[0], photo[2], pr, cfg.kpt_conf, styles)
            for i, r in zip(sel, (tags[i] for i in sel)):         # vignette du coach suivi
                if r is not None:
                    m = L["mini"][cmap[sel.index(i)]]
                    mr = m.get_rect(midleft=(r.right - px(2), r.centery))
                    screen.blit(m, mr)
                    pygame.draw.rect(screen, hud.pcolor(sel.index(i)), mr, px(2), border_radius=px(4))
            pygame.draw.rect(screen, (255, 255, 255), pr, px(2), border_radius=px(6))
            hud.text(screen, f"{len(sel)} joueur(s) · clic : changer de coach · clic droit : retirer / ajouter · ← / → : décaler",
                     18, (220, 220, 235), (W // 2, pr.bottom + px(8)), "midtop", bold=False)
            for k, lab, r in L["btns"]:
                col = {"redo": (200, 200, 215), "same": (255, 215, 64), "start": (118, 255, 3)}[k]
                draw_button(screen, lab, r, col, on=(k != "same" or same) and (k != "start" or bool(sel)))

        elif state == CALIB:
            screen.fill((14, 14, 28))
            hud.coach_thumbs_row(screen, L["thumbs"], fx, ccolors, L["row"])
            hud.webcam(screen, frame, poses, L["cam"], cfg.kpt_conf)
            # corps entier exigé (épaules + au moins une cheville) : en jeu, un membre manquant
            # vient alors d'une occlusion ou d'une détection ratée, qui ne sont pas pénalisées
            full = sum(1 for kp in poses if kp is not None and (kp[5:7, 2] >= cfg.kpt_conf).all()
                       and (kp[15:17, 2] >= cfg.kpt_conf).any())
            ok = full >= n
            kp = poses[0] if n == 1 and coach.n > 1 and not manual and frame is not None else None
            if kp is not None and (kp[:, 2] >= cfg.kpt_conf).any():
                x = float(np.median(kp[kp[:, 2] >= cfg.kpt_conf, 0])) / frame.shape[1]
                c = solo_coach(x, fx, coach.present.mean(0), cmap[0])
                if c != cmap[0]:
                    cmap, ccolors = [c], hud.coach_colors(coach.n, [c])
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
            if coach.n > 1 and not group:
                hud.text(screen, "← / → : changer de coach", 20, (200, 200, 215), (W // 2, px(648)), "center", bold=False)
            elif group:
                hud.text(screen, "Restez à peu près à votre place !", 24, (255, 255, 255), (W // 2, px(610)), "center")
            left = 5.0 - (now - countdown_t0)
            hud.text(screen, str(max(1, int(left) + 1)), 160, (255, 255, 255), (W // 2, H // 2), "center")
            if left <= 0:
                while not worker.q.empty():
                    worker.q.get()
                scores = [PlayerScore(cfg, coach, cmap[p]) for p in range(n)]
                if group:
                    team = TeamScore(scores, cfg.group_combo_frac, cfg.move_period, cfg.grace)
                hud.reset_crops()
                log.write("mapping", coach_of_player=cmap, players=n)
                pictos.set_colors(cmap)
                try:
                    rec = highlights.Recorder(cam, song)
                    rec.start()
                    started.append(rec)
                except Exception as e:      # pas d'enregistrement : la partie se joue sans highlights
                    print(f"[highlights] enregistrement impossible : {e}")
                clock.start()
                state = PLAY

        else:  # PLAY / RESULTS
            if state == PLAY:
                video.set_time(t)
                paused = clock.paused_at is not None
                if rec is not None:
                    rec.mark(now, t)
                while not worker.q.empty():
                    t_cap, ps = worker.q.get()
                    if paused:
                        continue
                    ts = t - (now - t_cap) - cfg.cam_latency  # temps chanson de la capture
                    log.track(ts, ps, cfg.kpt_conf)
                    if rec is not None:
                        rec.add_heads(ts, ps, cfg.kpt_conf)
                    for p, kp in enumerate(ps):
                        if kp is not None:
                            scores[p].add(ts, kp)
                for sc in scores:
                    sc.close_due(t)
                if team:
                    team.update()
                log.perf(now, t, tick, worker, cam, int(t * video.fps) - vid_id)
                log.moves(scores)
                if t >= meta["duration"] or not pygame.mixer.music.get_busy() and not paused and t > 1.0:
                    for sc in scores:
                        sc.close_due(t + 10)
                    if team:
                        team.update()
                        titles = fun_titles(scores)
                    state = RESULTS
                    worker.stop()           # plus besoin des poses : GPU libéré
                    log.close(scores)
                    records = library.save_result(song, scores, n)
                    print(f"[play] journal : {log.path}")
                    if rec is not None:
                        rec.stop()
                        rec.join(timeout=2)
                        try:
                            hl = highlights.Highlights(rec, scores, cfg, song, song / "audio.ogg", meta["title"])
                        except Exception as e:
                            print(f"[highlights] indisponibles : {e}")
                            hl = None
            if video.frame_id != vid_id and video.frame is not None:
                vid_id, vsurf = video.frame_id, hud.bgr_surface(video.frame)
            screen.fill((0, 0, 0))
            vrect = L["vrect"]
            if vsurf:
                screen.blit(vsurf, vsurf.get_rect(center=vrect.center))
                if 0 <= vid_id < coach.T:
                    hud.coach_overlay(screen, coach, vid_id, vrect, video.scale, video.y0,
                                      cfg.kpt_conf, ccolors, skel=show_coach_skel)
                if show_pictos and state == PLAY:
                    pictos.draw(screen, vid_id / coach.fps)
            ci = min(max(vid_id, 0), coach.T - 1)
            waiting = [not coach.present[ci, cmap[p]] for p in range(n)]
            hud.progress(screen, min(1.0, t / meta["duration"]))
            if group:
                hud.group_strip(screen, L["strip"], frame, poses, scores, team, t, cfg.kpt_conf)
            else:
                hud.player_strip(screen, L["strip"], frame, poses, scores, t, cfg.kpt_conf, waiting)
            hud.text(screen, f"{tick.get_fps():.0f} fps · pose {worker.infer_ms:.0f} ms", 14,
                     (180, 180, 190), (W - px(10), px(12)), "topright", bold=False)
            if state == RESULTS:
                hint = "Entrée : highlights · Échap : menu" if hl_ok() else "Entrée : retour"
                if group:
                    hud.group_results(screen, scores, team, titles, hint)
                else:
                    hud.results(screen, scores, records, hint)
                if hl_ok():
                    draw_button(screen, "Voir les highlights", hl_button(W, H), (118, 255, 3))
            elif pause_sel is not None:
                draw_pause(screen, pause_sel)
            elif resume_at is not None:
                hud.text(screen, str(int(resume_at - now) + 1), 160, (255, 255, 255), (W // 2, H // 2), "center")

        if state not in (RESULTS, HILITE) and now - cam.latest()[2] > 2.0:     # plus d'image depuis 2 s : câble débranché ?
            hud.tag(screen, "Caméra perdue : vérifie le câble (reconnexion automatique)", 26, (255, 90, 90),
                    (W // 2, H // 2 - px(120)), "center")
        pygame.display.flip()
        tick.tick(cfg.render_fps)

    pygame.mixer.music.stop()
    if not log.f.closed:  # partie interrompue : on garde ce qui a été joué (scores vides si avant le jeu)
        log.close(scores)
        print(f"[play] journal : {log.path}")
    worker.stop(); video.stop(); pictos.stop()
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
    ap.add_argument("--fps", type=int, help="images/s max de l'affichage (défaut 60)")
    ap.add_argument("--no-vsync", action="store_true", help="sans synchro écran")
    ap.add_argument("--log", help="journal JSONL (défaut : logs/<chanson>_<date>.jsonl)")
    a = ap.parse_args()
    cfg = config_from_args(a)
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    disp = Display(a.fullscreen, tuple(int(v) for v in a.window.lower().split("x")), vsync=cfg.vsync)
    sess = Session(cfg)
    try:
        while run_game(disp, sess, a.song, a.players, a.log)[0] == "restart":
            pass
    finally:
        sess.close()
        pygame.quit()


def config_from_args(a):
    cfg = Config(cam_index=a.camera)
    if a.model:
        cfg.rt_model = a.model
    if a.fps:
        cfg.render_fps = a.fps
    if getattr(a, "no_vsync", False):
        cfg.vsync = False
    if a.latency_ms is not None:
        cfg.cam_latency = a.latency_ms / 1000
    return cfg


if __name__ == "__main__":
    main()
