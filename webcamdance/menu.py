"""Menu principal : choix de la chanson et du nombre de joueurs, ajout de chansons par lien
(une vidéo, plusieurs liens collés d'un coup, ou une playlist), meilleurs scores.

    python -m webcamdance [--songs-dir songs] [--fullscreen] [--window 1920x1080]

Touches : ↑/↓ choisir · Entrée jouer · 1-4 joueurs, 5 groupe · C caméra · R réglages · A ajouter · Suppr supprimer ·
Ctrl+F rechercher · F11 plein écran · Échap quitter.
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pygame

from . import cameras, hud, library
from .play import Display, Session, config_from_args, run_game
from .songinfo import display_name

BG, PANEL, ACCENT = (14, 14, 28), (26, 26, 46), (0, 229, 255)


class Importer:
    """Imports en arrière-plan, un processus à la fois (le GPU est partagé). Chaque demande
    (liste de liens ou playlist) devient un lot de import_song ; sa sortie est lue pour
    afficher la progression."""

    def __init__(self, songs_dir):
        self.songs_dir = Path(songs_dir)
        self.pending = []           # [(liens, playlist ?)]
        self.proc = None
        self.lines = queue.SimpleQueue()
        self.status = ""            # ligne d'état affichée
        self.done_count = self.fail_count = self.skip_count = self.rej_count = 0
        self.changed = False        # une chanson vient d'arriver : rafraîchir la liste
        self.paused = False         # figé pendant une partie
        self._cur = {"k": 0, "n": 0, "title": "", "stage": ""}

    @property
    def busy(self):
        return self.proc is not None or bool(self.pending)

    def add(self, text):
        """Texte collé -> lots en file. -> nombre de liens retenus."""
        links = library.parse_links(text)
        playlists = [u for u in links if "list=" in u and not library.youtube_id(u)]
        videos = [u for u in links if u not in playlists]
        if videos:
            self.pending.append((videos, False))
        for u in playlists:
            self.pending.append(([u], True))
        return len(links)

    def _start(self, links, playlist):
        tmp = self.songs_dir / "_cache" / f"liens_{int(time.time() * 1000)}.txt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("\n".join(links), encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "webcamdance.import_song", "--songs-dir", str(self.songs_dir), "-y"]
        cmd += ["--playlist", *links] if playlist else ["--links-file", str(tmp)]
        # UTF-8 forcé : sous Windows, la sortie d'un processus vers un tube est sinon en cp1252
        # (accents illisibles ici, et plantage sur un titre hors cp1252)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1, env=env)
        self._tmp = tmp
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        self._cur = {"k": 0, "n": len(links) if not playlist else 0, "title": "", "stage": "démarrage"}

    def _read(self, proc):
        for line in proc.stdout:
            self.lines.put(line.rstrip())
        self.lines.put(None)

    def poll(self):
        if self.proc is None and self.pending:
            self._start(*self.pending.pop(0))
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:                        # fin du processus
                self.proc.wait()
                self._tmp.unlink(missing_ok=True)
                self.proc = None
                continue
            self._parse(line)
        c = self._cur
        if self.proc is not None:
            pos = f"{c['k']}/{c['n']} " if c["n"] else ""
            self.status = f"Import {pos}{c['title'][:40]} — {c['stage']}"
        elif self.done_count or self.fail_count or self.skip_count or self.rej_count:
            self.status = (f"Imports terminés : {self.done_count} ajoutée(s), {self.skip_count} déjà présente(s), "
                           f"{self.rej_count} rejetée(s) (coach mal détecté), {self.fail_count} échec(s)")

    def _parse(self, line):
        c = self._cur
        if m := re.match(r"\[lot\] (\d+)/(\d+)", line):
            c.update(k=int(m.group(1)), n=int(m.group(2)), title="", stage="téléchargement")
        elif m := re.match(r"\[import\] titre : (.*) ->", line):
            c["title"] = m.group(1)
        elif line.startswith("[import] détection"):
            c["stage"] = "recherche de la partie dansée"
        elif line.startswith("[import] normalisation"):
            c["stage"] = "conversion"
        elif m := re.match(r"\[extract\] (\d+) %", line):
            c["stage"] = f"analyse des poses {m.group(1)} %"
        elif line.startswith("[import] OK"):
            self.done_count += 1
            self.changed = True
        elif "REJETÉE" in line:
            self.rej_count += 1
        elif "ÉCHEC" in line:
            self.fail_count += 1
        elif line.startswith("[lot] déjà importée"):
            c["stage"] = "déjà présente"
            self.skip_count += 1

    def _tree(self):
        """Processus d'import et ses enfants (ffmpeg, yt-dlp…), ou [] s'il est fini."""
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            return [p] + p.children(recursive=True)
        except Exception:
            return []

    def pause(self):
        """Partie lancée : l'import est figé (GPU et CPU rendus au jeu), sans perdre son avancement."""
        if self.proc is None or self.paused:
            return
        self.paused = True
        for p in self._tree():      # parent d'abord : il ne peut plus lancer de nouvel enfant
            try:
                p.suspend()
            except Exception:
                pass

    def resume(self):
        if not self.paused:
            return
        self.paused = False
        if self.proc is None:
            return
        for p in reversed(self._tree()):
            try:
                p.resume()
            except Exception:
                pass

    def stop(self):
        self.pending.clear()
        self.resume()               # un processus figé ne traiterait pas l'arrêt (Linux)
        if self.proc is not None:
            children = self._tree()[1:]     # ffmpeg, yt-dlp… : sinon ils continuent seuls
            self.proc.terminate()
            for c in children:
                try:
                    c.terminate()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            for c in children:
                try:
                    c.wait(timeout=3)
                except Exception:
                    try:
                        c.kill()
                    except Exception:
                        pass


def clean_incomplete(songs_dir, min_age=3600):
    """Dossiers d'imports interrompus (sans meta.json) : à supprimer au démarrage. Ceux modifiés il y a
    moins d'une heure sont gardés : un import_list lancé à côté est peut-être en train de les remplir."""
    import shutil
    root = Path(songs_dir)
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and not d.name.startswith("_") and not (d / "meta.json").exists():
            try:
                if time.time() - d.stat().st_mtime < min_age:
                    continue
            except OSError:
                continue
            shutil.rmtree(d, ignore_errors=True)


def fold(s):
    """Minuscules sans accents, pour une recherche tolérante."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.casefold()) if not unicodedata.combining(c))


def filter_songs(songs, query):
    """Chansons dont le nom affiché contient tous les mots de la recherche (ordre libre)."""
    words = fold(query).split()
    return [s for s in songs if all(w in fold(display_name(s["meta"])) for w in words)]


def fit_text(s, size, max_w, bold=False):
    """Texte raccourci (…) pour tenir dans max_w pixels ; "" si rien ne tient."""
    f = hud.font(size, bold)
    if f.size(s)[0] <= max_w:
        return s
    while s and f.size(s + "…")[0] > max_w:
        s = s[:-1]
    return s.rstrip() + "…" if s else ""


def fmt_dur(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


# réglages du menu « R » : (clé settings.json, libellé, valeurs possibles, aide)
OPTIONS = [("vsync", "Synchro écran (vsync)", [True, False], "images régulières, sans saccade"),
           ("fps", "Images par seconde", [60, 30], "60 : défilement plus doux (un peu plus de CPU)"),
           ("pictos", "Pictos des mouvements à venir", [True, False], "aussi la touche P en jeu"),
           ("hl_dur", "Durée des highlights", [5, 6, 7, 8, 9, 10], "meilleur et pire passage, en fin de partie")]
DEFAULTS = {"vsync": True, "fps": 60, "pictos": True, "hl_dur": 5}


def apply_settings(cfg, settings, keep_fps=False):
    """Réglages persistants -> configuration de jeu."""
    cfg.vsync = bool(settings.get("vsync", DEFAULTS["vsync"]))
    cfg.show_pictos = bool(settings.get("pictos", DEFAULTS["pictos"]))
    try:
        cfg.highlight_seconds = min(10, max(5, int(settings.get("hl_dur", DEFAULTS["hl_dur"]))))
    except (TypeError, ValueError):     # settings.json modifié à la main
        cfg.highlight_seconds = DEFAULTS["hl_dur"]
    if not keep_fps:
        cfg.render_fps = int(settings.get("fps", DEFAULTS["fps"]))


def fmt_opt(v, key=None):
    if isinstance(v, bool):
        return "Oui" if v else "Non"
    return f"{v} s" if key == "hl_dur" else str(v)


KP_KEYS = [pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP4, pygame.K_KP5]


class Menu:
    ROW = 116

    def __init__(self, disp, sess, songs_dir, settings, cams):
        self.disp, self.sess, self.songs_dir = disp, sess, Path(songs_dir)
        self.settings, self.cams = settings, cams
        self._preview = (None, None)    # (fid, surface) de l'aperçu webcam
        self.importer = Importer(songs_dir)
        self.players, self.sel, self.scroll = 1, 0, 0
        self.mode = "list"          # list | add | delete | settings
        self.opt_sel = 0
        self.text = ""
        self.query, self.searching = "", False
        self.flash, self.flash_t = "", 0.0
        self._thumbs = {}
        self.refresh()

    def refresh(self, reload=True):
        cur = self.songs[self.sel]["path"] if getattr(self, "songs", None) and self.songs else None
        if reload:
            self.all_songs = library.list_songs(self.songs_dir)
        self.songs = filter_songs(self.all_songs, self.query)
        names = [s["path"] for s in self.songs]
        self.sel = names.index(cur) if cur in names else min(self.sel, max(0, len(self.songs) - 1))

    def say(self, msg):
        self.flash, self.flash_t = msg, time.perf_counter()

    def thumb(self, song, w, h):
        key = (song["path"], w, h)
        if key not in self._thumbs:
            p = library.thumbnail(song["path"])
            img = library.read_image(p) if p else None
            if img is None:
                img = np.full((h, w, 3), 50, np.uint8)
            s = max(w / img.shape[1], h / img.shape[0])            # remplit puis recadre au centre
            img = cv2.resize(img, (max(w, int(img.shape[1] * s)), max(h, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
            y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
            self._thumbs[key] = hud.bgr_surface(np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w]))
        return self._thumbs[key]

    # ------------------------------------------------------------------ boucle
    def run(self):
        tick = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if self.disp.handle(e):
                    self._thumbs.clear()
                    continue
                if self.handle(e) == "quit":
                    return
            self.importer.poll()
            if self.importer.changed:
                self.importer.changed = False
                self.refresh()
            self.draw()
            pygame.display.flip()
            tick.tick(30)

    def play(self):
        if not self.songs:
            return
        song = self.songs[self.sel]
        self.importer.pause()
        try:
            outcome = "restart"
            while outcome == "restart":             # « Recommencer » du menu pause
                outcome, _, _ = run_game(self.disp, self.sess, song["path"], self.players)
        except Exception as err:            # partie impossible : retour au menu, pas de fermeture du jeu
            if self.sess.cam is None:
                print(f"[menu] {err}")
                self.say("Caméra indisponible : choisis-en une autre (C)")
            else:
                import traceback
                traceback.print_exc()
                self.say(f"Partie impossible : {str(err)[:70]}")
            pygame.display.set_caption("WebcamDance")
            return
        finally:
            self.importer.resume()
            pygame.key.set_repeat(300, 40)      # coupée pendant la partie
        pygame.display.set_caption("WebcamDance")
        self.refresh()
        if outcome == "quit":
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def cam_name(self):
        i = self.sess.cfg.cam_index
        return next((n for k, n in self.cams if k == i), f"Caméra {i + 1}")

    def next_camera(self):
        """Caméra suivante (liste relue : prise en compte des branchements à chaud)."""
        self.cams = cameras.list_cameras(busy={self.sess.cfg.cam_index}) or self.cams
        idx = [i for i, _ in self.cams]
        cur = self.sess.cfg.cam_index
        nxt = idx[(idx.index(cur) + 1) % len(idx)] if cur in idx else (idx[0] if idx else cur)
        self._preview = (None, None)
        self.sess.set_camera(nxt)
        self.sess.cfg.cam_index = nxt
        self.settings.set(camera=self.cam_name())

    def set_query(self, q):
        self.query = q
        self.refresh(reload=False)
        self.sel = self.scroll = 0

    def focus_search(self, on):
        if on != self.searching:
            self.searching = on
            (pygame.key.start_text_input if on else pygame.key.stop_text_input)()

    def handle(self, e):
        if self.mode == "add":
            return self._handle_add(e)
        if self.mode == "settings":
            return self._handle_settings(e)
        if self.searching and self._handle_search(e):
            return None
        if self.mode == "delete":
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_o):
                library.delete_song(self.songs[self.sel]["path"], self.songs_dir)
                self.say("Chanson supprimée")
                self.refresh()
                self.mode = "list"
            elif e.type == pygame.KEYDOWN or e.type == pygame.MOUSEBUTTONDOWN:
                self.mode = "list"
            return None
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k == pygame.K_ESCAPE:
                if self.query:
                    self.set_query("")
                    return None
                return "quit"
            if k == pygame.K_f and e.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
                self.focus_search(True)
                return None
            if k in (pygame.K_DOWN, pygame.K_UP) and self.songs:
                self.sel = (self.sel + (1 if k == pygame.K_DOWN else -1)) % len(self.songs)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.play()
            elif pygame.K_1 <= k <= pygame.K_5:          # 5 = mode groupe « 5+ »
                self.players = k - pygame.K_0
            elif k in KP_KEYS:
                self.players = KP_KEYS.index(k) + 1
            elif k == pygame.K_c:
                self.next_camera()
            elif k == pygame.K_r:
                self.mode, self.opt_sel = "settings", 0
            elif k == pygame.K_a:
                self.mode, self.text = "add", ""
                pygame.key.start_text_input()
            elif k == pygame.K_DELETE and self.songs:
                self.mode = "delete"
        elif e.type == pygame.MOUSEWHEEL and self.songs:
            self.sel = min(max(self.sel - e.y, 0), len(self.songs) - 1)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self._clear_rect().collidepoint(e.pos) and self.query:
                self.set_query("")
                return None
            if self._search_rect().collidepoint(e.pos):
                self.focus_search(True)
                return None
            for i, r in self._chip_rects():
                if r.collidepoint(e.pos):
                    self.players = i
                    return None
            if self._cam_rect().collidepoint(e.pos):
                self.next_camera()
                return None
            if self._add_rect().collidepoint(e.pos):
                self.mode, self.text = "add", ""
                pygame.key.start_text_input()
                return None
            for i, r in self._visible_rows():
                if r.collidepoint(e.pos):
                    if i == self.sel and getattr(self, "_last_click", (None, 0))[1] > time.perf_counter() - 0.4:
                        self.play()
                    self.sel = i
                    self._last_click = (i, time.perf_counter())
        return None

    def set_option(self, key, value):
        self.settings.set(**{key: value})
        cfg = self.sess.cfg
        apply_settings(cfg, self.settings)
        if key == "vsync":
            self.disp.reopen(vsync=cfg.vsync)
            self._thumbs.clear()

    def _cycle_option(self, i, step=1):
        key, _, values, _ = OPTIONS[i]
        cur = self.settings.get(key, DEFAULTS[key])
        nxt = values[(values.index(cur) + step) % len(values)] if cur in values else values[0]
        self.set_option(key, nxt)

    def _handle_settings(self, e):
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k in (pygame.K_ESCAPE, pygame.K_r):
                self.mode = "list"
            elif k in (pygame.K_UP, pygame.K_DOWN):
                self.opt_sel = (self.opt_sel + (1 if k == pygame.K_DOWN else -1)) % len(OPTIONS)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
                self._cycle_option(self.opt_sel)
            elif k == pygame.K_LEFT:
                self._cycle_option(self.opt_sel, -1)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i, r in enumerate(self._option_rects()):
                if r.collidepoint(e.pos):
                    self.opt_sel = i
                    self._cycle_option(i)
                    return None
            if not self._settings_box().collidepoint(e.pos):
                self.mode = "list"
        return None

    def _settings_box(self):
        px = hud.px
        W, H = self.disp.screen.get_size()
        box = pygame.Rect(0, 0, min(W - px(60), px(760)), px(110 + 66 * len(OPTIONS)))
        box.center = (W // 2, H // 2)
        return box

    def _option_rects(self):
        px, box = hud.px, self._settings_box()
        return [pygame.Rect(box.x + px(24), box.y + px(70) + i * px(66), box.w - px(48), px(58))
                for i in range(len(OPTIONS))]

    def _draw_settings(self):
        s, px = self.disp.screen, hud.px
        W, H = s.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        s.blit(dim, (0, 0))
        box = self._settings_box()
        pygame.draw.rect(s, PANEL, box, border_radius=px(14))
        pygame.draw.rect(s, ACCENT, box, px(2), border_radius=px(14))
        hud.text(s, "Réglages", 26, (255, 255, 255), (box.centerx, box.y + px(20)), "midtop")
        for i, ((key, label, _, help_), r) in enumerate(zip(OPTIONS, self._option_rects())):
            on = i == self.opt_sel
            pygame.draw.rect(s, (40, 40, 70) if on else (32, 32, 54), r, border_radius=px(10))
            if on:
                pygame.draw.rect(s, ACCENT, r, px(2), border_radius=px(10))
            hud.text(s, label, 22, (255, 255, 255), (r.x + px(16), r.y + px(6)), shadow=False)
            hud.text(s, help_, 15, (160, 160, 180), (r.x + px(16), r.y + px(34)), bold=False, shadow=False)
            v = self.settings.get(key, DEFAULTS[key])
            hud.text(s, fmt_opt(v, key), 22, ACCENT if v is True or v == 60 else (220, 220, 230),
                     (r.right - px(16), r.centery), "midright", shadow=False)
        hud.text(s, "↑/↓ choisir · Entrée changer · Échap fermer", 15, (150, 150, 170),
                 (box.centerx, box.bottom - px(14)), "midbottom", bold=False, shadow=False)

    def _handle_search(self, e):
        """Saisie dans la barre de recherche. Renvoie True si l'événement est consommé ; sinon
        il passe au traitement normal (flèches, Entrée, souris, molette)."""
        if e.type == pygame.TEXTINPUT:
            self.set_query(self.query + e.text)
            return True
        if e.type == pygame.KEYDOWN:
            ctrl = e.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
            if e.key == pygame.K_ESCAPE:
                self.focus_search(False)
            elif e.key == pygame.K_BACKSPACE:
                self.set_query("" if ctrl else self.query[:-1])
            elif e.key == pygame.K_v and ctrl:
                try:
                    self.set_query(self.query + (pygame.scrap.get_text() or "").replace("\n", " "))
                except Exception:
                    self.say("Presse-papiers inaccessible")
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.focus_search(False)
                return False                            # Entrée lance la chanson sélectionnée
            elif e.key in (pygame.K_UP, pygame.K_DOWN):
                return False
            return True                                 # les autres raccourcis (A, 1-5, Suppr…) sont muets
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if not self._search_rect().collidepoint(e.pos):
                self.focus_search(False)
            return False
        return False

    def _handle_add(self, e):
        if e.type == pygame.TEXTINPUT:
            self.text += e.text
        elif e.type == pygame.KEYDOWN:
            ctrl = e.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
            if e.key == pygame.K_ESCAPE:
                self.mode = "list"
                pygame.key.stop_text_input()
            elif e.key == pygame.K_v and ctrl:
                try:
                    self.text += pygame.scrap.get_text() or ""
                except Exception:
                    self.say("Presse-papiers inaccessible")
            elif e.key == pygame.K_BACKSPACE:
                self.text = "" if ctrl else self.text[:-1]
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                n = self.importer.add(self.text)
                self.say(f"{n} lien(s) ajouté(s) à la file d'import" if n else "Aucun lien valide")
                self.mode = "list"
                pygame.key.stop_text_input()
        return None

    # ------------------------------------------------------------------ dessin
    def _chip_rects(self):
        px = hud.px
        W = self.disp.screen.get_width()
        return [(i, pygame.Rect(W - px(30) - (6 - i) * px(56), px(26), px(48), px(44))) for i in range(1, 6)]

    def _cam_rect(self):
        """Aperçu webcam 16:9, à gauche du libellé « Joueurs »."""
        px = hud.px
        h = px(56)
        r = pygame.Rect(0, 0, h * 16 // 9, h)
        r.topright = (self._chip_rects()[0][1].x - px(120), px(20))
        return r

    def _draw_camera(self):
        s, px = self.disp.screen, hud.px
        r = self._cam_rect()
        cam = self.sess.cam
        pygame.draw.rect(s, PANEL, r, border_radius=px(8))
        if cam is not None:
            fid, frame, _ = cam.latest()
            if self._preview[0] != fid or self._preview[1].get_size() != r.size:
                small = cv2.resize(frame, r.size, interpolation=cv2.INTER_AREA)
                self._preview = (fid, hud.bgr_surface(small))
            s.blit(self._preview[1], r)
        else:
            msg = "…" if not self.sess.cam_ready else "Aucune image"
            hud.text(s, msg, 16, (170, 170, 190), r.center, "center", bold=False, shadow=False)
        pygame.draw.rect(s, ACCENT if cam is not None else (200, 80, 80), r, px(2), border_radius=px(8))
        room = r.x - px(12) - getattr(self, "_logo_right", 0) - px(16)    # écran étroit (4:3) : pas sur le logo
        full = self.cam_name()[:28]
        name = fit_text(full, 20, room)
        if name and (name == full or len(name) >= 10):     # nom trop rogné (« In… ») : rien plutôt qu'illisible
            hud.text(s, name, 20, (220, 220, 230), (r.x - px(12), r.centery - px(11)), "midright", bold=False)
            hint = "C : changer" if len(self.cams) > 1 else "Caméra"
            hud.text(s, hint, 15, (150, 150, 170), (r.x - px(12), r.centery + px(13)), "midright", bold=False)

    def _add_rect(self):
        px = hud.px
        W, H = self.disp.screen.get_size()
        return pygame.Rect(px(30), H - px(64), px(330), px(46))

    def _search_rect(self):
        px = hud.px
        return pygame.Rect(px(30), px(88), self.disp.screen.get_width() - px(60), px(46))

    def _clear_rect(self):
        r = self._search_rect()
        return pygame.Rect(r.right - r.h, r.y, r.h, r.h)

    def _list_area(self):
        px = hud.px
        W, H = self.disp.screen.get_size()
        top = self._search_rect().bottom + px(12)
        bottom = H - px(84) - (px(36) if self.importer.status else 0)
        return pygame.Rect(px(30), top, W - px(60), bottom - top)

    def _visible_rows(self):
        area, rh = self._list_area(), hud.px(self.ROW)
        per = max(1, area.h // rh)
        self.scroll = min(max(self.scroll, self.sel - per + 1), self.sel)
        self.scroll = max(0, min(self.scroll, max(0, len(self.songs) - per)))
        return [(i, pygame.Rect(area.x, area.y + (i - self.scroll) * rh, area.w, rh - hud.px(8)))
                for i in range(self.scroll, min(len(self.songs), self.scroll + per))]

    def draw(self):
        s, px = self.disp.screen, hud.px
        W, H = s.get_size()
        s.fill(BG)
        self._logo_right = hud.tag(s, "WebcamDance", 44, ACCENT, (px(24), px(14)), "topleft").right
        hud.text(s, "Joueurs", 22, (200, 200, 215), (self._chip_rects()[0][1].x - px(12), px(36)), "topright",
                 bold=False)
        for i, r in self._chip_rects():
            on = i == self.players
            pygame.draw.rect(s, hud.COLORS[i - 1] if on else PANEL, r, border_radius=px(10))
            hud.text(s, "5+" if i == 5 else str(i), 26, (10, 10, 20) if on else (220, 220, 230), r.center, "center",
                     shadow=False)
        self._draw_camera()
        self._draw_search()
        if not self.all_songs:
            hud.text(s, "Aucune chanson : appuie sur A pour en ajouter (lien YouTube, liste ou playlist)",
                     24, (200, 200, 215), (W // 2, H // 2), "center", bold=False)
        elif not self.songs:
            hud.text(s, "Aucun résultat", 24, (200, 200, 215), (W // 2, H // 2), "center", bold=False)
        for i, r in self._visible_rows():
            song, on = self.songs[i], i == self.sel
            pygame.draw.rect(s, (40, 40, 70) if on else PANEL, r, border_radius=px(12))
            if on:
                pygame.draw.rect(s, ACCENT, r, px(3), border_radius=px(12))
            th = self.thumb(song, int(r.h * 16 / 9), r.h - px(12))
            s.blit(th, (r.x + px(6), r.y + px(6)))
            x = r.x + th.get_width() + px(22)
            m = song["meta"]
            hud.text(s, display_name(m)[:64], 26, (255, 255, 255), (x, r.y + px(14)))
            nc = m.get("n_coaches", 1)
            info = f"{nc} coach{'s' if nc > 1 else ''} · {fmt_dur(m.get('duration', 0))}"
            hud.text(s, info, 18, (170, 170, 190), (x, r.y + px(52)), bold=False)
            if "coach_presence" in m:                # qualité de détection des coachs (jugeabilité)
                p = m["coach_presence"]
                det = f"détection {p:.0%}" + (" (" + " / ".join(f"{q:.0%}" for q in m["coach_presence_per"]) + ")"
                                              if len(m.get("coach_presence_per", [])) > 1 else "")
                col = (120, 220, 120) if p >= 0.85 else (255, 200, 80)
                hud.text(s, det, 16, col, (x, r.y + px(78)), bold=False)
            if m.get("trim_uncertain"):
                hud.text(s, "  · coupe à vérifier", 18, (255, 170, 60), (x + hud.font(18, False).size(info)[0],
                                                                         r.y + px(52)), bold=False)
            b = song["best"]
            if b:
                hud.text(s, f"{b['total']:,}".replace(",", " "), 30, (255, 255, 255), (r.right - px(20), r.y + px(12)),
                         "topright")
                for k in range(5):
                    hud.star(s, r.right - px(30) - (4 - k) * px(26), r.y + px(66), px(10),
                             (255, 215, 64) if b["stars"] > k else (90, 90, 110))
            else:
                hud.text(s, "jamais jouée", 18, (140, 140, 160), (r.right - px(20), r.y + px(30)), "topright",
                         bold=False)
        if self.importer.status:
            hud.text(s, self.importer.status, 18, (255, 215, 64) if self.importer.busy else (160, 220, 160),
                     (px(30), H - px(108)), bold=False)
        ar = self._add_rect()
        pygame.draw.rect(s, (30, 70, 90), ar, border_radius=px(10))
        pygame.draw.rect(s, ACCENT, ar, px(2), border_radius=px(10))
        hud.text(s, "+ Ajouter des chansons (A)", 22, (255, 255, 255), ar.center, "center")
        hints = ["Entrée jouer · ↑/↓ choisir · 1-4 joueurs, 5 groupe · C caméra · R réglages · Suppr supprimer · "
                 "Ctrl+F rechercher · F11 plein écran · Échap quitter",
                 "Entrée jouer · 1-4 joueurs, 5 groupe · C caméra · R réglages · Suppr · Ctrl+F · F11 · Échap",
                 "Entrée jouer · 1-5 joueurs · R réglages · Échap quitter"]
        room = W - px(30) - ar.right - px(24)                                     # ne pas chevaucher « Ajouter »
        hint = next((h for h in hints if hud.font(16, False).size(h)[0] <= room), "")
        if hint:
            hud.text(s, hint, 16, (150, 150, 170), (W - px(30), H - px(41)), "midright", bold=False)
        if self.flash and time.perf_counter() - self.flash_t < 3:
            hud.tag(s, self.flash, 22, (255, 255, 255), (W // 2, px(28)), "midtop")
        if self.mode == "add":
            self._draw_add()
        elif self.mode == "settings":
            self._draw_settings()
        elif self.mode == "delete":
            self._modal(["Supprimer « " + display_name(self.songs[self.sel]["meta"])[:50]
                         + " » ?",
                         "Entrée : oui · autre touche : non"])

    def _draw_search(self):
        s, px = self.disp.screen, hud.px
        r = self._search_rect()
        pygame.draw.rect(s, PANEL, r, border_radius=px(10))
        pygame.draw.rect(s, ACCENT if self.searching else (60, 60, 90), r, px(2), border_radius=px(10))
        cy = r.centery
        pygame.draw.circle(s, (150, 150, 170), (r.x + px(22), cy - px(3)), px(8), px(2))      # loupe
        pygame.draw.line(s, (150, 150, 170), (r.x + px(28), cy + px(3)), (r.x + px(34), cy + px(9)), px(3))
        x = r.x + px(48)
        if self.query:
            x = hud.text(s, self.query[-80:], 22, (255, 255, 255), (x, cy), "midleft", bold=False,
                         shadow=False).right
            n = f"{len(self.songs)} / {len(self.all_songs)}"
            hud.text(s, n, 18, (150, 150, 170), (self._clear_rect().x - px(6), cy), "midright", bold=False,
                     shadow=False)
            c, d = self._clear_rect().center, px(7)                                    # croix « effacer »
            for sx in (-1, 1):
                pygame.draw.line(s, (200, 200, 215), (c[0] - d, c[1] - sx * d), (c[0] + d, c[1] + sx * d), px(3))
        elif not self.searching:
            hud.text(s, "Rechercher…  (Ctrl+F)", 22, (120, 120, 140), (x, cy), "midleft", bold=False, shadow=False)
        if self.searching and int(time.perf_counter() * 2) % 2 == 0:                   # curseur clignotant
            pygame.draw.line(s, (255, 255, 255), (x + px(2), cy - px(12)), (x + px(2), cy + px(12)), px(2))

    def _modal(self, lines, h=170):
        s, px = self.disp.screen, hud.px
        W, H = s.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        s.blit(dim, (0, 0))
        box = pygame.Rect(0, 0, min(W - px(60), px(900)), px(h))
        box.center = (W // 2, H // 2)
        pygame.draw.rect(s, PANEL, box, border_radius=px(14))
        pygame.draw.rect(s, ACCENT, box, px(2), border_radius=px(14))
        for k, line in enumerate(lines):
            hud.text(s, line, 24 if k == 0 else 18, (255, 255, 255) if k == 0 else (190, 190, 210),
                     (box.centerx, box.y + px(28) + k * px(40)), "midtop", bold=k == 0)
        return box

    def _draw_add(self):
        links = library.parse_links(self.text)
        last = self.text.strip().splitlines()[-1][:70] if self.text.strip() else "(vide)"
        n_pl = sum(1 for u in links if "list=" in u and not library.youtube_id(u))
        self._modal(["Colle un ou plusieurs liens YouTube (Ctrl+V)",
                     f"{len(links)} lien(s) détecté(s)" + (f", dont {n_pl} playlist(s)" if n_pl else "")
                     + " · les chansons déjà importées seront sautées",
                     last,
                     "Entrée : importer · Échap : annuler"], h=220)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs-dir", default="songs")
    ap.add_argument("--players", type=int, default=1)
    ap.add_argument("--camera", type=int, help="index OpenCV (sinon : caméra choisie dans le menu)")
    ap.add_argument("--model")
    ap.add_argument("--latency-ms", type=float)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--window", default="1280x720")
    ap.add_argument("--fps", type=int)
    a = ap.parse_args()
    settings = cameras.Settings(Path(a.songs_dir).resolve().parent / "settings.json")
    cams = cameras.list_cameras()
    if a.camera is None:
        a.camera = cameras.pick(cams, settings)
    cfg = config_from_args(a)
    apply_settings(cfg, settings, keep_fps=a.fps is not None)
    Path(a.songs_dir).mkdir(exist_ok=True)
    clean_incomplete(a.songs_dir)
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    pygame.key.set_repeat(300, 40)
    disp = Display(a.fullscreen, tuple(int(v) for v in a.window.lower().split("x")), vsync=cfg.vsync)
    sess = Session(cfg)
    menu = Menu(disp, sess, a.songs_dir, settings, cams)
    menu.players = max(1, min(5, a.players))
    try:
        menu.run()
    finally:
        menu.importer.stop()
        sess.close()
        pygame.quit()


if __name__ == "__main__":
    main()
