"""ffmpeg / yt-dlp, lecteur vidéo threadé, horloge chanson."""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


THREADS = 0     # threads CPU par processus d'import (ffmpeg, OpenCV, PyTorch) ; 0 = auto (tous les cœurs)


def default_threads(jobs=1):
    """Moitié des cœurs répartie entre les imports simultanés : sans limite, chaque bibliothèque
    lance un thread par cœur dans chaque processus et la machine sature (chauffe) sans gain."""
    return max(2, (os.cpu_count() or 4) // (2 * max(1, jobs)))


def limit_threads(n):
    """Limite les threads CPU du processus courant (et des ffmpeg qu'il lance) à n."""
    global THREADS
    THREADS = n
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(n)          # hérité par les processus enfants, lu au chargement des bibliothèques
    cv2.setNumThreads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def ff_threads():
    """Options ffmpeg (à placer avant la sortie) limitant ses threads."""
    return ["-threads", str(THREADS), "-filter_threads", str(THREADS)] if THREADS else []


def with_threads(cmd):
    """Commande ffmpeg avec threads limités au décodage (avant chaque -i) et à l'encodage
    (avant la sortie, dernier argument)."""
    if not THREADS:
        return cmd
    out = []
    for a in cmd[:-1]:
        if a == "-i":
            out += ["-threads", str(THREADS)]
        out.append(a)
    return out + ff_threads() + cmd[-1:]


def run_ffmpeg(cmd):
    """ffmpeg avec stderr capturé : ses messages vont dans l'exception (et donc le journal),
    jamais directement dans le terminal où ils casseraient la barre de progression."""
    r = subprocess.run(with_threads(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué ({r.returncode}) : {r.stderr.strip()[-300:]}")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg introuvable (pip install imageio-ffmpeg)")
        return exe


# deno (moteur JS requis par yt-dlp pour YouTube, installé par pip dans le venv) : le dossier
# de l'interpréteur (.venv/bin ou .venv\Scripts) n'est dans le PATH que si le venv est activé
_bin = os.path.dirname(sys.executable)
if _bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

YT_OPTS = {}    # options yt-dlp communes (ex. {"cookiesfrombrowser": ("firefox",)} contre le blocage anti-robot)


class YoutubeBlocked(RuntimeError):
    """YouTube demande une connexion (« confirm you're not a bot ») : inutile de continuer."""


def _yt_error(e):
    if "not a bot" in str(e) or "Sign in to confirm" in str(e):
        return YoutubeBlocked("YouTube bloque temporairement les téléchargements (anti-robot). Attendre quelques "
                              "heures, ou relancer avec --cookies-from-browser firefox (ou chrome).")
    return e


def download(url, cache_dir, max_height=720, hook=None):
    """Télécharge via yt-dlp (cache local). -> (chemin, titre, identifiant).
    max_height : inutile de récupérer plus que la résolution finale stockée."""
    import sys
    import yt_dlp
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    opts = {"format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
            "outtmpl": str(cache_dir / "%(id)s.%(ext)s"), "merge_output_format": "mkv",
            "ffmpeg_location": ffmpeg_exe(), "noplaylist": True,
            "noprogress": not sys.stdout.isatty()}          # pas de barre de progression dans le menu
    if hook:
        opts["progress_hooks"] = [hook]
    opts.update(YT_OPTS)
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise _yt_error(e) from e
        dl = info.get("requested_downloads") or []
        path = dl[0]["filepath"] if dl else ydl.prepare_filename(info)
    return path, info.get("title", info.get("id", "song")), info.get("id")


def video_title(url):
    """Titre d'une vidéo sans la télécharger (≈ 1 s)."""
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True, "skip_download": True, **YT_OPTS}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            raise _yt_error(e) from e
    return info.get("title", ""), info.get("id")


def expand_playlist(url):
    """URL de playlist -> liste des URL de ses vidéos (sans rien télécharger)."""
    import yt_dlp
    with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, **YT_OPTS}) as ydl:
        info = ydl.extract_info(url, download=False)
    return [e.get("url") or f"https://www.youtube.com/watch?v={e['id']}" for e in info.get("entries") or []
            if e and (e.get("url") or e.get("id"))]


def normalize(src, out_dir, height, fps, start=None, end=None, crf=18, video_name="video.mp4"):
    """-> video_name (H.264, CFR, sans audio) + audio.ogg. CFR => index frame = t * fps.
    crf 18 par défaut : vidéo intermédiaire de bonne qualité pour l'extraction des poses.

    start/end (s) : découpe identique vidéo et audio (-ss avant -i + ré-encodage = coupe
    précise à l'échantillon), donc t=0 du jeu = start dans la source.
    """
    out_dir = Path(out_dir)
    ff = ffmpeg_exe()
    cut = (["-ss", f"{start:.3f}"] if start else []) + \
          (["-t", f"{end - (start or 0.0):.3f}"] if end else [])
    run_ffmpeg([ff, "-y", "-loglevel", "error", *cut, "-i", str(src), "-an",
                    "-vf", f"scale=-2:{height},fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", str(out_dir / video_name)])
    run_ffmpeg([ff, "-y", "-loglevel", "error", *cut, "-i", str(src), "-vn",
                    "-c:a", "libvorbis", "-q:a", "4", str(out_dir / "audio.ogg")])


def crop_encode(src, out, y0, y1, crf):
    """Encodage final : lignes [y0, y1) seulement (pleine largeur), H.264 crf, fps conservé."""
    run_ffmpeg([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-an",
                    "-vf", f"crop=iw:{y1 - y0}:0:{y0}", "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)])


def spotlight(f, kpts, scale, y0, thr, dim):
    """Assombrit l'image hors des coachs. f : image bandeau déjà redimensionnée (modifiée en place).
    kpts (N,17,3) en pixels vidéo source ; scale, y0 : source -> bandeau (x*scale, (y-y0)*scale)."""
    h, w = f.shape[:2]
    r = 8                                              # masque calculé à 1/8 puis flouté
    m = np.zeros((h // r + 1, w // r + 1), np.float32)
    for kp in kpts:
        ok = kp[:, 2] >= thr
        if ok.sum() < 3:
            continue
        x, y = kp[ok, 0] * scale / r, (kp[ok, 1] - y0) * scale / r
        bh = max(y.max() - y.min(), 1.0)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        ax, ay = (x.max() - x.min()) / 2 + 0.25 * bh, bh / 2 + 0.2 * bh
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    g = ((dim + (1 - dim) * np.clip(m, 0, 1)) * 255).astype(np.uint8)
    cv2.multiply(f, cv2.merge([g, g, g]), dst=f, scale=1 / 255)
    return f


def upscale(f, size, sharpen=0.35):
    """Redimensionne pour l'affichage : Lanczos + léger masque flou (netteté) à l'agrandissement,
    INTER_AREA à la réduction. Le style cartoon des vidéos de danse supporte bien la netteté."""
    if size[0] <= f.shape[1]:
        return cv2.resize(f, size, interpolation=cv2.INTER_AREA)
    f = cv2.resize(f, size, interpolation=cv2.INTER_LANCZOS4)
    if sharpen > 0:
        blur = cv2.GaussianBlur(f, (0, 0), 1.2)
        f = cv2.addWeighted(f, 1 + sharpen, blur, -sharpen, 0)
    return f


class VideoPlayer(threading.Thread):
    """Décode en tâche de fond la frame correspondant à l'horloge ; saute les frames en retard.

    band (y0, y1) : bandeau affiché en fraction de hauteur (pleine largeur). coach_kpts
    (T,N,17,3) : si fourni, le décor hors coachs est assombri (facteur dim).
    Coordonnées source -> image affichée : (x * scale, (y - y0_px) * scale).
    """

    def __init__(self, path, fit_size, band=(0.0, 1.0), coach_kpts=None, thr=0.35, dim=0.4):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        vw, vh = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.src_w = vw
        self.y0, self.y1 = int(band[0] * vh), max(int(band[0] * vh) + 2, int(band[1] * vh))
        self.set_fit(fit_size)
        self.coach_kpts, self.thr, self.dim = coach_kpts, thr, dim
        self.target, self.idx = 0, -1
        self.frame, self.frame_id = None, -1
        self._halt = False

    def set_fit(self, fit_size):
        """(Re)calcule la taille d'affichage (changement de résolution / plein écran)."""
        bh = self.y1 - self.y0
        s = min(fit_size[0] / self.src_w, fit_size[1] / bh)
        self.size = (int(self.src_w * s) // 2 * 2, int(bh * s) // 2 * 2)
        self.scale = self.size[0] / self.src_w

    def set_time(self, t):
        self.target = int(t * self.fps)

    def stop(self):
        self._halt = True

    def run(self):
        while not self._halt:
            tgt = self.target
            if self.idx >= tgt:
                time.sleep(0.002)
                continue
            while self.idx < tgt - 1 and self.cap.grab():
                self.idx += 1
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self.idx += 1
            f = upscale(f[self.y0:self.y1], self.size)
            if self.coach_kpts is not None and self.dim < 1 and self.idx < len(self.coach_kpts):
                spotlight(f, self.coach_kpts[self.idx], self.scale, self.y0, self.thr, self.dim)
            self.frame, self.frame_id = f, self.idx
        self.cap.release()


class SongClock:
    """Horloge maître, ancrée au démarrage de l'audio (perf_counter)."""

    def __init__(self):
        self.t0 = self.paused_at = None

    def start(self):
        import pygame
        pygame.mixer.music.play()
        self.t0 = time.perf_counter()

    def time(self):
        if self.t0 is None:
            return 0.0
        return (self.paused_at or time.perf_counter()) - self.t0

    def toggle_pause(self):
        import pygame
        if self.paused_at is None:
            self.paused_at = time.perf_counter()
            pygame.mixer.music.pause()
        else:
            self.t0 += time.perf_counter() - self.paused_at
            self.paused_at = None
            pygame.mixer.music.unpause()
