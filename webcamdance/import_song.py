"""Pipeline offline : source (fichier ou URL) -> songs/<nom>/{video.mp4, audio.ogg, poses.npz, meta.json}

    python -m webcamdance.import_song chemin/ou/url [--name NOM] [--coaches N]
    python -m webcamdance.import_song URL1 URL2 …            (lot : déjà importées sautées)
    python -m webcamdance.import_song --playlist URL_PLAYLIST
    python -m webcamdance.import_song --links-file liens.txt  (un lien par ligne)
"""
import argparse
import json
import re
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import coach_id, library, media, trim
from .config import Config
from .songinfo import song_info
from .media import crop_encode, download, expand_playlist, normalize, video_title
from .pose import PoseEstimator
from .appearance import signatures
from .tracking import centers_scales, select_people


# Verrous des imports parallèles (import_list --jobs : remplacés par des verrous inter-processus) :
# créneaux de téléchargement YouTube, et écritures communes (noms de dossiers, liste des rejets).
LOCKS = SimpleNamespace(download=threading.BoundedSemaphore(1), fs=threading.Lock())


def smooth(K):
    """Moyenne glissante (±1 frame) pondérée par la confiance, in-place sur x,y."""
    c = K[..., 2:3]
    num, den = K[..., :2] * c, c.copy()
    num[1:] += K[:-1, ..., :2] * c[:-1]; den[1:] += c[:-1]
    num[:-1] += K[1:, ..., :2] * c[1:]; den[:-1] += c[1:]
    K[..., :2] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), K[..., :2])


def detect_all(video_path, cfg, batch=8):
    """Passe 1 : poses + signatures couleur de chaque détection, pour toute la vidéo.
    -> (frames, fps, largeur)."""
    pose = PoseEstimator(cfg.offline_model, cfg)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.video_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frames, done = [], False
    while not done:
        imgs = []
        while len(imgs) < batch:
            ok, f = cap.read()
            if not ok:
                done = True
                break
            imgs.append(f)
        if not imgs:
            break
        for img, (kpts, boxes, conf) in zip(imgs, pose(imgs)):
            idx = select_people(boxes, 8)                   # candidats (arrière-plan écarté)
            kpts, boxes, conf = kpts[idx], boxes[idx], conf[idx]
            cen, sc = centers_scales(kpts, boxes, cfg.kpt_conf)
            h = boxes[:, 3] - boxes[:, 1]
            fg = (conf >= 0.5) & (h >= 0.6 * h.max()) if len(h) else np.zeros(0, bool)
            frames.append({"k": kpts, "cen": cen, "sc": sc, "fg": fg, "ovl": coach_id.overlap_frac(boxes),
                           "sig": signatures(img, kpts, cfg.kpt_conf)})
        if total and len(frames) % max(1, total // 10) < batch:
            print(f"[extract] {100 * len(frames) // max(total, 1)} %")
    cap.release()
    return frames, fps, W


def extract(video_path, out_npz, cfg, n_coaches=0, batch=8):
    """Passe 1 : détections + signatures. Passe 2 : identités des coachs (coach_id.identify)."""
    frames, fps, W = detect_all(video_path, cfg, batch)
    n = n_coaches or coach_id.count_coaches(frames)
    assign, info = coach_id.identify(frames, n, fps)
    K = np.zeros((len(frames), n, 17, 3), np.float32)
    for t, f in enumerate(frames):
        for c, j in enumerate(assign[t]):
            if j >= 0:
                K[t, c] = f["k"][j]
    pres = (assign >= 0).mean(0)
    print(f"[extract] {n} coach(s), {len(K)} frames @ {fps:.2f} fps, {info}, présence "
          + " ".join(f"{p * 100:.0f}%" for p in pres))
    smooth(K)
    np.savez_compressed(out_npz, kpts=K, fps=np.float32(fps))
    return n, fps, len(K)


def finalize_video(full, out, cfg, band_override=None):
    """Recadre la vidéo sur le bandeau coach + marge (le reste n'est jamais affiché) et
    ramène les poses dans ce repère. -> (bandeau dans la vidéo finale, recadrage dans
    l'originale), en fractions de hauteur."""
    npz = out / "poses.npz"
    d = np.load(npz)
    K, fps = d["kpts"], d["fps"]
    H = int(cv2.VideoCapture(str(full)).get(cv2.CAP_PROP_FRAME_HEIGHT))
    tight = band_override or trim.coach_band(K, H, cfg.kpt_conf)
    y0 = int(max(0.0, tight[0] - cfg.band_margin) * H) // 2 * 2       # pair (yuv420)
    y1 = min(H, int(np.ceil(min(1.0, tight[1] + cfg.band_margin) * H / 2)) * 2)
    crop_encode(full, out / "video.mp4", y0, y1, cfg.video_crf)
    K = K.copy()
    K[..., 1] = np.where(K[..., 2] > 0, K[..., 1] - y0, 0.0)
    np.savez_compressed(npz, kpts=K, fps=fps)
    h = y1 - y0
    band = (round(max(0.0, (tight[0] * H - y0) / h), 4), round(min(1.0, (tight[1] * H - y0) / h), 4))
    return band, (round(y0 / H, 4), round(y1 / H, 4))


def choose_bounds(src, a, cfg):
    """Bornes de la partie dansée : manuelles > détection auto, confirmées si terminal."""
    """-> (début, fin, incertain ?). None = borne non coupée."""
    if a.no_trim:
        return None, None, False
    start, end = trim.parse_time(a.start), trim.parse_time(a.end)
    if start is not None and end is not None:
        return start, end, False
    print("[import] détection de la partie dansée…", flush=True)
    # gros modèle : certains coachs stylisés (néon, aplats) ne sont vus qu'à ~40 % par le petit
    seg, active, dt, pres = trim.detect(src, PoseEstimator(cfg.offline_model, cfg), cfg)
    duration = len(active) * dt
    print(trim.timeline(active, dt, seg))
    seg, uncertain = trim.safe_segment(active, dt, seg)
    # rejet anticipé : coach quasi invisible dès l'échantillonnage (2 img/s) -> inutile d'aller
    # jusqu'à l'extraction complète. Marge de 15 points : la mesure exacte tranche ensuite.
    i0, i1 = (int(seg[0] / dt), int(seg[1] / dt) + 1) if seg else (0, len(pres))
    est = float(pres[i0:i1].mean()) if i1 > i0 else 0.0
    if est < cfg.min_coach_presence - 0.15:
        raise CoachNotDetected(f"coach détecté ~{est:.0%} du temps (estimation rapide, minimum "
                               f"{cfg.min_coach_presence:.0%})")
    if seg is None:
        print("[import] partie dansée non trouvée : vidéo entière gardée (coupe à vérifier)")
    else:
        print(f"[import] danse détectée : {trim.fmt_time(seg[0])} → {trim.fmt_time(seg[1])}"
              + (" (coupe à vérifier)" if uncertain else ""))
        start = seg[0] if start is None else start
        end = seg[1] if end is None else end
    if sys.stdin.isatty() and not a.yes:
        start, end = trim.ask_bounds(src, start, end, duration)
        uncertain = False
    return start, end, uncertain


def short_name(title, root):
    """Titre -> nom de dossier court et unique : « Just Dance 2018: Waka Waka (This Time For
    Africa) - 4 players » -> waka_waka."""
    t = re.sub(r"(?i)^\s*just\s*dance\s*\(?(\d{1,4}|unlimited|now|kids|wii|summer\s*party)?\)?\s*[:\-|–]*\s*", "", title)
    t = re.sub(r"[(\[].*?[)\]]", " ", t)                       # (feat. …), [Official], …
    t = re.sub(r"(?i)\s[-|–]\s.*$", "", t)                       # « - 4 players », « | 5 stars »…
    t = re.sub(r"[^\w]+", "_", t.lower()).strip("_")[:40].strip("_") or "song"
    name, k = t, 2
    while (Path(root) / name).exists():
        name, k = f"{t}_{k}", k + 1
    return name


class CoachNotDetected(Exception):
    """Coachs trop peu détectés pour que la chanson soit jouable (costumes très stylisés…)."""


class Excluded(CoachNotDetected):
    """Titre exclu par le filtre (config.exclude_title) : sauté avant téléchargement."""


def coach_presence(K, thr):
    """Part moyenne (sur les coachs) des frames où le coach est détecté (≥ 5 keypoints)."""
    return float(presence_per_coach(K, thr).mean())


def presence_per_coach(K, thr):
    return ((K[..., 2] >= thr).sum(-1) >= 5).mean(0)


def import_one(source, a, cfg, name=None, dl_hook=None):
    """Importe une source (fichier ou URL). -> dossier de la chanson.
    dl_hook : callback de progression yt-dlp (barre de progression d'import_list)."""
    src, title, vid = source, Path(source).stem, None
    if re.match(r"^https?://", source):
        if not LOCKS.download.acquire(False):       # trop de téléchargements simultanés : on attend
            print("[import] attente d'un créneau de téléchargement", flush=True)
            LOCKS.download.acquire()
        try:
            if cfg.exclude_title:                   # filtre sur le titre, avant de télécharger
                t, i = video_title(source)
                if re.search(cfg.exclude_title, t):
                    with LOCKS.fs:
                        library.add_rejected(a.songs_dir, i, t, "exclue : filtre sur le titre")
                    print(f"[import] titre : {t} -> exclue", flush=True)
                    raise Excluded(f"exclue (titre : {t[:60]})")
            print(f"[import] téléchargement : {source}", flush=True)
            src, title, vid = download(source, Path(a.songs_dir) / "_cache", hook=dl_hook)
        finally:
            LOCKS.download.release()
    if not Path(src).is_file():
        raise FileNotFoundError(f"source introuvable : {src}")
    with LOCKS.fs:                                  # choix du nom + création : atomique entre imports
        out = Path(a.songs_dir) / (name or short_name(title, a.songs_dir))
        out.mkdir(parents=True, exist_ok=True)
    print(f"[import] titre : {title} -> {out}", flush=True)
    def reject(e):
        with LOCKS.fs:
            library.add_rejected(a.songs_dir, vid, title, str(e))
        if src != source:
            Path(src).unlink(missing_ok=True)
        raise e

    try:
        try:
            start, end, uncertain = choose_bounds(src, a, cfg)
        except CoachNotDetected as e:
            reject(e)
        print(f"[import] normalisation -> {out}", flush=True)
        full = out / "_full.mp4"                    # intermédiaire de bonne qualité, supprimé à la fin
        normalize(src, out, cfg.video_height, cfg.video_fps, start, end, video_name=full.name)
        n, fps, T = extract(full, out / "poses.npz", cfg, a.coaches)
        per = presence_per_coach(np.load(out / "poses.npz")["kpts"], cfg.kpt_conf)
        pres = float(per.mean())
        if pres < cfg.min_coach_presence:
            reject(CoachNotDetected(f"coach détecté {pres:.0%} du temps (minimum {cfg.min_coach_presence:.0%})"))
        band, crop = finalize_video(full, out, cfg, a.band)
        full.unlink()
        print(f"[import] bandeau coach : {band[0]:.3f} → {band[1]:.3f} (vidéo recadrée sur {crop[0]:.3f} → "
              f"{crop[1]:.3f} de l'original ; resserrer : python tools/set_band.py {out} --band Y0 Y1 --preview)")
        name = song_info(title, vid, a.songs_dir)
        (out / "meta.json").write_text(json.dumps(
            {"title": title, "artist": name["artist"], "song": name["song"], "version": name["version"],
             "name_source": name["source"], "source": source, "id": vid, "n_coaches": n, "fps": fps,
             "duration": T / fps, "model": cfg.offline_model, "trim": [start, end],
             "trim_uncertain": uncertain,
             "coach_presence": round(pres, 3), "coach_presence_per": [round(float(p), 3) for p in per],
             "band": list(band), "crop": list(crop)},
            indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        if not (out / "meta.json").exists():        # import raté : pas de dossier à moitié fait
            shutil.rmtree(out, ignore_errors=True)
        raise
    if src != source and not a.keep_source:         # source téléchargée : inutile une fois l'import réussi
        Path(src).unlink(missing_ok=True)
    print(f"[import] OK : {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="fichier(s) vidéo local / URL(s) (yt-dlp)")
    ap.add_argument("--playlist", action="store_true", help="les URL sont des playlists : importer chaque vidéo")
    ap.add_argument("--links-file", action="store_true", help="les sources sont des fichiers texte de liens")
    ap.add_argument("--name", help="nom du dossier (une seule source)")
    ap.add_argument("--coaches", type=int, default=0, help="0 = auto")
    ap.add_argument("--model")
    ap.add_argument("--songs-dir", default="songs")
    ap.add_argument("--height", type=int)
    ap.add_argument("--keep-source", action="store_true", help="garder la vidéo téléchargée (songs/_cache)")
    ap.add_argument("--start", help="début de la danse (s, m:ss ou h:mm:ss) ; défaut : détection auto")
    ap.add_argument("--end", help="fin de la danse ; défaut : détection auto")
    ap.add_argument("--no-trim", action="store_true", help="garder la vidéo entière")
    ap.add_argument("-y", "--yes", action="store_true", help="accepter la détection sans confirmation")
    ap.add_argument("--cookies-from-browser", metavar="NAVIGATEUR", help="session YouTube du navigateur (anti-robot)")
    ap.add_argument("--band", nargs=2, type=float, metavar=("Y0", "Y1"),
                    help="bandeau coach affiché (fractions de hauteur, ex. 0.18 0.96) ; défaut : auto")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="threads CPU (défaut : moitié des cœurs ; ffmpeg, OpenCV, PyTorch)")
    a = ap.parse_args()
    media.limit_threads(a.cpu_threads or media.default_threads())
    if a.cookies_from_browser:
        media.YT_OPTS["cookiesfrombrowser"] = (a.cookies_from_browser,)
    if sys.platform != "win32":     # arrêt demandé par le menu : nettoyer l'import en cours
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    cfg = Config()
    if a.model:
        cfg.offline_model = a.model
    if a.height:
        cfg.video_height = a.height

    sources = a.sources
    if a.links_file:
        sources = [u for f in sources for u in library.parse_links(Path(f).read_text(encoding="utf-8"))]
    if a.playlist:
        sources = [u for url in sources for u in expand_playlist(url)]
    if len(sources) == 1 and not (a.playlist or a.links_file):
        try:
            import_one(sources[0], a, cfg, a.name)
        except FileNotFoundError as e:
            sys.exit(f"[import] {e}")
        return
    # lot : doublons et chansons déjà importées sautés, un échec n'arrête pas la suite
    a.yes = True
    done = library.imported_ids(a.songs_dir) | set(library.load_rejected(a.songs_dir))
    ok, skipped, failed, rejected = [], [], [], []
    todo = library.parse_links(" ".join(s for s in sources if re.match(r"^https?://", s)))
    todo += [s for s in sources if not re.match(r"^https?://", s)]
    for k, src in enumerate(todo, 1):
        print(f"[lot] {k}/{len(todo)} {src}", flush=True)
        if library.youtube_id(src) in done:
            skipped.append(src)
            print("[lot] déjà importée, sautée", flush=True)
            continue
        try:
            ok.append(import_one(src, a, cfg))
            done.add(library.youtube_id(src))
        except KeyboardInterrupt:
            raise
        except media.YoutubeBlocked as e:           # tout le reste échouerait : on arrête
            print(f"[lot] ARRÊT : {e}", flush=True)
            break
        except CoachNotDetected as e:               # injouable : noté, jamais retenté
            rejected.append(src)
            print(f"[lot] REJETÉE {src} : {e}", flush=True)
        except Exception as e:                      # vidéo privée, supprimée, réseau, etc.
            failed.append((src, f"{type(e).__name__}: {e}"))
            print(f"[lot] ÉCHEC {src} : {e}", flush=True)
    print(f"[lot] terminé : {len(ok)} importée(s), {len(skipped)} déjà présente(s), {len(rejected)} rejetée(s), "
          f"{len(failed)} échec(s)")
    for src, err in failed:
        print(f"  ÉCHEC {src} : {err[:200]}")


if __name__ == "__main__":
    main()
