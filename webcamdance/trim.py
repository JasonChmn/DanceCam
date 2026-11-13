"""Découpage de la partie dansée : bornes manuelles et détection automatique.

Détection : échantillonnage de la source à `rate` img/s (ffmpeg -> rawvideo, indépendant
du codec supporté par OpenCV), pose sur chaque image, une image est « dansée » si une
personne de premier plan est visible en entier ET bouge. On garde le plus long segment
dansé en tolérant de courts trous (pauses, poses figées, coupes de plan).
"""
import re
import subprocess

import numpy as np

from . import media
from .media import ffmpeg_exe
from .skeleton import L_AN, L_SH, R_AN, R_SH
from .tracking import select_people


def parse_time(s):
    """'75', '75.5', '1:15', '1:15.5', '0:01:15' -> secondes (float). None/'' -> None."""
    if s is None or str(s).strip() == "":
        return None
    s = str(s).strip()
    if not re.fullmatch(r"\d+(\.\d+)?(:\d{1,2}(\.\d+)?){0,2}", s):
        raise ValueError(f"temps invalide : {s!r} (attendu : secondes, m:ss ou h:mm:ss)")
    t = 0.0
    for part in s.split(":"):
        t = t * 60 + float(part)
    return t


def fmt_time(t):
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


def find_segment(active, dt, max_gap=4.0, min_len=10.0):
    """active (N,) bool échantillonné tous les dt s -> (début, fin) en s du plus long segment
    actif (trous <= max_gap comblés), ou None s'il n'y en a pas d'au moins min_len s."""
    a = np.asarray(active, bool)
    if not a.any():
        return None
    idx = np.flatnonzero(a)
    # coupe là où l'écart entre deux échantillons actifs dépasse max_gap
    breaks = np.flatnonzero(np.diff(idx) * dt > max_gap)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    t0, t1 = starts[k] * dt, (ends[k] + 1) * dt
    return (t0, t1) if t1 - t0 >= min_len else None


def sample_frames(src, rate, height=360):
    """Générateur d'images BGR (h=height) à `rate` img/s via ffmpeg."""
    ff = ffmpeg_exe()
    # largeur inconnue a priori : on force une largeur paire fixe après mise à l'échelle 16:9 max
    w = height * 16 // 9 // 2 * 2
    vf = f"fps={rate},scale={w}:{height}:force_original_aspect_ratio=decrease,pad={w}:{height}:-1:-1"
    p = subprocess.Popen(media.with_threads([ff, "-loglevel", "error", "-i", str(src), "-an", "-vf", vf,
                                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)   # « Broken pipe » attendu si on arrête de lire
    n = w * height * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def activity(src, pose, cfg, rate=2.0, batch=16):
    """-> (visible (N,) bool, motion (N,) float, dt, présent (N,) bool).
    visible : danseur de premier plan en pied ; présent : danseur de premier plan avec ≥ 5
    keypoints (même critère que la présence des coachs mesurée après extraction).
    motion = déplacement moyen des keypoints entre deux échantillons / hauteur de la personne."""
    vis, mot, pres, prev = [], [], [], None
    frames = []

    def flush():
        nonlocal prev
        for kpts, boxes, conf in pose(frames):
            keep = conf >= 0.5
            kpts, boxes = kpts[keep], boxes[keep]
            idx = select_people(boxes, 4, min_rel_h=0.6)
            H = frames[0].shape[0]
            full = [j for j in idx
                    if (kpts[j, [L_SH, R_SH], 2] >= cfg.kpt_conf).all()
                    and (kpts[j, [L_AN, R_AN], 2] >= cfg.kpt_conf).any()
                    and boxes[j, 3] - boxes[j, 1] >= 0.25 * H]
            vis.append(bool(full))
            pres.append(any((kpts[j, :, 2] >= cfg.kpt_conf).sum() >= 5 and boxes[j, 3] - boxes[j, 1] >= 0.25 * H
                            for j in idx))
            m = 0.0
            if full:
                j = max(full, key=lambda j: boxes[j, 3] - boxes[j, 1])
                kp, h = kpts[j], boxes[j, 3] - boxes[j, 1]
                if prev is not None:
                    ok = (kp[:, 2] >= cfg.kpt_conf) & (prev[0][:, 2] >= cfg.kpt_conf)
                    if ok.sum() >= 6:
                        m = float(np.linalg.norm(kp[ok, :2] - prev[0][ok, :2], axis=1).mean() / h)
                prev = (kp, h)
            else:
                prev = None
            mot.append(m)
        frames.clear()

    for f in sample_frames(src, rate):
        frames.append(f)
        if len(frames) == batch:
            flush()
    if frames:
        flush()
    return np.array(vis, bool), np.array(mot, np.float32), 1.0 / rate, np.array(pres, bool)


def safe_segment(active, dt, seg, min_len=60.0, min_frac=0.4, pad=1.0):
    """Garde-fou : un segment trop court (danse vue par intermittence : coach mal détecté,
    coupes de plan…) couperait la chanson. -> (segment ou None = vidéo entière, incertain ?)
    Repli : de la première à la dernière frame dansée, sinon la vidéo entière."""
    duration = len(active) * dt
    need = min(min_len, 0.5 * duration)
    if seg and seg[1] - seg[0] >= need and seg[1] - seg[0] >= min_frac * duration:
        return seg, False
    idx = np.flatnonzero(active)
    if len(idx):
        span = (max(0.0, idx[0] * dt - pad), min(duration, (idx[-1] + 1) * dt + pad))
        if span[1] - span[0] >= need:
            return span, True
    return None, True


def detect(src, pose, cfg, rate=2.0, motion_thr=0.04, pad=1.0):
    """-> (segment (t0, t1) ou None, active (N,) bool, dt, présent (N,) bool).
    pad : marge ajoutée de chaque côté.

    motion_thr calé sur une vidéo Just Dance : intro figée ~0.02-0.04, danse 0.05-0.6.
    """
    vis, mot, dt, pres = activity(src, pose, cfg, rate)
    # mouvement lissé sur ~2 s : une pose tenue au milieu d'un enchaînement reste « dansée »
    k = max(1, int(round(2.0 / dt)))
    mot_s = np.convolve(mot, np.ones(k) / k, mode="same") if len(mot) else mot
    active = vis & (mot_s >= motion_thr)
    seg = find_segment(active, dt)
    if seg:
        seg = (max(0.0, seg[0] - pad), min(len(active) * dt, seg[1] + pad))
    return seg, active, dt, pres


def timeline(active, dt, seg, width=64):
    """Frise texte : '█' dansé, '·' non dansé, '^' sous la sélection."""
    n = len(active)
    if n == 0:
        return ""
    cells = np.array_split(np.arange(n), min(width, n))
    bar = "".join("█" if active[c].mean() >= 0.5 else "·" for c in cells)
    total = n * dt
    sel = ""
    if seg:
        sel = "".join("^" if seg[0] <= c[0] * dt < seg[1] else " " for c in cells)
    return f"0:00 {bar} {fmt_time(total)}\n     {sel}"


def grab(src, t, height=270):
    """Image BGR de la source au temps t (s), ou None."""
    w = height * 16 // 9 // 2 * 2
    vf = f"scale={w}:{height}:force_original_aspect_ratio=decrease,pad={w}:{height}:-1:-1"
    r = subprocess.run(media.with_threads([ffmpeg_exe(), "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
                                           "-i", str(src), "-frames:v", "1", "-vf", vf, "-f", "rawvideo",
                                           "-pix_fmt", "bgr24", "-"]),
                       capture_output=True)
    n = w * height * 3
    return np.frombuffer(r.stdout[:n], np.uint8).reshape(height, w, 3) if len(r.stdout) >= n else None


def preview(src, start, end):
    """Fenêtre OpenCV : 1 s avant / au début / à la fin / 1 s après. Une touche pour fermer."""
    import cv2
    shots = [(start - 1, "avant debut"), (start + 0.5, "DEBUT"), (end - 0.5, "FIN"), (end + 1, "apres fin")]
    tiles = []
    for t, lab in shots:
        f = grab(src, t)
        f = np.zeros((270, 480, 3), np.uint8) if f is None else f.copy()
        cv2.putText(f, f"{lab} {fmt_time(t)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(f)
    img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imshow("Apercu des bornes (une touche pour fermer)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ask_bounds(src, start, end, duration):
    """Boucle de confirmation interactive. -> (start, end) ; None = pas de coupe."""
    while True:
        if start is None and end is None:
            print("  Sélection : vidéo entière")
        else:
            s, e = start or 0.0, end or duration
            print(f"  Sélection : {fmt_time(s)} → {fmt_time(e)}  ({fmt_time(e - s)})")
        r = input("  [Entrée] valider · [a] aperçu · « début fin » corriger (ex. 0:16 3:59, '-' = inchangé)"
                  " · [t] vidéo entière : ").strip().lower()
        if r == "":
            return start, end
        if r == "t":
            start = end = None
            continue
        if r == "a":
            try:
                preview(src, start or 0.0, end or duration)
            except Exception as ex:  # pas d'affichage disponible
                print(f"  aperçu impossible : {ex}")
            continue
        parts = r.split()
        try:
            vals = [None if p == "-" else parse_time(p) for p in parts]
        except ValueError as ex:
            print(f"  {ex}")
            continue
        if len(vals) != 2:
            print("  il faut deux valeurs : début fin")
            continue
        ns = vals[0] if vals[0] is not None else start
        ne = vals[1] if vals[1] is not None else end
        if ns is not None and ne is not None and ne <= ns:
            print("  la fin doit être après le début")
            continue
        start, end = ns, ne


def coach_band(K, height, thr=0.35, q=1.0, margin=0.02, min_frac=0.4):
    """Bandeau vertical pleine largeur couvrant les coachs sur toute la chanson.

    K (T,N,17,3) en pixels vidéo -> (y0, y1) en fraction de la hauteur. Par frame et coach :
    haut = min(keypoint le plus haut - 4 % corps (doigts au-delà du poignet),
               sommet du crâne ~ nez - 0.6 * (épaules - nez)),
    bas = keypoint le plus bas + 4 % corps (pieds sous les chevilles).
    Percentiles q / 100-q : ignore les extrêmes isolés (saut, faux positif). + marge,
    hauteur minimale min_frac. Ce qui reste du HUD d'origine est atténué par l'assombrissement.
    """
    y, c = K[..., 1], K[..., 2] >= thr
    ok = c.sum(-1) >= 5                                      # (T,N) pose exploitable
    if not ok.any():
        return 0.0, 1.0
    big, small = np.where(c, y, -np.inf).max(-1), np.where(c, y, np.inf).min(-1)
    h = big - small
    top = small - 0.04 * h
    head = c[..., 0] & c[..., 5] & c[..., 6]
    crown = y[..., 0] - 0.6 * (0.5 * (y[..., 5] + y[..., 6]) - y[..., 0])
    top = np.where(head, np.minimum(top, crown), top)[ok]
    bot = (big + 0.04 * h)[ok]
    y0 = max(0.0, np.percentile(top, q) / height - margin)
    y1 = min(1.0, np.percentile(bot, 100 - q) / height + margin)
    if y1 - y0 < min_frac:                                   # élargit autour du centre
        mid = min(max((y0 + y1) / 2, min_frac / 2), 1 - min_frac / 2)
        y0, y1 = mid - min_frac / 2, mid + min_frac / 2
    return round(float(y0), 4), round(float(y1), 4)
