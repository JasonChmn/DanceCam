"""Signature couleur des personnes, pour les identifier (coachs et joueurs).

Masque grossier tiré des keypoints : polygone du torse + traits épais le long des membres +
disque de tête (cheveux, coiffe), en trois parties (haut = torse + bras, bas = jambes, tête) :
deux coachs au même maillot se distinguent souvent par la coiffure.
Les pixels couverts par plusieurs personnes sont exclus (on ne mesure pas celui de devant).
Signature = racine des histogrammes HSV normalisés : la distance euclidienne entre deux
signatures est alors la distance de Hellinger (0 = identiques, 1 = disjointes).
"""
import cv2
import numpy as np

from .skeleton import L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, R_AN, R_EL, R_HIP, R_KN, R_SH, R_WR

UPPER = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)]
LOWER = [(L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN)]
TORSO = [L_SH, R_SH, R_HIP, L_HIP]
H_BINS, S_BINS, V_BINS = 12, 3, 3
PART = H_BINS * S_BINS * V_BINS
NPART = 3
DIM = NPART * PART


def _masks(kp, shape, thr, s):
    """kp (17,3) en pixels source, s = facteur d'échelle -> (haut, bas, tête) uint8 ou None."""
    ok = kp[:, 2] >= thr
    xy = kp[:, :2] * s
    if ok[L_SH] and ok[R_SH]:
        w = np.linalg.norm(xy[L_SH] - xy[R_SH])
    elif ok[L_HIP] and ok[R_HIP]:
        w = 1.3 * np.linalg.norm(xy[L_HIP] - xy[R_HIP])
    else:
        return None, None, None
    th = max(2, int(0.28 * w))
    up, lo, hd = (np.zeros(shape, np.uint8) for _ in range(3))
    face = [i for i in range(5) if ok[i]]
    if face:                                      # disque centré un peu au-dessus du visage : cheveux
        c = xy[face].mean(0) - [0, 0.25 * w]
        cv2.circle(hd, tuple(c.astype(int)), max(2, int(0.45 * w)), 1, -1)
    if ok[TORSO].all():
        cv2.fillConvexPoly(up, xy[TORSO].astype(np.int32), 1)
    for m, bones in ((up, UPPER), (lo, LOWER)):
        for a, b in bones:
            if ok[a] and ok[b]:
                cv2.line(m, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), 1, th)
    return up, lo, hd


def _hist(hsv, m):
    n = int(m.sum())
    if n < 12:
        return None
    h = cv2.calcHist([hsv], [0, 1, 2], m, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(h / n)


def signatures(frame, kpts, thr, s=0.25):
    """frame BGR, kpts (M,17,3) -> (M, DIM) float32 ; ligne de NaN si non mesurable.

    s : sous-échantillonnage (0.25 sur du 720p ~ 320x180, < 1 ms pour quelques personnes).
    Une partie absente (jambes hors champ, tête cachée) est remplacée par la moyenne des autres.
    """
    M = len(kpts)
    out = np.full((M, DIM), np.nan, np.float32)
    if M == 0:
        return out
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    shape = hsv.shape[:2]
    masks = [_masks(kp, shape, thr, s) for kp in kpts]
    cover = np.zeros(shape, np.uint8)
    for ms in masks:
        if ms[0] is not None:
            cover += ((ms[0] | ms[1] | ms[2]) > 0).astype(np.uint8)
    alone = (cover <= 1).astype(np.uint8)
    for i, ms in enumerate(masks):
        if ms[0] is None:
            continue
        hs = [_hist(hsv, m & alone) for m in ms]
        got = [h for h in hs if h is not None]
        if not got:
            continue
        fill = np.mean(got, 0)
        fill /= max(np.linalg.norm(fill), 1e-6)
        for k, h in enumerate(hs):
            out[i, k * PART:(k + 1) * PART] = h if h is not None else fill
    return out


def distance(a, b):
    """Distance de Hellinger moyenne (quadratique) sur les parties, dans [0,1]."""
    d2 = ((a - b) ** 2).sum(-1)
    return np.sqrt(0.5 * d2 / NPART)
