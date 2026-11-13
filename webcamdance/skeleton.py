"""Squelette COCO-17 et features d'os (vecteurs unitaires 2D)."""
import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = range(5, 17)

# os scorés : bras, avant-bras, cuisses, tibias (+ tronc ajouté en dernier)
BONES = np.array([(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
                  (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN)])
BONE_W = np.array([1.2, 1.5, 1.2, 1.5, 1.0, 1.0, 1.0, 1.0, 0.8], np.float32)
N_BONES = len(BONE_W)

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


def bone_features(kpts, thr, min_len=0.12):
    """kpts (...,17,3) [x,y,conf] -> U (...,9,2) unitaires (0 si invalide), V (...,9) bool.

    Invariant translation/échelle/longueurs de membres. Un os trop raccourci en
    projection (< min_len * tronc) est invalidé : direction 2D non fiable (membre vers la caméra).
    """
    xy, c = kpts[..., :2], kpts[..., 2]
    a, b = BONES[:, 0], BONES[:, 1]
    d = xy[..., b, :] - xy[..., a, :]
    v = (c[..., a] >= thr) & (c[..., b] >= thr)
    sh = 0.5 * (xy[..., L_SH, :] + xy[..., R_SH, :])
    hp = 0.5 * (xy[..., L_HIP, :] + xy[..., R_HIP, :])
    tv = (c[..., [L_SH, R_SH, L_HIP, R_HIP]] >= thr).all(-1)[..., None]
    d = np.concatenate([d, (sh - hp)[..., None, :]], -2)
    v = np.concatenate([v, tv], -1)
    n = np.linalg.norm(d, axis=-1)
    ref = np.where(tv, n[..., -1:], 0.0)  # longueur du tronc, 0 si inconnue
    v = v & (n > np.maximum(min_len * ref, 1e-3))
    u = np.where(v[..., None], d / np.maximum(n, 1e-6)[..., None], 0.0)
    return u.astype(np.float32), v
