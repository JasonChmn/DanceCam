"""Identité des coachs par segments : croisement, changement de tenue, tenues identiques."""
import numpy as np

from webcamdance.appearance import DIM, NPART, PART
from webcamdance.coach_id import identify, identity_jumps

FPS = 30


def _sig(seed):
    h = np.random.default_rng(seed).random(DIM).astype(np.float32) ** 4
    for k in range(NPART):
        h[k * PART:(k + 1) * PART] /= np.linalg.norm(h[k * PART:(k + 1) * PART])
    return h


def _frames(track_a, track_b, sig_a, sig_b, merge=()):
    """track_x(t) -> x ; sig_x(t) -> signature. merge : frames où une seule détection (occultation)."""
    rng = np.random.default_rng(0)
    out = []
    for t in range(len(track_a)):
        pts = [(track_a[t], sig_a(t)), (track_b[t], sig_b(t))]
        if t in merge:
            pts = [((track_a[t] + track_b[t]) / 2, sig_a(t))]
        if t % 2:
            pts.reverse()                                       # ordre de détection arbitraire
        cen = np.array([[x + rng.normal(0, 2), 400.0] for x, _ in pts], np.float32)
        out.append({"cen": cen, "sc": np.full(len(pts), 400.0, np.float32), "fg": np.ones(len(pts), bool),
                    "ovl": np.zeros(len(pts), np.float32), "sig": np.array([s for _, s in pts]),
                    "k": np.zeros((len(pts), 17, 3), np.float32)})
    return out


def _x_end(frames, assign, k):
    t = max(t for t in range(len(frames)) if assign[t, k] >= 0)
    return frames[t]["cen"][assign[t, k], 0]


def test_swap_places_distinct_outfits():
    """Ils échangent leurs places (avec occultation au croisement) et restent échangés."""
    T = 150
    u = np.clip(np.arange(T) / 90, 0, 1)
    a, b = 300 + 600 * u, 900 - 600 * u
    sa, sb = _sig(1), _sig(2)
    fr = _frames(a, b, lambda t: sa, lambda t: sb, merge=set(range(43, 48)))
    assign, _ = identify(fr, 2, FPS)
    assert _x_end(fr, assign, 0) > 800 and _x_end(fr, assign, 1) < 400   # coach 0 (parti à gauche) fini à droite
    assert identity_jumps(fr, assign) == 0


def test_outfit_change_in_place():
    """Les deux coachs changent de tenue au même moment sans bouger : identités conservées."""
    T = 120
    a, b = 300 + 20 * np.sin(np.arange(T) / 5), 900 + 20 * np.cos(np.arange(T) / 5)
    s = [_sig(i) for i in range(4)]
    fr = _frames(a, b, lambda t: s[0] if t < 60 else s[2], lambda t: s[1] if t < 60 else s[3])
    assign, _ = identify(fr, 2, FPS)
    assert (assign >= 0).mean() > 0.95
    assert all(abs(fr[t]["cen"][assign[t, 0], 0] - 300) < 60 for t in range(T) if assign[t, 0] >= 0)
    assert identity_jumps(fr, assign) == 0


def test_identical_outfits_crossing():
    """Tenues identiques, croisement à vitesse constante avec 4 frames d'occultation : la
    position prédite à travers le trou départage."""
    T = 90
    a, b = 300 + 8 * np.arange(T), 960 - 8 * np.arange(T)              # se croisent vers t = 41
    s = _sig(5)
    fr = _frames(a, b, lambda t: s, lambda t: s, merge=set(range(39, 44)))
    assign, _ = identify(fr, 2, FPS)
    assert _x_end(fr, assign, 0) > _x_end(fr, assign, 1)               # coach 0 a continué vers la droite
    assert identity_jumps(fr, assign) == 0


def _near(sig, seed, eps=0.08):
    """Signature proche de sig (même tenue, légère variation)."""
    h = np.abs(sig + np.random.default_rng(seed).normal(0, eps, sig.shape)).astype(np.float32)
    for k in range(NPART):
        h[k * PART:(k + 1) * PART] /= np.linalg.norm(h[k * PART:(k + 1) * PART])
    return h


def test_outfit_change_crossed_colors():
    """Pire cas : après le changement, la tenue de A ressemble à l'ancienne tenue de B et
    inversement. Un regroupement global par couleur échange les coachs ; les segments continus
    (position) relient chaque nouvelle tenue à son porteur."""
    T = 240
    a, b = 300 + 20 * np.sin(np.arange(T) / 5), 900 + 20 * np.cos(np.arange(T) / 5)
    A, B = _sig(10), _sig(11)
    A2, B2 = _near(B, 1), _near(A, 2)                   # A porte une tenue proche de celle de B, etc.
    fr = _frames(a, b, lambda t: A if t < 120 else A2, lambda t: B if t < 120 else B2)
    assign, info = identify(fr, 2, FPS)
    assert "changements de tenue" in info
    left = [abs(fr[t]["cen"][assign[t, 0], 0] - 300) < 60 for t in range(T) if assign[t, 0] >= 0]
    assert np.mean(left) > 0.98 and (assign >= 0).mean() > 0.95
