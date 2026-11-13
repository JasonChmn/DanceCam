"""Suivi joueurs : échange de places volontaire, occlusion, tenues identiques, ré-entrée."""
import numpy as np

from webcamdance.appearance import DIM, NPART, PART
from webcamdance.tracking import PlayerTracker, SlotTracker

FPS = 30


def _sig(seed):
    """Signature couleur synthétique (racine d'histogramme normalisée par partie)."""
    h = np.random.default_rng(seed).random(DIM).astype(np.float32) ** 4
    return _norm(h)


def _norm(h):
    for k in range(NPART):
        h[k * PART:(k + 1) * PART] /= np.linalg.norm(h[k * PART:(k + 1) * PART])
    return h


def _noisy(s, rng):
    return _norm(np.abs(s + rng.normal(0, 0.02, s.shape)).astype(np.float32))


def _run(tr, steps, sa, sb, occl=(), rng=None):
    """steps : liste de (xa, xb) ; occl : frames où A est masqué derrière B. -> slot de A à la fin."""
    rng = rng or np.random.default_rng(0)
    sc = np.array([400., 400.], np.float32)
    asg = None
    for i, (xa, xb) in enumerate(steps, start=1):
        t = i / FPS
        if i in occl:                                        # une seule détection (B devant)
            cen, sig, who = np.array([[xb, 360.]], np.float32), np.array([_noisy(sb, rng)]), ["B"]
        else:
            order = [("A", xa, sa), ("B", xb, sb)]
            if i % 2:
                order.reverse()                               # ordre de détection arbitraire
            cen = np.array([[x, 360.] for _, x, _ in order], np.float32)
            sig = np.array([_noisy(s, rng) for _, _, s in order])
            who = [w for w, _, _ in order]
        a = tr.update(cen, sc[: len(cen)], sig, t)
        asg = [who[j] if j is not None else None for j in a]
    return asg


def _swap_steps(n=120, hold=60):
    """A (gauche) et B (droite) échangent leurs places puis RESTENT échangés."""
    out = []
    for i in range(n):
        u = 0.5 - 0.5 * np.cos(np.pi * min(i / (n - 1), 1.0))
        out.append((300 + 600 * u, 900 - 600 * u))
    return out + [out[-1]] * hold


def _init(tr, sa, sb):
    tr.init(np.array([[300., 360.], [900., 360.]], np.float32), np.array([400., 400.], np.float32),
            np.array([sa, sb]), 0.0)


def test_swap_places_with_occlusion():
    sa, sb = _sig(1), _sig(2)
    tr = PlayerTracker(2, 1280)
    _init(tr, sa, sb)
    asg = _run(tr, _swap_steps(), sa, sb, occl=set(range(55, 66)))
    assert asg == ["A", "B"]                                  # P1 reste A, même passé à droite


def test_lane_tracker_reverts_on_swap():
    """Documente le défaut de l'ancien suivi par couloir : il ré-inverse les joueurs."""
    tr = SlotTracker(2, 1280, lane_weight=1.5)
    sc = np.array([400., 400.], np.float32)
    tr.init(np.array([[300., 360.], [900., 360.]]), sc, 0.0)
    for i, (xa, xb) in enumerate(_swap_steps(), start=1):
        a = tr.update(np.array([[xa, 360.], [xb, 360.]], np.float32), sc, i / FPS)
    assert a == [1, 0]                                         # slot 0 (P1) a pris B


def test_same_clothes_crossing():
    """Tenues identiques : seule la continuité (Kalman) départage, croisement à vitesse
    constante avec 4 frames où A est masqué."""
    s = _sig(3)
    tr = PlayerTracker(2, 1280)
    _init(tr, s, s)
    steps = [(300 + 10 * i, 900 - 10 * i) for i in range(1, 61)]   # se croisent vers i=30
    asg = _run(tr, steps, s, s, occl={29, 30, 31, 32})
    assert asg == ["A", "B"]


def test_reentry_other_side():
    """A sort du champ 2 s puis revient du côté de B : récupéré par la couleur."""
    sa, sb = _sig(4), _sig(5)
    tr = PlayerTracker(2, 1280)
    _init(tr, sa, sb)
    rng = np.random.default_rng(1)
    sc = np.array([400.], np.float32)
    for i in range(1, 61):                                     # A absent
        tr.update(np.array([[900., 360.]], np.float32), sc, np.array([_noisy(sb, rng)]), i / FPS)
    a = tr.update(np.array([[1100., 360.], [700., 360.]], np.float32), np.array([400., 400.], np.float32),
                  np.array([_noisy(sa, rng), _noisy(sb, rng)]), 61 / FPS)
    assert a == [0, 1]                                         # A (à droite) -> P1
