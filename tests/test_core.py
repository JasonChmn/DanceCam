"""Tests synthétiques scoring + tracking (sans GPU/caméra) : python -m pytest tests"""
import numpy as np

from webcamdance.config import Config
from webcamdance.scoring import CoachTrack, PlayerScore
from webcamdance.tracking import SlotTracker


def _dance(T, fps, seed=0):
    """Squelette articulé plausible (17,3) animé par des sinusoïdes."""
    rng = np.random.default_rng(seed)
    ph, fr = rng.uniform(0, 6, 8), rng.uniform(0.5, 1.5, 8)
    K = np.zeros((T, 17, 3), np.float32)
    K[..., 2] = 0.9
    for i in range(T):
        a = 1.2 * np.sin(2 * np.pi * fr * i / fps + ph)
        sh = {5: np.array([440., 200.]), 6: np.array([360., 200.])}
        hp = {11: np.array([425., 400.]), 12: np.array([375., 400.])}
        d = lambda ang, L: L * np.array([np.sin(ang), np.cos(ang)])
        K[i, 5, :2], K[i, 6, :2], K[i, 11, :2], K[i, 12, :2] = sh[5], sh[6], hp[11], hp[12]
        K[i, 7, :2] = sh[5] + d(a[0], 90); K[i, 9, :2] = K[i, 7, :2] + d(a[1], 80)
        K[i, 8, :2] = sh[6] + d(a[2], 90); K[i, 10, :2] = K[i, 8, :2] + d(a[3], 80)
        K[i, 13, :2] = hp[11] + d(0.4 * a[4], 110); K[i, 15, :2] = K[i, 13, :2] + d(0.4 * a[5], 100)
        K[i, 14, :2] = hp[12] + d(0.4 * a[6], 110); K[i, 16, :2] = K[i, 14, :2] + d(0.4 * a[7], 100)
    return K


def _coach(tmp_path, K, fps=30):
    p = tmp_path / "poses.npz"
    np.savez_compressed(p, kpts=K[:, None], fps=np.float32(fps))
    return CoachTrack(p, Config())


def _play(coach, frames, fps, T):
    sc = PlayerScore(Config(), coach, 0)
    for i in range(T):
        if frames(i) is not None:
            sc.add(i / fps, frames(i))
    sc.close_due(T / fps + 1)
    return sc


def test_scoring(tmp_path):
    fps, T = 30, 600
    K, K2 = _dance(T, fps), _dance(T, fps, seed=7)
    coach = _coach(tmp_path, K, fps)

    def scaled(i):                                   # même danse, autre échelle/position
        p = K[i].copy(); p[:, :2] = p[:, :2] * 0.4 + [300, 50]; return p

    def nolegs(i):                                   # jambes masquées côté joueur
        p = K[i].copy(); p[13:17, 2] = 0.0; return p

    same = _play(coach, scaled, fps, T)
    late = _play(coach, lambda i: K[max(0, i - 6)], fps, T)      # 200 ms de retard
    other = _play(coach, lambda i: K2[i], fps, T)                # autre danse
    still = _play(coach, lambda i: K[0], fps, T)                 # joueur immobile
    occl = _play(coach, nolegs, fps, T)
    for n, s in [("same", same), ("late", late), ("other", other), ("still", still), ("occl", occl)]:
        print(f"{n:6s} ratio={s.ratio:.2f} stars={s.stars} {s.counts}")
    assert same.ratio > 0.95 and late.ratio > 0.9
    assert other.ratio < 0.35 and still.ratio < 0.35
    assert occl.ratio > 0.9                     # jambes masquées (occlusion) : plus pénalisées
    assert other.ratio < occl.ratio


def test_undetected_player_not_judged(tmp_path):
    """Joueur non détecté (YOLO) ou trop peu visible : moves non jugés, pas des MISS."""
    fps, T = 30, 300
    K = _dance(T, fps)
    coach = _coach(tmp_path, K, fps)

    def one_arm(i):                              # seul un avant-bras visible : trop peu pour juger
        p = K[i].copy(); p[5:17, 2] = 0.0; p[[7, 9], 2] = 0.9; return p

    def gap(i):                                  # détecté sauf pendant 2 s (≈ 2-3 moves)
        return None if 90 <= i < 150 else K[i]

    for frames in (lambda i: None, one_arm):
        sc = _play(coach, frames, fps, T)
        assert sc.counts["MISS"] == 0 and sc.total == 0 and sc.skipped > 0
    sc = _play(coach, gap, fps, T)
    assert sc.counts["MISS"] == 0 and sc.skipped >= 2
    assert sc.ratio > 0.95                       # les moves non jugés sortent du maximum
    assert sc.best_combo == sum(sc.counts.values())   # combo jamais cassé par un trou de détection


def test_lane_tracker_crossing():
    tr = SlotTracker(2, 1280, lane_weight=1.5)
    sc = np.array([400., 400.], np.float32)
    tr.init(np.array([[300., 360.], [900., 360.]]), sc, 0.0)
    last = None
    for i in range(1, 181):                                             # A et B se croisent puis reviennent
        u = 0.5 - 0.5 * np.cos(2 * np.pi * i / 180)
        a, b = [300 + 600 * u, 360.], [900 - 600 * u, 360.]
        dets = np.array([b, a] if i % 2 else [a, b], np.float32)        # ordre de détection arbitraire
        asg = tr.update(dets, sc, i / 30)
        last = (asg, 1 if i % 2 else 0)
    asg, a_idx = last
    assert asg[0] == a_idx and asg[1] == 1 - a_idx                      # A de retour dans le slot 0
