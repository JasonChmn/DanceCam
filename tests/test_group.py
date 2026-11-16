"""Mode groupe (5+) : bras levés, coachs par zone, score d'équipe, suivi après la photo, rendu."""
import os
import time

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from webcamdance import hud  # noqa: E402
from webcamdance.capture import PlayerWorker  # noqa: E402
from webcamdance.config import Config  # noqa: E402
from webcamdance.scoring import TeamScore, fun_titles, group_coaches  # noqa: E402
from webcamdance.tracking import arms_up  # noqa: E402


def person(x, wrist_y=300.0, h=300.0):
    """Squelette debout à l'abscisse x (tête vers y=100), poignets à wrist_y."""
    kp = np.zeros((17, 3), np.float32)
    ys = {0: 100, 1: 95, 2: 95, 3: 100, 4: 100, 5: 140, 6: 140, 7: 200, 8: 200, 9: wrist_y, 10: wrist_y,
          11: 250, 12: 250, 13: 320, 14: 320, 15: 100 + h, 16: 100 + h}
    for i, y in ys.items():
        kp[i] = (x + (-20 if i % 2 else 20), y, 0.9)
    return kp


def test_arms_up():
    assert arms_up(person(100, wrist_y=60), 0.35)
    assert not arms_up(person(100, wrist_y=300), 0.35)
    kp = person(100, wrist_y=60)
    kp[9:11, 2] = 0                                   # poignets invisibles
    assert not arms_up(kp, 0.35)


def test_group_coaches_zones_and_rows():
    fx, pres = [0.2, 0.4, 0.6, 0.8], [1, 1, 1, 1]
    # 8 joueurs sur 2 rangées : deux à la même abscisse suivent le même coach
    xs = [100, 100, 300, 300, 500, 500, 700, 700]
    a = group_coaches(xs, fx, pres)
    assert a == [0, 0, 1, 1, 2, 2, 3, 3]
    assert group_coaches(xs, fx, pres, offset=1)[:2] == [1, 1]
    assert group_coaches([400, 400], fx, pres) == [2, 2]           # tous au même endroit : coach du milieu
    assert group_coaches([100, 700], [0.2, 0.8], [1, 0.3]) == [0, 0]  # coach peu présent : pas proposé


class _Sc:
    def __init__(self, labels, total=0, ratio=0.0):
        self.history = [{"move": i, "label": lb} for i, lb in enumerate(labels)]
        self.w, self.total, self.ratio = len(labels), total, ratio
        self.best_combo, self.counts = 0, {"PERFECT": labels.count("PERFECT")}


def test_team_combo_ignores_hidden_players():
    a = _Sc(["PERFECT", "SUPER", "MISS", "PERFECT"], 300, 0.6)
    b = _Sc(["SUPER", "SKIP", "MISS", "SKIP"], 100, 0.4)       # caché aux moves 1 et 3
    c = _Sc(["PERFECT", "PERFECT", "GOOD", "SUPER"], 350, 0.8)
    team = TeamScore([a, b, c], frac=0.7)
    team.update()
    assert team.done == 4
    assert team.best_combo == 2 and team.combo == 1 and team.hits == 3
    assert team.total == 750 and abs(team.ratio - 0.6) < 1e-6 and team.stars == 3
    assert fun_titles([a, b, c])[0] == ("Le plus synchro", 2)


class _Cam:
    w, h = 1280, 720

    def __init__(self, people):
        self.i, self.people = 0, people

    def latest(self):
        self.i += 1
        return self.i, np.zeros((self.h, self.w, 3), np.uint8), time.perf_counter()


def test_worker_group_lock_and_keep():
    xs = [900, 200, 600, 400]                         # ordre de détection quelconque
    kps = np.stack([person(x) for x in xs])
    boxes = np.array([[x - 60, 80, x + 60, 420] for x in xs], np.float32)
    pose = lambda f, imgsz=None: [(kps, boxes, np.ones(len(xs), np.float32))]
    w = PlayerWorker(_Cam(xs), pose, Config(), 0, group=True)
    w.start()
    try:
        for _ in range(200):
            if w.raw is not None:
                break
            time.sleep(0.01)
        raw = w.raw
        assert list(raw[5][:, 0]) == sorted(raw[5][:, 0])          # triés gauche -> droite
        w.lock_people(raw, range(4))
        w.keep([1, 3])                                             # P1 = x 400, P2 = x 900
        for _ in range(200):
            fr, poses = w.view
            if w.locked and len(poses) == 2 and all(p is not None for p in poses):
                break
            time.sleep(0.01)
        assert w.n == 2
        assert [round(float(p[5:7, 0].mean())) for p in poses] == [400, 900]
        w.unlock()
        for _ in range(200):
            if not w.locked:
                break
            time.sleep(0.01)
        assert not w.locked
    finally:
        w.stop()
        w.join(timeout=2)


def test_group_hud_renders():
    pygame.init()
    hud.set_scale(720)
    surf = pygame.Surface((1280, 720))
    scores = []
    for i in range(8):
        s = _Sc(["PERFECT"], 100 * i, 0.1 * i)
        s.last, s.combo, s.stars = ("PERFECT", 1.0), 0, i % 6
        scores.append(s)
    team = TeamScore(scores)
    team.update()
    frame = np.zeros((720, 1280, 3), np.uint8)
    poses = [person(100 + 140 * i) for i in range(8)]
    hud.group_strip(surf, pygame.Rect(0, 468, 1280, 252), frame, poses, scores, team, 1.2, 0.35)
    hud.group_results(surf, scores, team, fun_titles(scores))
    assert len({hud.pcolor(i) for i in range(8)}) == 8
    pygame.quit()
