"""Tests découpage (sans GPU) : parsing des temps et choix du segment dansé."""
import numpy as np
import pytest

from webcamdance.trim import find_segment, fmt_time, parse_time


def test_parse_time():
    assert parse_time("75") == 75 and parse_time("75.5") == 75.5
    assert parse_time("1:15") == 75 and parse_time("1:15.5") == 75.5
    assert parse_time("0:01:15") == 75 and parse_time(None) is None and parse_time(" ") is None
    for bad in ["1:2:3:4", "abc", "-5", "1:"]:
        with pytest.raises(ValueError):
            parse_time(bad)
    assert fmt_time(75.5) == "1:15.5"


def test_find_segment():
    dt = 0.5
    a = np.zeros(400, bool)
    a[20:30] = True             # menu animé court (5 s)
    a[40:300] = True            # danse 20 s -> 150 s
    a[150:155] = False          # pause de 2.5 s au milieu : comblée
    a[350:360] = True           # écran de fin animé
    assert find_segment(a, dt) == (20.0, 150.0)
    a[150:165] = False          # trou de 7.5 s > max_gap : coupe, garde le plus long morceau
    assert find_segment(a, dt) == (82.5, 150.0)
    assert find_segment(np.zeros(10, bool), dt) is None
    assert find_segment(np.r_[np.ones(10, bool), np.zeros(10, bool)], dt) is None  # < min_len


def _person(T, top, bottom):
    """Coach statique dont les keypoints vont de top à bottom (pixels)."""
    K = np.zeros((T, 1, 17, 3), np.float32)
    K[..., 1] = np.linspace(top, bottom, 17)
    K[..., 0] = 640
    K[..., 2] = 0.9
    return K


def test_coach_band():
    from webcamdance.trim import coach_band
    H = 720
    y0, y1 = coach_band(_person(100, 200, 650), H)          # coach Just Dance : haut coupé
    assert 0.1 < y0 < 0.25 and y1 > 0.9
    K = _person(100, 200, 650)
    K[50, 0, :, 1] = np.linspace(10, 100, 17)                 # 1 frame aberrante : ignorée
    assert coach_band(K, H)[0] > 0.1
    assert coach_band(_person(100, 30, 710), H) == (0.0, 1.0)  # danseur plein cadre : rien coupé
    y0, y1 = coach_band(_person(100, 300, 400), H)            # coach minuscule : hauteur min
    assert abs((y1 - y0) - 0.4) < 1e-3
    assert coach_band(np.zeros((10, 1, 17, 3), np.float32), H) == (0.0, 1.0)


def test_safe_segment():
    from webcamdance.trim import safe_segment
    dt = 0.5
    a = np.zeros(500, bool)                     # vidéo de 250 s
    a[100:460] = True                           # danse 50 s -> 230 s : segment fiable
    assert safe_segment(a, dt, (50.0, 230.0)) == ((50.0, 230.0), False)
    b = np.zeros(500, bool)                     # danse vue par intermittence (coach mal détecté)
    b[120::9] = True
    b[440:] = False
    seg, unc = safe_segment(b, dt, (110.0, 128.0))
    assert unc and seg[0] <= 60.0 and seg[1] >= 215.0      # première -> dernière frame dansée
    assert safe_segment(np.zeros(500, bool), dt, None) == (None, True)   # rien : vidéo entière
    c = np.zeros(500, bool); c[200:220] = True  # 10 s de danse seulement : vidéo entière
    assert safe_segment(c, dt, (100.0, 110.0)) == (None, True)
