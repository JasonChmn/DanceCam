"""Choix du meilleur / pire passage (fenêtre glissante sur les notes des moves)."""
from types import SimpleNamespace

from webcamdance.config import Config
from webcamdance.highlights import pick_windows


def _player(scores):
    """scores : liste de notes par move (None = move non jugé)."""
    return SimpleNamespace(history=[{"move": m, "score": s, "label": "SKIP" if s is None else "OK"}
                                    for m, s in enumerate(scores)])


def test_meilleur_et_pire_sans_chevauchement():
    cfg = Config()
    notes = [0.6] * 40
    notes[10:17] = [0.95] * 7          # passage brillant
    notes[25:32] = [0.2] * 7           # passage raté
    best, worst = pick_windows([_player(notes)], cfg, 5, 0.0, 100.0)
    p = cfg.move_period
    assert best["a"] == 10 * p and best["label"] == "PERFECT"
    assert worst["a"] == 25 * p and worst["label"] == "MISS"
    assert worst["a"] >= best["b"] or worst["b"] <= best["a"]


def test_moves_sans_la_moitie_des_joueurs_ignores():
    cfg = Config()
    p = cfg.move_period
    seul = [0.6] * 40
    seul[20:30] = [0.1] * 10           # seul joueur encore détecté, et il rate
    absents = [0.6] * 20 + [None] * 10 + [0.6] * 10
    wins = pick_windows([_player(seul)] + [_player(absents) for _ in range(3)], cfg, 5, 0.0, 100.0)
    for w in wins:                     # 1 joueur jugé sur 4 : ces moves ne comptent pas
        assert w["b"] <= 20 * p or w["a"] >= 30 * p or w["mean"] > 0.3


def test_partie_trop_courte():
    assert pick_windows([_player([0.5] * 4)], Config(), 5, 0.0, 100.0) == []


def test_fenetre_dans_l_enregistrement():
    cfg = Config()
    wins = pick_windows([_player([0.5] * 40)], cfg, 5, 2.0, 20.0)
    assert all(w["a"] >= 2.0 and w["b"] <= 20.0 for w in wins)
