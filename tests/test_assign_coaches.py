"""Association joueurs -> coachs par rang (gauche->droite), décalable avec ←/→."""
from webcamdance.scoring import assign_coaches

X = [0.2, 0.4, 0.6, 0.8]                 # formation Waka Waka : n°9, 5, 4, 7
X_SHUFFLED = [0.6, 0.2, 0.8, 0.4]        # indices coach dans le désordre spatial
PRES = [0.8, 0.95, 0.9, 0.9]


def test_same_count_by_rank():
    assert assign_coaches(4, X, PRES) == [0, 1, 2, 3]
    assert assign_coaches(4, X_SHUFFLED, PRES) == [1, 3, 0, 2]   # tri spatial, pas par indice


def test_one_player_main_coach():
    assert assign_coaches(1, X, PRES) == [1]                      # le plus présent
    assert assign_coaches(1, X, PRES, offset=1) == [2]
    assert assign_coaches(1, X, PRES, offset=-2) == [3]           # circulaire


def test_fewer_players_spread():
    assert assign_coaches(2, X, PRES) == [0, 3]
    assert assign_coaches(3, X, PRES) == [0, 2, 3]
    a = assign_coaches(2, X, PRES, offset=1)
    assert a == [1, 0] and len(set(a)) == 2                      # décalé, toujours distincts


def test_more_players_shared():
    assert assign_coaches(4, [0.3, 0.7], [1, 1]) == [0, 0, 1, 1]
    assert assign_coaches(3, [0.7], [1]) == [0, 0, 0]


def test_solo_coach_suit_la_position():
    from webcamdance.scoring import solo_coach
    x, pres = [0.22, 0.48, 0.74], [0.971, 0.976, 0.977]
    assert solo_coach(0.1, x, pres) == 0
    assert solo_coach(0.5, x, pres) == 1
    assert solo_coach(0.9, x, pres) == 2
    assert solo_coach(0.35, x, pres, current=1) == 1                # hystérésis à la frontière
    assert solo_coach(0.1, x, [0.2, 0.95, 0.9]) == 1               # coach peu présent écarté
