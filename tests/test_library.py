"""Bibliothèque : liens collés, noms courts, scores / records, suppression protégée."""
import json
from types import SimpleNamespace

import pytest

from webcamdance import library
from webcamdance.import_song import short_name


def test_parse_links_dedup():
    text = ("https://www.youtube.com/watch?v=OobR681KUTo\r\nhttps://www.youtube.com/watch?v=OobR681KUTo\n"
            "  https://youtu.be/f0hxDWRq7fk  pas-un-lien https://www.youtube.com/watch?v=OobR681KUTo&t=3s")
    assert library.parse_links(text) == ["https://www.youtube.com/watch?v=OobR681KUTo", "https://youtu.be/f0hxDWRq7fk"]


def test_short_name(tmp_path):
    assert short_name("Just Dance 2018: Waka Waka (This Time For Africa) - 4 players", tmp_path) == "waka_waka"
    assert short_name("JUST DANCE 2014 - i need your love - 5 stars", tmp_path) == "i_need_your_love"
    assert short_name("Just Dance 2015 Tetris", tmp_path) == "tetris"
    (tmp_path / "tetris").mkdir()
    assert short_name("Just Dance 2015 Tetris", tmp_path) == "tetris_2"


def _song(root, name):
    d = root / name
    d.mkdir()
    for f in library.REQUIRED:
        (d / f).write_text("{}" if f == "meta.json" else "x")
    (d / "meta.json").write_text(json.dumps({"title": name, "source": "https://youtu.be/f0hxDWRq7fk"}))
    return d


def test_scores_and_records(tmp_path):
    d = _song(tmp_path, "a")
    sc = lambda t: SimpleNamespace(total=t, ratio=t / 10000, stars=1, counts={})
    assert library.save_result(d, [sc(5000)], 1) == [True]
    assert library.save_result(d, [sc(4000), sc(6000)], 2) == [False, True]
    assert library.best_score(d)["total"] == 6000
    songs = library.list_songs(tmp_path)
    assert [s["path"].name for s in songs] == ["a"] and songs[0]["best"]["total"] == 6000
    assert library.imported_ids(tmp_path) == {"f0hxDWRq7fk"}


def test_delete_guard(tmp_path):
    d = _song(tmp_path, "a")
    with pytest.raises(ValueError):
        library.delete_song(tmp_path, tmp_path.parent)             # pas une chanson
    library.delete_song(d, tmp_path)
    assert not d.exists()


def test_display_title():
    assert library.display_title("JUST DANCE 2014 - i need your love - 5 stars") == "i need your love"
    assert library.display_title("Just Dance 2018: Waka Waka (This Time For Africa) - 4 players") == \
        "Waka Waka (This Time For Africa)"
    assert library.display_title("Just Dance 2015 Tetris") == "Tetris"
    assert library.display_title("Ma choré") == "Ma choré"
    assert library.display_title("I'm So Excited - Pointer Sisters | Just Dance 3") == "I'm So Excited - Pointer Sisters"
    assert library.display_title("If You Wanna Party - The Just Dancers - Just Dance China") == \
        "If You Wanna Party - The Just Dancers"


def test_titles_short_versions(tmp_path):
    t = "Just Dance 3: Apache (Jump On It) by The Sugarhill Gang [11.7k]"
    assert library.display_title(t) == "Apache (Jump On It) by The Sugarhill Gang"
    assert short_name(t, tmp_path) == "apache_by_the_sugarhill_gang"
    assert library.display_title("Just Dance (2020) China Unlimited Dancing Diva") == "China Unlimited Dancing Diva"


def test_rejected_and_presence(tmp_path):
    import numpy as np
    from webcamdance.import_song import coach_presence
    library.add_rejected(tmp_path, "YkERUUA-MNg", "Apache", "coach détecté 59 %")
    assert "YkERUUA-MNg" in library.load_rejected(tmp_path)
    K = np.zeros((10, 2, 17, 3), np.float32)
    K[:6, 0, :, 2] = 0.9                       # coach 0 vu 60 % du temps
    K[:, 1, :, 2] = 0.9                        # coach 1 tout le temps
    assert abs(coach_presence(K, 0.35) - 0.8) < 1e-6


def test_terminal_width_cjk():
    from webcamdance.import_list import disp_width, fit
    assert disp_width("abc") == 3 and disp_width("百乐门") == 6
    line = fit("2/427 Bailemen Swing - 百乐门Swing : recherche", 30)
    assert disp_width(line) == 30


def test_menu_filter_songs():
    from webcamdance.menu import filter_songs
    songs = [{"meta": {"artist": "Beyoncé", "song": "Halo"}},
             {"meta": {"artist": "Rihanna", "song": "Umbrella"}},
             {"meta": {"artist": "Shakira", "song": "Waka Waka", "version": "Extended"}}]
    names = lambda q: [s["meta"]["song"] for s in filter_songs(songs, q)]
    assert names("") == ["Halo", "Umbrella", "Waka Waka"]
    assert names("beyonce") == ["Halo"]            # sans accents ni casse
    assert names("waka shak") == ["Waka Waka"]     # mots dans le désordre
    assert names("extended") == ["Waka Waka"]
    assert names("zzz") == []
