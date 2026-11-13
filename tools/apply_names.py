"""Réapplique les noms « Artiste – Titre » (songs/_noms.json prioritaire, sinon automatique)
aux chansons de la bibliothèque, et régénère la liste lisible des liens.

    python tools/apply_names.py [--offline] [--links liste_chansons_link.txt.bak]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance import library  # noqa: E402
from webcamdance.songinfo import display_name, load_names, song_info  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--songs-dir", default="songs")
ap.add_argument("--offline", action="store_true", help="pas de recherche Deezer")
ap.add_argument("--links", help="fichier de liens à décrire dans liste_chansons_titres.txt")
a = ap.parse_args()
for s in library.list_songs(a.songs_dir):
    m = s["meta"]
    n = song_info(m.get("title", ""), m.get("id") or library.youtube_id(m.get("source", "")), a.songs_dir,
                  online=not a.offline)
    m.update(artist=n["artist"], song=n["song"], version=n["version"], name_source=n["source"])
    (s["path"] / "meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{n['source']:7s} {display_name(m)}")
if a.links:
    names, done, rej = load_names(a.songs_dir), library.imported_ids(a.songs_dir), library.load_rejected(a.songs_dir)
    rows = []
    for u in library.parse_links(Path(a.links).read_text(encoding="utf-8")):
        i = library.youtube_id(u)
        n = names.get(i)
        label = display_name(n) if n else "(nom inconnu)"
        st = "importée" if i in done else ("rejetée" if i in rej else "")
        rows.append((label, st + (" · à vérifier" if n and n.get("verifier") else ""), u))
    rows.sort(key=lambda r: r[0].lower())
    out = Path("liste_chansons_titres.txt")
    out.write_text(f"# {len(rows)} chansons — Artiste – Titre (version) | statut | lien\n"
                   + "".join(f"{l} | {st} | {u}\n" for l, st, u in rows), encoding="utf-8")
    print(f"{out} : {len(rows)} lignes")
