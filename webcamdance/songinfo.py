"""Nom lisible « Artiste – Titre » d'une chanson à partir du titre YouTube.

Ordre de priorité :
1. songs/_noms.json (corrections manuelles, par identifiant YouTube) ;
2. champs artist / track fournis par YouTube (musique reconnue) ;
3. « Titre by Artiste » dans le titre ;
4. recherche Deezer (API publique, sans compte) sur le titre nettoyé — on n'envoie que le
   titre de la chanson ; pour « A - B », on teste les deux sens ;
5. titre nettoyé seul (artiste inconnu).
"""
import difflib
import json
import re
from pathlib import Path

NAMES = "_noms.json"

# préfixes / suffixes de jeu ou de mise en ligne, sans rapport avec la chanson
_PREFIX = re.compile(r"(?i)^\s*(?:[•·\-\s]*)(?:just\s*dance\s*\(?(?:\d{1,4}|unlimited|now|kids|wii|summer\s*party|china)?\)?"
                     r"\s*[:\-|–]*\s*)?(?:\((?:china|unlimited|mod|pc|ubisoft\s*club|kids)\)\s*[:\-|–]*\s*)?")
_NOISE = re.compile(r"(?i)\s*(?:[-|–]\s*)?(?:\b\d\s*\*?\s*stars?\b|\bmegastar\b|\bsuperstar\b|\bgameplay\b|"
                    r"\b\d+\s*players?\b|\b1080p\b|\bhd\b|\b4k\b|\bfull\b|\bno\s*hud\b|\bofficial\b)")
_VERSION = re.compile(r"(?i)\((?:[^)]*\b(?:version|extreme|alternate|alternative|remix|mashup|sweat|"
                      r"kids|on stage|classic|duet|dance crew|fanmade|mod)\b[^)]*)\)")


def clean(title):
    """Titre YouTube -> (texte nettoyé, mention de version ou "")."""
    t = _PREFIX.sub("", title or "")
    t = re.sub(r"\s*\[[^\]]*\]", "", t)                           # [11.7k], [MEGASTAR]…
    t = re.sub(r"(?i)\s*[-|–]\s*just\s*dance.*$", "", t)          # « | Just Dance 3 »
    versions = [m.group(0).strip("() ") for m in _VERSION.finditer(t)]
    t = _VERSION.sub("", t)
    t = _NOISE.sub("", t)
    t = re.sub(r"[\"'’`]{2,}", '"', t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -|–•:\"'")
    return t, ", ".join(versions)


def _sim(a, b):
    norm = lambda s: re.sub(r"[^\w]+", " ", s.lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def deezer(query, timeout=8):
    """-> liste de (artiste, titre) des meilleurs résultats Deezer, [] si indisponible."""
    try:
        import requests
        r = requests.get("https://api.deezer.com/search", params={"q": query, "limit": 5}, timeout=timeout)
        return [(d["artist"]["name"], d["title"]) for d in r.json().get("data", [])]
    except Exception:
        return []


def _best(song, results, artist_hint=None):
    """Résultat Deezer dont le titre ressemble au titre cherché (et à l'artiste supposé)."""
    best, score = None, 0.0
    for art, tit in results:
        base = re.sub(r"\s*[\(\[].*", "", tit)                     # « (feat. …) », « (Remix) »
        s = max(_sim(song, tit), _sim(song, base))
        if artist_hint:
            s = 0.5 * s + 0.5 * _sim(artist_hint, art)
        if s > score:
            best, score = (art, base if _sim(song, base) >= _sim(song, tit) else tit), s
    return best if score >= 0.75 else None


def load_names(root):
    try:
        return json.loads((Path(root) / NAMES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def song_info(title, vid=None, root=None, yt_artist=None, yt_track=None, online=True):
    """-> {"artist", "song", "version", "source"} ; artist "" si inconnu."""
    if root and vid:
        manual = load_names(root).get(vid)
        if manual:
            return {"artist": manual.get("artist", ""), "song": manual.get("song", ""),
                    "version": manual.get("version", ""), "source": "manuel"}
    text, version = clean(title)
    if yt_artist and yt_track:
        return {"artist": yt_artist, "song": yt_track, "version": version, "source": "youtube"}
    m = re.match(r"(?i)^(.*?)\s+by\s+(.+)$", text)
    if m:
        song, artist = m.group(1).strip(" -|–:"), m.group(2).strip(" -|–:")
        song = re.sub(r"^.*?[:]\s*", "", song) if ":" in song else song
        return {"artist": artist, "song": song, "version": version, "source": "titre"}
    parts = [p.strip() for p in re.split(r"\s+[-|–]\s+", text) if p.strip()]
    if online:
        if len(parts) >= 2:                                         # « A - B » : quel sens ?
            a, b = parts[0], parts[1]
            for song, art in ((a, b), (b, a)):
                hit = _best(song, deezer(f"{art} {song}"), artist_hint=art)
                if hit:
                    return {"artist": hit[0], "song": song, "version": version, "source": "deezer"}
        hit = _best(parts[0] if parts else text, deezer(parts[0] if parts else text))
        if hit:
            return {"artist": hit[0], "song": hit[1], "version": version, "source": "deezer"}
    if len(parts) >= 2:                                             # sans réseau : « Titre - Artiste » (forme JD courante)
        return {"artist": parts[1], "song": parts[0], "version": version, "source": "titre"}
    return {"artist": "", "song": text, "version": version, "source": "brut"}


def display_name(meta):
    """Nom affiché dans le menu : « Artiste – Titre (version) »."""
    song = meta.get("song")
    if not song:
        from .library import display_title
        return display_title(meta.get("title", ""))
    v = f" ({meta['version']})" if meta.get("version") else ""
    return f"{meta['artist']} – {song}{v}" if meta.get("artist") else f"{song}{v}"
