"""Bibliothèque de chansons importées : liste, vignettes, meilleurs scores, suppression."""
import json
import re
import shutil
import time
from pathlib import Path

import cv2

REQUIRED = ("meta.json", "video.mp4", "audio.ogg", "poses.npz")
YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})")


def youtube_id(url):
    m = YT_ID.search(url or "")
    return m.group(1) if m else None


def parse_links(text):
    """Texte collé (un ou plusieurs liens, séparés par espaces / retours ligne) -> liste
    d'URL http(s) sans doublons, ordre conservé. Les doublons sont repérés par identifiant
    YouTube quand il existe (le même lien peut porter des paramètres différents)."""
    out, seen = [], set()
    for tok in re.split(r"\s+", text or ""):
        if not re.match(r"^https?://", tok):
            continue
        key = youtube_id(tok) or tok
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out


def display_title(title):
    """Titre YouTube -> titre lisible : « JUST DANCE 2014 - i need your love - 5 stars »
    -> « i need your love » ; garde les parenthèses (« Waka Waka (This Time For Africa) »)."""
    t = re.sub(r"(?i)^\s*just\s*dance\s*\(?(\d{1,4}|unlimited|now|kids|wii|summer\s*party)?\)?\s*[:\-|–]*\s*", "", title or "")
    t = re.sub(r"(?i)\s*[-|–]\s*(\d+\s*(players?|joueurs?|stars?|étoiles?)|megastar|superstar|gameplay).*$", "", t)
    t = re.sub(r"(?i)\s*[\[(](official|gameplay|full|hd|4k)[^\])]*[\])]", "", t)
    t = re.sub(r"\s*\[[^\]]*\]", "", t)                           # [11.7k], [MEGASTAR]…
    t = re.sub(r"(?i)\s*[-|–]\s*just\s*dance.*$", "", t)              # « | Just Dance 3 », « - Just Dance China »
    return t.strip(" -|–") or (title or "")


def list_songs(root):
    """-> liste de dicts {path, meta, best}, triée par titre. Ignore les dossiers incomplets
    (import en cours ou raté)."""
    root = Path(root)
    songs = []
    if not root.is_dir():
        return songs
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("_") or not all((d / f).is_file() for f in REQUIRED):
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        songs.append({"path": d, "meta": meta, "best": best_score(d)})
    from .songinfo import display_name
    songs.sort(key=lambda s: display_name(s["meta"]).lower())
    return songs


REJECTED = "_rejetees.json"


def load_rejected(root):
    """Liens rejetés à l'import (coach mal détecté…) : {id: {"title", "reason", "date"}}."""
    try:
        return json.loads((Path(root) / REJECTED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def add_rejected(root, vid, title, reason):
    if not vid:
        return
    d = load_rejected(root)
    d[vid] = {"title": title, "reason": reason, "date": time.strftime("%Y-%m-%d %H:%M")}
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / REJECTED).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def imported_ids(root):
    """Identifiants YouTube déjà importés (pour sauter les doublons d'une liste)."""
    ids = set()
    for s in list_songs(root):
        i = s["meta"].get("id") or youtube_id(s["meta"].get("source", ""))
        if i:
            ids.add(i)
    return ids


def load_scores(song):
    p = Path(song) / "scores.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def best_score(song):
    """Meilleure entrée (tous nombres de joueurs confondus) ou None."""
    s = load_scores(song)
    return max(s, key=lambda e: e["total"]) if s else None


def save_result(song, scores, n_players):
    """Ajoute les résultats d'une partie. -> liste de bool par joueur : nouveau record ?"""
    prev = best_score(song)
    prev_total = prev["total"] if prev else -1
    entries = load_scores(song)
    date = time.strftime("%Y-%m-%d %H:%M")
    records = []
    for p, sc in enumerate(scores):
        entries.append({"date": date, "players": n_players, "player": p + 1, "total": sc.total,
                        "ratio": round(sc.ratio, 4), "stars": sc.stars, "counts": sc.counts})
        records.append(sc.total > prev_total and sc.total > 0)
    (Path(song) / "scores.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    return records


def thumbnail(song, width=320):
    """Chemin de la vignette (thumb.jpg), générée au besoin depuis la vidéo (image à 40 %)."""
    song = Path(song)
    p = song / "thumb.jpg"
    if not p.is_file():
        cap = cv2.VideoCapture(str(song / "video.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.4))
        ok, f = cap.read()
        cap.release()
        if not ok:
            return None
        h = int(f.shape[0] * width / f.shape[1])
        cv2.imwrite(str(p), cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return p


def delete_song(song, root):
    """Supprime un dossier de chanson, uniquement s'il est bien une chanson sous root."""
    song, root = Path(song).resolve(), Path(root).resolve()
    if song.parent != root or not (song / "meta.json").is_file():
        raise ValueError(f"refus de supprimer {song}")
    shutil.rmtree(song)
