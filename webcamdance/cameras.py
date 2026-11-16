"""Inventaire des webcams (index OpenCV + nom lisible) et mémorisation du choix du joueur."""
import json
import re
import sys
from pathlib import Path

import cv2


def _clean(name):
    a, _, b = name.partition(": ")
    return a if a == b else name                 # V4L2 : « Nuroum V11: Nuroum V11 »


def _linux():
    out = []
    for d in Path("/sys/class/video4linux").glob("video*"):
        try:
            if (d / "index").read_text().strip() != "0":    # 1 = nœud de métadonnées
                continue
            name = _clean((d / "name").read_text().strip())
        except OSError:
            continue
        if re.search(r"\bIR\b", name):                      # caméra infrarouge (Windows Hello)
            continue
        out.append((int(d.name[5:]), name))
    return sorted(out)


def _windows():
    try:  # même ordre d'énumération DirectShow que cv2.CAP_DSHOW
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception:
        return None
    return [(i, n) for i, n in enumerate(names) if not re.search(r"\bIR\b", n)]


def _probe(busy=(), n=6):
    """Repli sans noms : on tente d'ouvrir les premiers index (lent, ~0,5 s par index)."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    out = []
    for i in range(n):
        if i in busy:
            out.append((i, f"Caméra {i + 1}"))
            continue
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            out.append((i, f"Caméra {i + 1}"))
        cap.release()
    return out


def list_cameras(busy=()):
    """-> [(index OpenCV, nom)]. `busy` : index déjà ouverts par le jeu (pas de sonde dessus)."""
    if sys.platform.startswith("linux") and Path("/sys/class/video4linux").is_dir():
        return _linux()
    if sys.platform == "win32":
        cams = _windows()
        if cams is not None:
            return cams
    return _probe(busy)


class Settings:
    """Réglages persistants (settings.json à côté du dossier des chansons)."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, **kv):
        self.data.update(kv)
        try:
            from .library import write_text_atomic
            write_text_atomic(self.path, json.dumps(self.data, indent=2, ensure_ascii=False))
        except OSError:
            pass


def pick(cams, settings, default=0):
    """Index à ouvrir au démarrage : la caméra mémorisée (par nom, les index bougent au
    rebranchement), sinon la première disponible."""
    name = settings.get("camera")
    for i, n in cams:
        if n == name:
            return i
    if any(i == default for i, _ in cams) or not cams:
        return default
    return cams[0][0]
