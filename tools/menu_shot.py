"""Pilote le menu avec des événements simulés et prend des captures (test sans clavier).

    python tools/menu_shot.py OUT_DIR "t:action" ...
actions : shot, key:<nom pygame>, text:<texte>, quit
"""
import sys
import threading
import time
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance import menu  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
steps = [(float(a.split(":", 1)[0]), a.split(":", 1)[1]) for a in sys.argv[2:]]
t0, _orig_flip, shots = time.perf_counter(), pygame.display.flip, []


def _flip_shot():
    while shots:
        pygame.image.save(pygame.display.get_surface(), str(out / f"menu_{shots.pop(0):05.1f}.png"))
    _orig_flip()


def driver():
    for t, act in steps:
        time.sleep(max(0.0, t - (time.perf_counter() - t0)))
        if act == "shot":
            shots.append(t)
        elif act.startswith("key:"):
            k = getattr(pygame, "K_" + act[4:])
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
        elif act.startswith("text:"):
            pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=act[5:]))
        elif act == "quit":
            pygame.event.post(pygame.event.Event(pygame.QUIT))


pygame.display.flip = _flip_shot
threading.Thread(target=driver, daemon=True).start()
sys.argv = ["menu", "--window", "1600x900"]
menu.main()
