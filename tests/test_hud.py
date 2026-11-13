"""Rendu HUD headless : keypoints numpy float32 (régression pygame-ce 'invalid start_pos')."""
import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from webcamdance import hud  # noqa: E402


def test_webcam_skeleton_numpy():
    pygame.init()
    surf = pygame.Surface((1280, 720))
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0] = np.linspace(100, 300, 17)
    kp[:, 1] = np.linspace(100, 500, 17)
    kp[:, 2] = 0.9
    frame = np.zeros((720, 1280, 3), np.uint8)
    hud.webcam(surf, frame, [kp, None], pygame.Rect(160, 60, 960, 540), 0.35)
    pygame.quit()
