"""Régression : les threads ne doivent pas masquer Thread._stop (join() plantait)."""
import time

import numpy as np

from webcamdance.capture import PlayerWorker
from webcamdance.config import Config
from webcamdance.pose import _EMPTY


class _FakeCam:
    w, h = 640, 480

    def __init__(self):
        self.i = 0

    def latest(self):
        self.i += 1
        return self.i, np.zeros((self.h, self.w, 3), np.uint8), time.perf_counter()


def test_worker_stop_join():
    w = PlayerWorker(_FakeCam(), lambda f: [_EMPTY], Config(), 2)
    w.start()
    time.sleep(0.05)
    w.stop()
    w.join(timeout=2)
    assert not w.is_alive()
