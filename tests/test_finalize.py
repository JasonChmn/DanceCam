"""Encodage final recadré sur le bandeau : hauteur vidéo, recalage des poses, bandeau relatif."""
import subprocess

import cv2
import numpy as np

from webcamdance.config import Config
from webcamdance.import_song import finalize_video
from webcamdance.media import ffmpeg_exe


def test_finalize_crop(tmp_path):
    H = 540
    full = tmp_path / "_full.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=960x{H}:rate=30", "-frames:v", "10", "-pix_fmt", "yuv420p", str(full)], check=True)
    K = np.zeros((10, 1, 17, 3), np.float32)
    K[..., 0], K[..., 1], K[..., 2] = 480, np.linspace(200, 500, 17), 0.9     # coach de y=200 à 500
    K[5, 0] = 0                                                                # frame sans coach
    np.savez_compressed(tmp_path / "poses.npz", kpts=K, fps=np.float32(30))
    band, crop = finalize_video(full, tmp_path, Config())
    cap = cv2.VideoCapture(str(tmp_path / "video.mp4"))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0 = round(crop[0] * H)
    assert h == round((crop[1] - crop[0]) * H) and h % 2 == 0 and h < H
    K2 = np.load(tmp_path / "poses.npz")["kpts"]
    assert np.allclose(K2[0, 0, :, 1], K[0, 0, :, 1] - y0)                   # poses recalées
    assert (K2[5] == 0).all()                                                 # absent reste à 0
    assert 0.0 < band[0] < 0.25 and band[1] > 0.9                            # bandeau dans la vidéo recadrée
    # la tête du coach (bandeau) tombe bien dans l'image recadrée
    assert band[0] * h <= K2[0, 0, :, 1].min()
