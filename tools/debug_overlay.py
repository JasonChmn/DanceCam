"""Vidéo de contrôle : squelettes coachs incrustés + audio.  python tools/debug_overlay.py songs/<nom>"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance.config import Config  # noqa: E402
from webcamdance.media import ffmpeg_exe  # noqa: E402
from webcamdance.scoring import CoachTrack  # noqa: E402
from webcamdance.skeleton import EDGES  # noqa: E402

COLORS = [(255, 229, 0), (160, 64, 255), (0, 214, 255), (3, 255, 118)]  # BGR


def main(song):
    song = Path(song)
    cfg = Config()
    coach = CoachTrack(song / "poses.npz", cfg)
    cap = cv2.VideoCapture(str(song / "video.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = song / "debug_overlay.mp4"
    ff = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(coach.fps), "-i", "-", "-i", str(song / "audio.ogg"),
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i < coach.T:
            for c in range(coach.n):
                kp = coach.kpts[i, c]
                for a, b in EDGES:
                    if kp[a, 2] >= cfg.kpt_conf and kp[b, 2] >= cfg.kpt_conf:
                        cv2.line(f, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), COLORS[c % 4], 3)
                for j in range(5, 17):
                    if kp[j, 2] >= cfg.kpt_conf:
                        # gauche (indices impairs) en blanc, droite en rouge : contrôle G/D
                        cv2.circle(f, tuple(kp[j, :2].astype(int)), 5, (255, 255, 255) if j % 2 else (0, 0, 255), -1)
                if (kp[:, 2] > 0).any():
                    x, y = kp[5:7, :2].mean(0).astype(int)
                    cv2.putText(f, f"C{c} cov={coach.cover[i, c]:.2f}", (x - 60, max(20, y - 60)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[c % 4], 2)
        cv2.putText(f, f"t={i / coach.fps:7.2f}s  f={i}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ff.stdin.write(f.tobytes())
        i += 1
    ff.stdin.close()
    ff.wait()
    print(f"{out} ({i} frames)")


if __name__ == "__main__":
    main(sys.argv[1])
