"""(Re)calcule ou force le bandeau coach d'une chanson importée.

    python tools/set_band.py songs/<nom> [--band Y0 Y1] [--preview]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webcamdance.config import Config  # noqa: E402
from webcamdance.trim import coach_band  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--band", nargs=2, type=float, metavar=("Y0", "Y1"))
    ap.add_argument("--preview", action="store_true", help="montre 6 images avec le bandeau")
    a = ap.parse_args()
    song, cfg = Path(a.song), Config()
    meta_p = song / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(song / "video.mp4"))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    band = a.band or coach_band(np.load(song / "poses.npz")["kpts"], H, cfg.kpt_conf)
    meta["band"] = [float(band[0]), float(band[1])]
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"bandeau : {band[0]:.3f} → {band[1]:.3f}")
    if a.preview:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for i in np.linspace(0, n - 1, 8)[1:-1].astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                continue
            y0, y1 = int(band[0] * H), int(band[1] * H)
            f[:y0] //= 4
            f[y1:] //= 4
            cv2.line(f, (0, y0), (f.shape[1], y0), (0, 255, 255), 3)
            cv2.line(f, (0, y1 - 1), (f.shape[1], y1 - 1), (0, 255, 255), 3)
            tiles.append(cv2.resize(f, (480, 270)))
        img = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
        cv2.imshow("Bandeau coach (une touche pour fermer)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
