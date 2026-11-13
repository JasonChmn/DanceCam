"""Analyse d'un journal de partie.  python tools/analyze_log.py logs/<fichier>.jsonl"""
import json
import sys
from collections import Counter

import numpy as np


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    perf = [r for r in rows if r["type"] == "perf"]
    moves = [r for r in rows if r["type"] == "move"]
    hdr = rows[0]
    print(f"{path} : {hdr.get('players')} joueur(s), cam_latency {hdr['cfg']['cam_latency'] * 1000:.0f} ms")
    if perf:
        f = lambda k: np.array([r[k] for r in perf if r.get(k) is not None], float)
        for k in ("render_fps", "infer_ms", "cam_fps", "video_lag_frames"):
            v = f(k)
            if len(v):
                print(f"  {k:17s} moy {v.mean():6.1f}  min {v.min():6.1f}  p5 {np.percentile(v, 5):6.1f}  max {v.max():6.1f}")
    for p in sorted({m["player"] for m in moves}):
        allmv = [m for m in moves if m["player"] == p]
        mv = [m for m in allmv if m["label"] != "SKIP"]
        if len(allmv) > len(mv):
            print(f"  P{p + 1}: {len(allmv) - len(mv)} move(s) non jugé(s) (joueur pas assez détecté)")
        if not mv:
            continue
        s = np.array([m["score"] for m in mv])
        lags = np.array([m["lag"] for m in mv if m["lag"] is not None]) * 1000
        print(f"  P{p + 1}: {len(mv)} moves, score moy {s.mean():.3f} p10 {np.percentile(s, 10):.3f} "
              f"p50 {np.median(s):.3f} p90 {np.percentile(s, 90):.3f}  {dict(Counter(m['label'] for m in mv))}")
        if len(lags):
            print(f"      lag retenu (ms) : médiane {np.median(lags):+.0f}  moy {lags.mean():+.0f}  "
                  f"histo {dict(sorted(Counter(np.round(lags / 33.3).astype(int) * 33).items()))}")
    tracks = [r for r in rows if r["type"] == "track"]
    if tracks:
        X = np.array([[np.nan if x is None else x for x in r["x"]] for r in tracks], float)
        for p in range(X.shape[1]):
            x = X[:, p]
            jumps = np.flatnonzero(np.abs(np.diff(x)) > 200)       # saut > 200 px en 1 frame : échange probable
            print(f"  suivi P{p + 1}: présent {np.isfinite(x).mean() * 100:.0f}% des frames, sauts >200px : {len(jumps)}"
                  + (f" à t={[round(tracks[j]['t'], 1) for j in jumps[:8]]}" if len(jumps) else ""))
    for r in rows:
        if r["type"] == "result":
            print(f"  résultat P{r['player'] + 1}: {r['total']} pts, ratio {r['ratio']:.2f}, {r['stars']}★, {r['counts']}")


if __name__ == "__main__":
    main(sys.argv[1])
