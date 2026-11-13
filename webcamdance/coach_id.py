"""Identité des coachs (offline), par assemblage de segments (« tracklets »).

1. Segments sûrs : d'une frame à la suivante, une détection n'est reliée que si la position
   est sans ambiguïté (meilleur candidat mutuel, nettement plus proche que tout autre).
   Croisement, occultation, regroupement -> le segment est coupé. Un changement de tenue ne
   coupe rien (la position reste continue) : il est géré sans cas particulier.
2. Assemblage en N coachs : chaque segment est raccordé au coach libre le plus compatible,
   couleur AU RACCORD (fin du segment précédent vs début du nouveau : tenues proches dans le
   temps) + position prédite à travers le trou. Le poids de la couleur suit la séparation des
   tenues (forte : Tetris ; faible : tenues identiques -> la position décide).
L'ancien regroupement global par couleur reste disponible (identify_legacy) pour comparaison.
"""
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

from .appearance import distance
from .tracking import SlotTracker

MIN_SEPARATION = 0.35   # distance mini entre groupes ; mesuré Tetris : 0.73-0.86 entre, ~0.3 dedans
APP_GATE = 0.6          # au-delà, une détection n'est rattachée à aucun coach
POS_WEIGHT = 0.3        # poids du terme de continuité (départage)


def overlap_frac(boxes):
    """Fraction de chaque bbox recouverte par la plus chevauchante des autres. (M,)"""
    M = len(boxes)
    out = np.zeros(M, np.float32)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if w > 0 and h > 0:
                a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                out[i] = max(out[i], w * h / max(a, 1e-6))
    return out


def count_coaches(frames, q=75):
    """Nombre de coachs = percentile q du nombre de danseurs de premier plan par frame
    (la médiane sous-estime : les chevauchements font perdre des détections)."""
    counts = [int(f["fg"].sum()) for f in frames]
    return int(np.clip(round(float(np.percentile(counts, q))) if counts else 1, 1, 4))


def cluster(frames, n, seeds=10, max_samples=6000, rng=0):
    """k-means sur les signatures « propres » (premier plan, peu chevauchées).
    -> (centroïdes (n, D), séparation minimale entre centroïdes)."""
    X = [f["sig"][j] for f in frames for j in range(len(f["sig"]))
         if f["fg"][j] and f["ovl"][j] < 0.3 and not np.isnan(f["sig"][j]).any()]
    if len(X) < 3 * n:
        return None, 0.0
    X = np.asarray(X, np.float32)
    if len(X) > max_samples:
        X = X[np.random.default_rng(rng).choice(len(X), max_samples, replace=False)]
    if n == 1:
        return X.mean(0, keepdims=True), 1.0
    import warnings
    best = None
    for s in range(seeds):
        with warnings.catch_warnings():             # tenues identiques : groupes vides, sans gravité
            warnings.simplefilter("ignore")
            C, L = kmeans2(X, n, minit="++", seed=s)
        if len(np.unique(L)) < n:
            continue
        inertia = ((X - C[L]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C)
    if best is None:
        return None, 0.0
    C = best[1]
    D = distance(C[:, None], C[None])
    return C, float(D[~np.eye(n, dtype=bool)].min())


def assign_by_appearance(frames, C, fps, memory=0.5):
    """-> (T, n) int : indice de détection par coach et par frame, -1 si absent."""
    n, T = len(C), len(frames)
    out = np.full((T, n), -1, int)
    pos = np.zeros((n, 2), np.float32)
    scale = np.ones(n, np.float32)
    seen = np.full(n, -1e9)
    for t, f in enumerate(frames):
        m = len(f["sig"])
        if m == 0:
            continue
        sig = np.where(np.isnan(f["sig"]), 0.0, f["sig"])
        app = distance(sig[:, None], C[None])                        # (m, n)
        app[np.isnan(f["sig"]).any(1)] = APP_GATE                    # non mesurable : neutre
        recent = (t - seen) / fps < memory
        cont = np.minimum(np.linalg.norm(f["cen"][:, None] - pos[None], axis=-1) / scale[None], 1.0)
        cost = app + POS_WEIGHT * np.where(recent[None], cont, 0.5)
        nan = np.isnan(f["sig"]).any(1)
        for j, k in zip(*linear_sum_assignment(cost)):
            # signature mesurable : il faut la bonne couleur ; sinon (torse caché) : il faut
            # être tout près de la position récente du coach
            ok = (recent[k] and cont[j, k] < 0.5) if nan[j] else app[j, k] < APP_GATE
            if not ok:
                continue
            out[t, k] = j
            scale[k] = scale[k] + 0.2 * (f["sc"][j] - scale[k]) if recent[k] else f["sc"][j]
            pos[k], seen[k] = f["cen"][j], t
    return out


def assign_by_position(frames, n, fps, width):
    """Repli : suivi par continuité de position (ancien comportement)."""
    tracker = SlotTracker(n, width, lane_weight=0.3, anchor_adapt=0.05)
    out = np.full((len(frames), n), -1, int)
    lost = 0
    for t, f in enumerate(frames):
        idx = np.arange(min(len(f["cen"]), n + 2))       # n+2 plus grands
        cen, sc = f["cen"][idx], f["sc"][idx]
        if not tracker.ready:
            if len(idx) >= n:  # (ré)initialisation : n plus grands, triés gauche->droite
                order = np.argsort(cen[:n, 0])
                tracker.init(cen[order], sc[order], t / fps)
                out[t] = idx[order]
                lost = 0
            continue
        assign = tracker.update(cen, sc, t / fps)
        for k, j in enumerate(assign):
            if j is not None:
                out[t, k] = idx[j]
        lost = lost + 1 if (len(idx) >= n and all(j is None for j in assign)) else 0
        if lost > fps * 0.3:  # tous perdus alors que n danseurs visibles => coupe de plan
            tracker.reset()
    return out


def order_left_right(frames, assign, first=15):
    """Permutation des coachs triés par x médian sur leurs `first` premières apparitions (~0,5 s :
    la formation de départ, avant tout déplacement)
    (joueur p, à gauche, suit le coach p)."""
    n = assign.shape[1]
    xs = []
    for k in range(n):
        v = [frames[t]["cen"][j, 0] for t, j in enumerate(assign[:, k]) if j >= 0][:first]
        xs.append(np.median(v) if v else np.inf)
    return np.argsort(xs)


# ----------------------------------------------------------------------------- tracklets
LINK_D = 0.20        # déplacement max entre 2 frames (en hauteurs de corps) pour relier
AMBIGUITY = 2.5      # le 2e candidat doit être au moins 2,5x plus loin que le 1er
MIN_TRACK = 5        # frames mini d'un segment pour l'assembler
BATCH = 3            # segments démarrant à ±3 frames : assemblés ensemble (hongrois)
ACCEPT = 0.95        # coût max d'un raccord
OVERLAP = 6          # chevauchement toléré (frames) entre la fin d'un segment et le suivant


def build_tracklets(frames):
    """-> liste de segments, chacun = liste de (t, j) consécutifs, liaison sans ambiguïté."""
    tracks, active, prev = [], {}, None             # active : indice détection (frame t-1) -> segment
    for t, f in enumerate(frames):
        cen, sc = f["cen"], f["sc"]
        m = len(cen)
        cur = {}
        if prev is not None and m and len(prev[0]):
            pc, ps = prev
            s = np.sqrt(np.maximum(sc[:, None], 1) * np.maximum(ps[None], 1))
            D = np.linalg.norm(cen[:, None] - pc[None], axis=-1) / s          # (m, m_prev)
            ratio = np.abs(np.log(np.maximum(sc[:, None], 1) / np.maximum(ps[None], 1)))
            D = np.where(ratio < 0.35, D, np.inf)                            # taille incohérente
            for j in range(m):
                i = int(np.argmin(D[j]))
                d1 = D[j, i]
                if not np.isfinite(d1) or d1 > LINK_D or int(np.argmin(D[:, i])) != j:
                    continue
                others = np.r_[np.delete(D[j], i), np.delete(D[:, i], j)]
                if len(others) and np.min(others) < AMBIGUITY * max(d1, 0.02):
                    continue                                                 # ambigu : on coupe
                if i in active:
                    cur[j] = active[i]
                    tracks[cur[j]].append((t, j))
        for j in range(m):
            if j not in cur:
                cur[j] = len(tracks)
                tracks.append([(t, j)])
        active, prev = cur, (cen, sc)
    return tracks


def _seg_info(tr, frames, k=15):
    """Couleur au début / à la fin (médiane des signatures mesurables), position et vitesse aux bords."""
    def app(part):
        S = [frames[t]["sig"][j] for t, j in part if not np.isnan(frames[t]["sig"][j]).any()
             and frames[t]["ovl"][j] < 0.3]
        return np.median(S, 0) if S else None
    pos = lambda t, j: frames[t]["cen"][j]
    (t0, j0), (t1, j1) = tr[0], tr[-1]
    v_end = (pos(t1, j1) - pos(*tr[max(0, len(tr) - 6)])) / max(1, t1 - tr[max(0, len(tr) - 6)][0])
    return {"t0": t0, "t1": t1, "app0": app(tr[:k]), "app1": app(tr[-k:]), "p0": pos(t0, j0), "p1": pos(t1, j1),
            "v1": v_end, "sc": float(np.median([frames[t]["sc"][j] for t, j in tr])), "n": len(tr)}


def stitch(tracks, frames, n, fps, w_app):
    """Assemble les segments en n coachs. -> (T, n) indices de détection, -1 si absent."""
    # segments de premier plan seulement (taille des coachs principaux la majorité du temps) :
    # les danseurs d'arrière-plan occuperaient sinon les coachs momentanément libres
    fg = lambda tr: np.mean([frames[t]["fg"][j] for t, j in tr]) >= 0.5
    infos = sorted((_seg_info(tr, frames) | {"tr": tr} for tr in tracks if len(tr) >= MIN_TRACK and fg(tr)),
                   key=lambda x: x["t0"])
    last = [None] * n                               # dernier segment de chaque coach
    out = np.full((len(frames), n), -1, int)

    def cost(k, S):
        L = last[k]
        if L is None:
            return 0.5                              # coach encore jamais vu : neutre
        gap = (S["t0"] - L["t1"]) / fps
        a = 0.5 if L["app1"] is None or S["app0"] is None else float(distance(L["app1"], S["app0"]))
        pred = L["p1"] + L["v1"] * min(S["t0"] - L["t1"], 0.5 * fps)          # extrapolation ≤ 0.5 s
        d_pred = float(np.linalg.norm(S["p0"] - pred)) / max(L["sc"], 1.0)
        d_last = float(np.linalg.norm(S["p0"] - L["p1"])) / max(L["sc"], 1.0)   # réapparaît où il a disparu
        d = d_pred if gap < 0.5 else d_last         # trou court : trajectoire ; trou long : où il a disparu
        pos = min(d, 2.0) / 2.0
        return (w_app * a + pos) / (w_app + 1.0)

    i = 0
    while i < len(infos):
        batch = [S for S in infos[i:] if S["t0"] <= infos[i]["t0"] + BATCH]
        i += len(batch)
        # libre = dernier segment fini (tolérance OVERLAP frames : un débris de détection qui
        # chevauche le retour du coach ne doit pas le bloquer)
        free = [k for k in range(n) if last[k] is None or last[k]["t1"] < min(S["t0"] for S in batch) + OVERLAP]
        if not free:
            continue                                # plus de segments que de coachs : figurant, faux positif
        C = np.array([[cost(k, S) for k in free] for S in batch])
        for r, c in zip(*linear_sum_assignment(C)):
            if C[r, c] > ACCEPT:
                continue
            S, k = batch[r], free[c]
            last[k] = S
            for t, j in S["tr"]:
                out[t, k] = j
    return out


def identity_jumps(frames, assign, thr=0.6):
    """Contrôle : nombre de sauts de position > thr hauteurs de corps entre frames consécutives
    d'un même coach (un échange d'identité en produit un)."""
    jumps = 0
    for k in range(assign.shape[1]):
        prev = None
        for t, j in enumerate(assign[:, k]):
            if j < 0:
                prev = None
                continue
            p, sc = frames[t]["cen"][j], frames[t]["sc"][j]
            if prev is not None and np.linalg.norm(p - prev) / max(sc, 1) > thr:
                jumps += 1
            prev = p
    return jumps


def _clean_sigs(frames, tr):
    return [frames[t]["sig"][j] for t, j in tr
            if not np.isnan(frames[t]["sig"][j]).any() and frames[t]["ovl"][j] < 0.3]


def outfit_groups(frames, tracks, n, min_seg=60, k_over=2):
    """Changements de tenue : si beaucoup de longs segments changent de couleur en cours de route,
    on regroupe les couleurs en k_over*n tenues, puis on relie deux tenues portées dans un même
    segment continu (même personne, position continue). -> liste de n groupes de centroïdes, ou None."""
    C, _ = cluster(frames, n)
    longs = [tr for tr in tracks if len(tr) >= min_seg]
    switch = 0
    for tr in longs:
        S = _clean_sigs(frames, tr)
        if len(S) < 30:
            continue
        lab = np.argmin(distance(np.asarray(S)[:, None], C[None]), 1)
        third = len(lab) // 3
        if np.bincount(lab[:third], minlength=n).argmax() != np.bincount(lab[-third:], minlength=n).argmax():
            switch += 1
    if not longs or switch < max(2, 0.25 * len(longs)):
        return None                                     # pas de changement de tenue
    Co, _ = cluster(frames, k_over * n)
    if Co is None:
        return None
    K = len(Co)
    link = np.zeros((K, K))
    for tr in longs:
        S = _clean_sigs(frames, tr)
        if len(S) < 30:
            continue
        cnt = np.bincount(np.argmin(distance(np.asarray(S)[:, None], Co[None]), 1), minlength=K)
        big = np.flatnonzero(cnt >= 10)
        for a in big:
            for b in big:
                if a < b:
                    link[a, b] += min(cnt[a], cnt[b])
    groups = [{k} for k in range(K)]                    # fusion par liens les plus forts jusqu'à n groupes
    for a, b in sorted(zip(*np.nonzero(link)), key=lambda ab: -link[ab]):
        ga = next(g for g in groups if a in g)
        gb = next(g for g in groups if b in g)
        if ga is not gb and len(groups) > n:
            groups.remove(gb)
            ga |= gb
    if len(groups) != n:
        return None
    return [Co[sorted(g)] for g in groups]


MAX_CONFLICT = 0.20   # au-delà, la couleur ne distingue pas les coachs présents ensemble


def color_conflict(frames, C):
    """Part des frames où deux coachs de premier plan présents ensemble ont la même tenue la plus
    proche (coachs « jumeaux », tenues qui changent) : la couleur ne peut alors pas les distinguer.
    Mesuré : Tetris 0 %, Waka Waka 4 %, If You Wanna Party 46 %, Bang Bang Bang 100 %."""
    conf = tot = 0
    for f in frames:
        idx = [j for j in range(len(f["cen"])) if f["fg"][j] and not np.isnan(f["sig"][j]).any() and f["ovl"][j] < 0.3]
        if len(idx) >= 2:
            lab = np.argmin(distance(f["sig"][idx][:, None], C[None]), 1)
            tot += 1
            conf += len(set(lab.tolist())) < len(lab)
    return conf / max(tot, 1)


def identify(frames, n, fps):
    """Identités des coachs. -> (assign (T, n), description).

    Segments sûrs (build_tracklets) identifiés chacun par sa couleur moyenne sur toute sa durée,
    comparée aux tenues de toute la vidéo (regroupement global) ; affectation exclusive par frame.
    Changements de tenue : outfit_groups. Tenues indiscernables : assemblage par position."""
    tracks = build_tracklets(frames)
    C, sep = cluster(frames, n) if n > 1 else (None, 1.0)
    if n == 1:
        C, _ = cluster(frames, 1)
    conflict = color_conflict(frames, C) if (C is not None and n > 1) else 0.0
    if C is None or (n > 1 and (sep < MIN_SEPARATION or conflict > MAX_CONFLICT)):
        # couleur non fiable (tenues identiques, coachs « jumeaux », tenues qui changent) :
        # segments assemblés par la position (trajectoire à travers les trous courts,
        # dernière position connue pour les longs)
        assign = stitch(tracks, frames, n, fps, w_app=0.25)
        return assign[:, order_left_right(frames, assign)], (
            f"segments par position (couleur non fiable : séparation {sep:.2f}, conflit {conflict:.0%})")
    groups = outfit_groups(frames, tracks, n) if n > 1 else None
    if groups is None:
        groups = [C[k:k + 1] for k in range(n)]
        how = f"segments + couleur (séparation {sep:.2f})"
    else:
        how = f"segments + couleur, changements de tenue détectés ({sum(len(g) for g in groups)} tenues)"
    dist_to = lambda S: np.stack([distance(np.asarray(S)[:, None], g[None]).min(1) for g in groups], 1)
    seg_of, seg_cost = {}, {}
    for s_id, tr in enumerate(tracks):
        for t, j in tr:
            seg_of[(t, j)] = s_id
        S = _clean_sigs(frames, tr)
        if len(S) >= 3:
            seg_cost[s_id] = dist_to(S).mean(0)          # (n,) couleur moyenne du segment vs chaque coach
    out = np.full((len(frames), n), -1, int)
    last_pos, last_t = np.zeros((n, 2), np.float32), np.full(n, -10 ** 9)
    for t, f in enumerate(frames):
        m = len(f["cen"])
        if m == 0:
            continue
        cost = np.full((m, n), np.inf, np.float32)
        for j in range(m):
            sid = seg_of[(t, j)]
            if sid in seg_cost and len(tracks[sid]) >= MIN_TRACK:
                c = seg_cost[sid].copy()
                c[c > c.min() + 0.08] = np.inf           # pas forcé vers un coach qu'il ne ressemble pas
                cost[j] = c
            else:                                         # détection isolée : couleur ET proximité exigées
                c = dist_to([f["sig"][j]])[0] + 0.05 if not np.isnan(f["sig"][j]).any() else np.full(n, APP_GATE - 0.01)
                near = (t - last_t < 0.5 * fps) & (np.linalg.norm(last_pos - f["cen"][j], axis=1)
                                                    < 0.5 * max(f["sc"][j], 1.0))
                cost[j] = np.where(near, c, np.inf)
        big = np.where(np.isfinite(cost), cost, 1e3)
        for j, k in zip(*linear_sum_assignment(big)):
            if cost[j, k] < APP_GATE:
                out[t, k] = j
                last_pos[k], last_t[k] = f["cen"][j], t
    return out[:, order_left_right(frames, out)], how


def identify_legacy(frames, n, fps, width):
    """Ancienne méthode : regroupement global par couleur, repli par position."""
    C, sep = cluster(frames, n)
    if C is not None and (n == 1 or sep >= MIN_SEPARATION):
        assign = assign_by_appearance(frames, C, fps)
    else:
        assign = assign_by_position(frames, n, fps, width)
    return assign[:, order_left_right(frames, assign)], f"ancienne méthode (séparation {sep:.2f})"
