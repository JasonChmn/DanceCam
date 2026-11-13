"""Scoring : similarité cosinus d'os unitaires, pondérée mouvement, meilleur lag constant par move."""
import numpy as np
from scipy.ndimage import uniform_filter1d

from .skeleton import BONE_W, bone_features


def _scan(mask, reverse):
    """(T, N) bool -> (T, N) int : indice de la prochaine (reverse) ou dernière frame True, -1 sinon."""
    T, N = mask.shape
    out = np.full((T, N), -1, int)
    cur = np.full(N, -1)
    rng = range(T - 1, -1, -1) if reverse else range(T)
    for t in rng:
        cur = np.where(mask[t], t, cur)
        out[t] = cur
    return out


class CoachTrack:
    def __init__(self, npz_path, cfg):
        d = np.load(npz_path)
        self.kpts = d["kpts"].astype(np.float32)  # (T, N, 17, 3)
        self.fps = float(d["fps"])
        self.T, self.n = self.kpts.shape[:2]
        self.cfg = cfg
        self.U, self.V = bone_features(self.kpts, cfg.kpt_conf)
        # vitesse angulaire par os -> les os qui bougent pèsent plus (sinon rester immobile paie)
        ang = np.arctan2(self.U[..., 1], self.U[..., 0])
        dang = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=0)))) * self.fps
        dang *= self.V[1:] & self.V[:-1]
        omega = np.concatenate([dang[:1], dang], 0) if self.T > 1 else np.zeros_like(ang)
        omega = uniform_filter1d(omega, size=max(3, int(0.3 * self.fps)), axis=0, mode="nearest")
        self.W = BONE_W * (cfg.static_weight + np.clip(omega / cfg.omega0, 0, 2))  # (T, N, B)
        self.cover = (BONE_W * self.V).sum(-1) / BONE_W.sum()            # (T, N)
        self.theta_max = np.deg2rad(cfg.theta_max_deg)
        self.lags = np.arange(-int(cfg.win_back * self.fps), int(cfg.win_fwd * self.fps) + 1)
        # présence et repères par frame (tag flottant, formation de départ)
        ok = self.kpts[..., 2] >= cfg.kpt_conf
        self.present = ok.sum(-1) >= 5                                       # (T, N)
        sh = ok[..., 5] & ok[..., 6]
        x = np.where(sh, self.kpts[..., 5:7, 0].mean(-1), np.nan)
        ytop = np.where(ok, self.kpts[..., 1], np.inf).min(-1)
        self.head = np.stack([x, np.where(self.present & sh, ytop, np.nan)], -1)  # (T, N, 2) px vidéo
        self.next_idx = _scan(self.present & sh, reverse=True)                 # prochaine frame visible
        self.last_idx = _scan(self.present & sh, reverse=False)                # dernière frame visible

    def formation(self, width, first=60):
        """x normalisé [0,1] de chaque coach sur ses `first` premières frames visibles
        (sa place dans la formation de départ) et frame de première apparition."""
        xs, t0 = [], []
        for c in range(self.n):
            idx = np.flatnonzero(np.isfinite(self.head[:, c, 0]))
            xs.append(float(np.median(self.head[idx[:first], c, 0])) / width if len(idx) else 0.5)
            t0.append(int(idx[0]) if len(idx) else self.T)
        return np.array(xs), np.array(t0)

    def _rng(self, t0, t1):
        return max(0, int(t0 * self.fps)), min(self.T, int(t1 * self.fps) + 1)

    def active(self, t0, t1, c):
        """Le coach c est-il suffisamment visible sur [t0, t1] pour juger ?"""
        i0, i1 = self._rng(t0, t1)
        return i1 > i0 and self.cover[i0:i1, c].mean() >= 0.5

    def count_active(self, c, period):
        n = int(np.ceil(self.T / self.fps / period))
        return sum(self.active(w * period, (w + 1) * period, c) for w in range(n))

    def frame_scores(self, kp_player, t, c):
        """Scores [0,1] d'une pose joueur (17,3) au temps chanson t vs coach c, un par décalage
        temporel candidat (grille de lags, NaN si non jugeable). -> (K,)

        Le lag est choisi ensuite PAR MOVE (un seul décalage constant par fenêtre), pas par
        frame : tolère la latence/le retard du joueur sans le sur-ajustement d'un max par frame
        ni d'un DTW (Dynamic Time Warping) libre.
        """
        cfg = self.cfg
        idx = int(round(t * self.fps)) + self.lags
        inb = (idx >= 0) & (idx < self.T)
        idx = np.clip(idx, 0, self.T - 1)
        u, v = bone_features(kp_player, cfg.kpt_conf)                    # (B,2), (B,)
        Uc, Vc, W = self.U[idx, c], self.V[idx, c], self.W[idx, c]
        cos = np.clip((Uc * u).sum(-1), -1.0, 1.0)                       # (K, B)
        s = np.clip(1.0 - np.arccos(cos) / self.theta_max, 0.0, 1.0)
        both = Vc & v
        num = (W * s * both).sum(1)
        den = (W * both).sum(1)
        # os masqués chez le joueur (occlusion, détection ratée) : ignorés, mais il faut en
        # montrer assez (part pondérée des os visibles du coach) pour que la frame compte
        seen = den / np.maximum((W * Vc).sum(1), 1e-6)
        ok = inb & (self.cover[idx, c] >= cfg.min_cover) & (seen >= cfg.player_min_cover) & (den > 1e-6)
        return np.where(ok, num / np.maximum(den, 1e-6), np.nan)


class PlayerScore:
    """Agrège les scores par frame en jugements par 'move' (fenêtre de move_period s)."""

    def __init__(self, cfg, coach, coach_idx):
        self.cfg, self.coach, self.c = cfg, coach, coach_idx
        self.acc = {}
        self.w = 0
        self.total = 0
        self.max_total = max(1, coach.count_active(coach_idx, cfg.move_period)) * cfg.judgements[0][2]
        self.combo = self.best_combo = 0
        self.last = None            # (label, t_chanson)
        self.counts = {j[1]: 0 for j in cfg.judgements}
        self.skipped = 0            # moves non jugés (joueur pas assez détecté) : ni points ni MISS
        self.history = []           # un dict par move jugé (log/analyse)

    def add(self, t, kp):
        if t >= 0:
            self.acc.setdefault(int(t / self.cfg.move_period), []).append(self.coach.frame_scores(kp, t, self.c))

    def close_due(self, t):
        p = self.cfg.move_period
        while (self.w + 1) * p + self.cfg.grace <= t:
            vals = self.acc.pop(self.w, [])
            if self.coach.active(self.w * p, (self.w + 1) * p, self.c):
                A = np.stack(vals) if vals else np.full((0, len(self.coach.lags)), np.nan)
                nvalid = int((~np.isnan(A)).any(1).sum())      # frames où le joueur était jugeable
                if nvalid < self.cfg.judge_min_frames:
                    # joueur pas (assez) détecté : non jugé, retiré du maximum, combo conservé
                    self.skipped += 1
                    self.max_total -= self.cfg.judgements[0][2]
                    self.history.append({"move": self.w, "t": self.w * p, "score": None, "lag": None,
                                         "label": "SKIP", "frames": nvalid})
                    self.w += 1
                    continue
                # moyenne par lag sur le move, puis meilleur lag (assez de frames exigées)
                cnt = (~np.isnan(A)).sum(0)
                m = np.where(cnt >= max(1, nvalid // 2), np.nansum(A, 0) / np.maximum(cnt, 1), 0.0)
                k = int(m.argmax())
                s, lag = float(m[k]), float(self.coach.lags[k] / self.coach.fps)
                thr, label, pts = next(j for j in self.cfg.judgements if s >= j[0])
                self.total += pts
                self.counts[label] += 1
                self.combo = self.combo + 1 if pts >= 50 else 0
                self.best_combo = max(self.best_combo, self.combo)
                self.last = (label, (self.w + 1) * p + self.cfg.grace)
                self.history.append({"move": self.w, "t": self.w * p, "score": round(s, 4), "lag": lag,
                                     "label": label, "frames": nvalid})
            self.w += 1

    @property
    def ratio(self):
        return self.total / max(1, self.max_total)

    @property
    def stars(self):
        return min(5, int(self.ratio / 0.16))


def assign_coaches(n_players, coach_x, presence, offset=0):
    """Association joueurs -> coachs par RANG (pas par position exacte).

    Joueurs triés gauche->droite, coachs triés par place dans la formation de départ :
    - autant de joueurs que de coachs : rang i -> rang i ;
    - moins de joueurs : coachs pris à intervalles réguliers ; 1 seul joueur : coach principal
      (le plus présent dans la chanson) ;
    - plus de joueurs : coachs partagés dans l'ordre.
    offset décale toute l'association d'un cran (touches ←/→), circulairement.
    -> liste d'indices de coach, un par joueur (gauche->droite)."""
    order = np.argsort(coach_x)
    nc, n = len(order), n_players
    if n == nc:
        ranks = np.arange(n)
    elif n < nc:
        if n == 1:
            ranks = np.array([int(np.flatnonzero(order == int(np.argmax(presence)))[0])])
        else:
            ranks = np.round(np.arange(n) * (nc - 1) / (n - 1)).astype(int)
    else:
        ranks = np.arange(n) * nc // n
    return order[(ranks + offset) % nc].tolist()
