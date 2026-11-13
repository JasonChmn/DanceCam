"""Affectation détections -> slots (joueurs ou coachs) + lissage One-Euro."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton import L_HIP, R_HIP


def select_people(boxes, k, min_rel_h=0.45):
    """Indices des k plus grandes personnes (hauteur bbox), en écartant l'arrière-plan."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    h = boxes[:, 3] - boxes[:, 1]
    idx = np.argsort(-h)
    return idx[h[idx] >= min_rel_h * h[idx[0]]][:k]


def centers_scales(kpts, boxes, thr):
    """Centre = milieu des hanches si visibles (stable bras levés), sinon centre bbox."""
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    ok = (kpts[:, L_HIP, 2] >= thr) & (kpts[:, R_HIP, 2] >= thr)
    hips = 0.5 * (kpts[:, L_HIP, :2] + kpts[:, R_HIP, :2])
    return np.where(ok[:, None], hips, c).astype(np.float32), (boxes[:, 3] - boxes[:, 1]).astype(np.float32)


class SlotTracker:
    """N slots fixes. Coût = continuité (bornée à 1) + lane_weight * distance au couloir.

    lane_weight élevé + anchor_adapt=0  -> identité = couloir (joueurs) : après un
    croisement, l'affectation revient d'elle-même au bon joueur.
    lane_weight faible + anchor_adapt>0 -> suivi par continuité (coachs qui se déplacent).
    """

    def __init__(self, n, frame_w, lane_weight=1.5, anchor_adapt=0.0, timeout=1.0):
        self.n, self.lane_w = n, frame_w / n
        self.lane_weight, self.anchor_adapt, self.timeout = lane_weight, anchor_adapt, timeout
        self.max_cost = 1.0 + 1.5 * lane_weight
        self.ready = False

    def reset(self):
        self.ready = False

    def init(self, centers, scales, t):
        """centers/scales : exactement n personnes, triées par x croissant."""
        self.pos = np.array(centers, np.float32)
        self.scale = np.array(scales, np.float32)
        self.anchor = self.pos[:, 0].copy()
        self.seen = np.full(self.n, t, np.float64)
        self.ready = True

    def update(self, centers, scales, t):
        """-> liste de longueur n : indice de détection par slot, ou None."""
        out = [None] * self.n
        m = len(centers)
        if m == 0:
            return out
        C = np.empty((self.n, m), np.float32)
        for i in range(self.n):
            lane = np.abs(centers[:, 0] - self.anchor[i]) / self.lane_w
            if t - self.seen[i] < self.timeout:
                cont = np.minimum(np.linalg.norm(centers - self.pos[i], axis=1) / self.scale[i], 1.0)
            else:  # slot perdu : réacquisition par couloir seul
                cont = np.ones(m, np.float32)
            C[i] = cont + self.lane_weight * lane
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > self.max_cost:
                continue
            out[i] = int(j)
            self.pos[i] = centers[j]
            self.scale[i] += 0.2 * (scales[j] - self.scale[i])
            self.anchor[i] += self.anchor_adapt * (centers[j, 0] - self.anchor[i])
            self.seen[i] = t
        return out


class OneEuro:
    """Filtre One-Euro vectorisé (Casiez et al. 2012). mask=False -> point gelé."""

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t, mask=None):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        if mask is not None:
            x = np.where(mask, x, self.x)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class Kalman2D:
    """Kalman vitesse constante sur (x, y). Bruit de mesure et d'accélération relatifs à
    l'échelle de la personne (hauteur bbox) : même comportement quelle que soit la distance."""

    def __init__(self, xy, scale, t, sigma_meas=0.03, sigma_acc=4.0):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 4.0, 4.0]) * (0.1 * scale) ** 2
        self.t, self.scale = t, scale
        self.sm, self.sa = sigma_meas, sigma_acc

    def predict(self, t):
        """État prédit à t sans le modifier -> (x, y)."""
        dt = max(t - self.t, 0.0)
        return self.x[:2] + self.x[2:] * dt

    def update(self, xy, scale, t):
        dt = max(t - self.t, 1e-3)
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = (self.sa * scale) ** 2
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q
        H = np.eye(2, 4)
        R = np.eye(2) * (self.sm * scale) ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(xy, float) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.t, self.scale = t, scale


class PlayerTracker:
    """N joueurs. Identité = couleur de la tenue (mémorisée au verrouillage) + continuité
    de mouvement (Kalman) ; le couloir de départ n'est qu'un faible départage, car les
    chorégraphies peuvent faire échanger les places.

    Coût(slot, détection) = W_APP * Hellinger(signature, prototype)
                          + W_POS * min(écart à la position prédite / échelle, 1.5) / 1.5
                          + W_LANE * écart au couloir initial / largeur couloir
    """
    W_APP, W_POS, W_LANE = 1.0, 0.6, 0.1
    APP_GATE = 0.55       # couleur trop différente : jamais ce joueur
    MAX_COST = 1.0
    MEMORY = 0.7          # s pendant lesquelles la prédiction Kalman est utilisée
    PROTO_RATE = 0.05     # adaptation lente du prototype couleur (éclairage)

    def __init__(self, n, frame_w, timeout=1.0):
        self.n, self.lane_w, self.timeout = n, frame_w / n, timeout
        self.ready = False

    def reset(self):
        self.ready = False

    def init(self, centers, scales, sigs, t):
        """Exactement n personnes, triées par x croissant."""
        self.kf = [Kalman2D(c, s, t) for c, s in zip(centers, scales)]
        self.scale = np.array(scales, np.float32)
        self.anchor = np.array(centers, np.float32)[:, 0].copy()
        self.proto = np.array(sigs, np.float32)
        self.seen = np.full(self.n, t, np.float64)
        self.ready = True

    def _proto_update(self, i, sig):
        if np.isnan(self.proto[i]).any():
            self.proto[i] = sig
            return
        p = (1 - self.PROTO_RATE) * self.proto[i] + self.PROTO_RATE * sig
        from .appearance import NPART, PART
        for k in range(NPART):                          # renormalise chaque partie (racine d'histo)
            p[k * PART:(k + 1) * PART] /= max(np.linalg.norm(p[k * PART:(k + 1) * PART]), 1e-6)
        self.proto[i] = p

    def update(self, centers, scales, sigs, t, overlap=None):
        """-> liste de longueur n : indice de détection par slot, ou None."""
        from .appearance import distance
        out = [None] * self.n
        m = len(centers)
        if m == 0:
            return out
        sigs = np.asarray(sigs, np.float32)
        nan_d = np.isnan(sigs).any(1)
        C = np.empty((self.n, m), np.float32)
        app = np.full((self.n, m), np.nan, np.float32)
        for i in range(self.n):
            if not np.isnan(self.proto[i]).any():
                app[i, ~nan_d] = distance(sigs[~nan_d], self.proto[i][None])
            a = np.where(np.isnan(app[i]), 0.35, app[i])           # non mesurable : neutre
            if t - self.seen[i] < self.MEMORY:
                d = np.linalg.norm(centers - self.kf[i].predict(t), axis=1) / self.scale[i]
                pos = np.minimum(d, 1.5) / 1.5
            else:                                                   # perdu : couleur seule
                pos = np.full(m, 0.5, np.float32)
            lane = np.abs(centers[:, 0] - self.anchor[i]) / self.lane_w
            C[i] = self.W_APP * a + self.W_POS * pos + self.W_LANE * np.minimum(lane, 2.0)
            C[i][app[i] > self.APP_GATE] = 1e3
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > self.MAX_COST:
                continue
            out[i] = int(j)
            if t - self.seen[i] >= self.MEMORY:                     # réacquisition : repart à zéro
                self.kf[i] = Kalman2D(centers[j], scales[j], t)
            else:
                self.kf[i].update(centers[j], scales[j], t)
            self.scale[i] += 0.2 * (scales[j] - self.scale[i])
            self.seen[i] = t
            clean = overlap is None or overlap[j] < 0.2
            if not nan_d[j] and clean and (np.isnan(app[i, j]) or app[i, j] < 0.35):
                self._proto_update(i, sigs[j])
        return out
