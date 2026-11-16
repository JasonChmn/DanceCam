"""Bandeau des mouvements à venir (bas-droite de la vidéo coach), à la manière des pictos Just Dance.

Les pictos d'origine sont souvent coupés par le recadrage : on les masque sous un bandeau noir et
on fait défiler nos propres silhouettes, tirées des poses des coachs (poses.npz).

- Poses clés : minima locaux de la vitesse des membres (moyenne des coachs), espacées d'au moins
  `gap` s (plus pour 3-4 coachs, dont les pictos de groupe sont plus larges).
- Picto de groupe : un emplacement fixe par coach, ordre gauche->droite selon la position (lissée)
  des coachs à cet instant ; même échelle pour tous. Coach suivi : couleur de son joueur ; autres :
  couleur propre atténuée.
- Une barre de validation par emplacement : chaque silhouette l'atteint au moment de la pose ; la
  couleur de la barre est celle du coach qui occupe l'emplacement à l'image courante.

Poses clés et ordre des coachs : calculés au chargement (~0,1 s). Images des pictos (~20 ms
chacune) : dessinées à l'avance par un thread, pour ne jamais ralentir l'affichage.
"""
import threading
import time
import warnings

import cv2
import numpy as np
import pygame
from scipy.ndimage import uniform_filter1d

from . import hud

COACH_COLORS = [(255, 160, 60), (190, 120, 220), (120, 210, 110), (255, 120, 170)]   # RGB, coachs non suivis
LIMBS = [(5, 7, 9, "L"), (6, 8, 10, "R"), (11, 13, 15, "L"), (12, 14, 16, "R")]
DARK = (15, 15, 15)
HORIZON_PAST = 0.8      # s pendant lesquelles un picto passé s'efface


def keyframes(k, fps, thr, gap):
    """Indices des poses clés : instants où les coachs marquent une position."""
    ok = k[..., 2] >= thr
    d = np.linalg.norm(np.diff(k[..., :2], axis=0), axis=-1)               # (T-1, N, 17)
    both = ok[1:] & ok[:-1]
    torso = np.where(ok[..., 5] & ok[..., 11], np.linalg.norm(k[..., 5, :2] - k[..., 11, :2], axis=-1), np.nan)
    with warnings.catch_warnings():                                      # coach jamais vu : NaN attendu
        warnings.simplefilter("ignore", RuntimeWarning)
        sc = np.nan_to_num(np.nanmedian(torso, axis=0), nan=100.0)
    v = (d * both).sum(-1) / np.maximum(both.sum(-1), 1) / np.maximum(sc, 1)[None]
    v = uniform_filter1d(v.mean(1), max(1, int(0.25 * fps)))
    v = np.concatenate([v[:1], v])
    if len(v) < 3:
        return np.zeros(0, int)
    loc = np.where((v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
    lim = np.percentile(v, 60)
    chosen = []
    for i in loc[np.argsort(v[loc])]:
        if v[i] > lim:
            break
        if all(abs(i - j) >= gap * fps for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), int)


def fill_gaps(k, fps, thr, max_gap=0.5):
    """Articulations manquantes (cachées par une jupe, un bras…) interpolées depuis les images
    voisines, si l'écart ne dépasse pas max_gap s."""
    k = k.copy()
    t = np.arange(len(k))
    for n in range(k.shape[1]):
        for j in range(17):
            ok = k[:, n, j, 2] >= thr
            if ok.sum() < 2 or ok.all():
                continue
            idx = t[ok]
            pos = np.searchsorted(idx, t)
            prev = idx[np.clip(pos - 1, 0, len(idx) - 1)]
            nxt = idx[np.clip(pos, 0, len(idx) - 1)]
            fill = ~ok & (np.minimum(np.abs(t - prev), np.abs(nxt - t)) <= max_gap * fps)
            for c in (0, 1):
                k[fill, n, j, c] = np.interp(t[fill], idx, k[ok, n, j, c])
            k[fill, n, j, 2] = max(thr, 0.5)
    return k


def coach_order(k, fps, thr):
    """(T, N) : coachs triés de gauche à droite à chaque image (position lissée, trous comblés)."""
    ok = k[..., 2] >= thr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = np.nanmedian(np.where(ok, k[..., 0], np.nan), axis=-1)
    x[ok.sum(-1) < 5] = np.nan
    t = np.arange(len(x))
    for n in range(x.shape[1]):
        good = ~np.isnan(x[:, n])
        x[:, n] = np.interp(t, t[good], x[good, n]) if good.any() else n
    return np.argsort(uniform_filter1d(x, max(1, int(0.5 * fps)), axis=0), axis=1)


def _tint(c, f):
    """f > 0 : vers le blanc ; f < 0 : vers le noir."""
    return tuple(int(v + (255 - v) * f) if f > 0 else int(v * (1 + f)) for v in c)


def _figure(img, kp, color, th, ss, thr):
    """Silhouette d'un coach : torse plein, membres épais à contour sombre, tête ronde. Côté gauche
    plus clair ; membres peu sûrs (souvent derrière le corps) dessinés avant le torse, assombris."""
    ok = kp[:, 2] >= thr
    P = lambda i: (int(kp[i, 0] * ss), int(kp[i, 1] * ss))   # noqa: E731
    o, t = int((th + 2.2) * ss), max(1, int(th * ss))
    limbs = []
    for a, b, c, side in LIMBS:
        pts = [i for i in (a, b, c) if ok[i]]
        if len(pts) >= 2:
            limbs.append((kp[[a, b, c], 2].mean(), pts, side))
    limbs.sort(key=lambda l: l[0])

    def limb(pts, side, behind):
        col = _tint(color, 0.35 if side == "L" else -0.15)
        if behind:
            col = _tint(col, -0.35)
        for i, j in zip(pts, pts[1:]):
            cv2.line(img, P(i), P(j), DARK, o, cv2.LINE_AA)
        for i, j in zip(pts, pts[1:]):
            cv2.line(img, P(i), P(j), col, t, cv2.LINE_AA)
        for i in pts:
            cv2.circle(img, P(i), t // 2, col, -1, cv2.LINE_AA)

    for conf, pts, side in limbs:
        if conf < 0.6:
            limb(pts, side, True)
    if ok[[5, 6, 11, 12]].all():
        poly = np.array([P(5), P(6), P(12), P(11)], np.int32)
        cv2.polylines(img, [poly], True, DARK, o, cv2.LINE_AA)
        cv2.fillConvexPoly(img, poly, color, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, color, max(1, t // 2), cv2.LINE_AA)
    for conf, pts, side in limbs:
        if conf >= 0.6:
            limb(pts, side, False)
    head = [i for i in range(5) if ok[i]]
    if head and ok[5] and ok[6]:
        c = kp[head, :2].mean(0)
        r = max(np.linalg.norm(kp[5, :2] - kp[6, :2]) * 0.38, th * 1.2)
        cen = (int(c[0] * ss), int(c[1] * ss))
        cv2.circle(img, cen, int((r + 2.2) * ss), DARK, -1, cv2.LINE_AA)
        cv2.circle(img, cen, int(r * ss), _tint(color, 0.15), -1, cv2.LINE_AA)


def slots(n, h):
    """-> (largeur d'un emplacement, marge, largeur du picto de groupe) pour une hauteur h."""
    slot = int(0.62 * h)
    return slot, int(0.06 * h), slot * n + int(0.12 * h)


def render_group(kps, colors, alphas, h, order, thr, ss=3):
    """Picto de groupe RGBA (h, w, 4), ou None si aucun coach visible."""
    n = len(kps)
    slot, m, w = slots(n, h)
    oks = [kp[:, 2] >= thr for kp in kps]
    vis = [c for c in range(n) if oks[c].sum() >= 5]
    if not vis:
        return None
    bh = max(np.ptp(kps[c][oks[c], 1]) for c in vis)
    bw = max(np.ptp(kps[c][oks[c], 0]) for c in vis)
    sc = min(0.84 * h / max(bh, 1), 1.05 * slot / max(bw, 1))
    img = np.zeros((h * ss, w * ss, 4), np.uint8)
    for j, c in enumerate(order):
        if c not in vis:
            continue
        kp, ok = kps[c], oks[c]
        k2 = kp.copy()
        k2[:, 0] = (kp[:, 0] - (kp[ok, 0].max() + kp[ok, 0].min()) / 2) * sc + m + slot * (j + 0.5)
        k2[:, 1] = (kp[:, 1] - kp[ok, 1].max()) * sc + 0.94 * h
        layer = np.zeros((h * ss, w * ss, 3), np.uint8)
        _figure(layer, k2, colors[c], max(2.0, h * 0.075), ss, thr)
        mk = layer.any(-1)
        img[mk, :3] = layer[mk]
        img[mk, 3] = int(255 * alphas[c])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _surface(rgba):
    h, w = rgba.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA").convert_alpha()


class Pictos:
    AHEAD = 8.0         # s d'avance prises par le thread de rendu

    def __init__(self, kpts, fps, thr):
        self.fps, self.thr = fps, thr
        self.n = kpts.shape[1]
        self.gap = 1.1 if self.n <= 2 else 1.4
        self.kf = keyframes(kpts, fps, thr, self.gap)
        self.kf_t = self.kf / fps
        self.k = fill_gaps(kpts, fps, thr)
        self.order = coach_order(self.k, fps, thr)
        self.rect, self.cell_h, self._t, self._gen = None, 0, 0.0, 0
        self._ready, self._cache = {}, {}           # i -> (rgba, glow) prêts ; i -> surfaces pygame
        self._halt = False
        self.set_colors([])
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._halt = True

    def _reset(self):
        """Couleurs ou taille changées : tout redessiner."""
        self._gen += 1
        self._ready, self._cache = {}, {}

    def _render_loop(self):
        while not self._halt:
            gen, h, t = self._gen, self.cell_h, self._t
            a0, a1 = np.searchsorted(self.kf_t, [t - HORIZON_PAST, t + self.AHEAD])
            todo = [i for i in self.kf[a0:a1] if i not in self._ready] if h else []
            if not todo:
                time.sleep(0.05)
                continue
            i = todo[0]
            rgba = render_group(self.k[i], self.colors, self.alphas, h, self.order[i], self.thr)
            if rgba is not None:
                glow = cv2.GaussianBlur(rgba, (0, 0), max(1.0, h / 25))
                glow[..., :3] = 255
                rgba = (rgba, glow)
            if gen == self._gen:
                self._ready[i] = rgba

    def set_colors(self, cmap):
        """cmap : joueur p -> coach suivi. Coach suivi : couleur du (premier) joueur ; autres : atténués."""
        self.tracked = {}
        for p, c in enumerate(cmap):
            self.tracked.setdefault(c, hud.pcolor(p))
        self.colors = [self.tracked.get(c, COACH_COLORS[c % 4]) for c in range(self.n)]
        self.alphas = [1.0 if c in self.tracked else 0.6 for c in range(self.n)]
        self._reset()

    def layout(self, vrect):
        """Géométrie à partir du rectangle de la vidéo à l'écran (à rappeler au redimensionnement)."""
        pw, ph = int(vrect.w * 0.34), int(vrect.h * 0.21)
        self.rect = pygame.Rect(vrect.right - pw, vrect.bottom - ph, pw, ph)
        self.cell_h = max(20, int(ph * 0.74))
        slot, m, self.gw = slots(self.n, self.cell_h)
        self.bar_x = self.rect.x + int(pw * 0.30)                 # centre du groupe au moment de la pose
        self.bars = [self.bar_x - self.gw // 2 + m + int(slot * (j + 0.5)) for j in range(self.n)]
        self.speed = (self.gw + hud.px(12)) / self.gap               # px/s : deux pictos ne se touchent jamais
        self.horizon = (self.rect.right - self.bar_x + self.gw / 2) / self.speed
        self.ty = self.rect.y + (ph - self.cell_h) // 2 - hud.px(3)
        # fond : noir à 95 %, fondu sur le bord gauche
        self.bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        a = np.clip(np.arange(pw) / max(1, 0.10 * pw), 0, 1) * 0.95 * 255
        pygame.surfarray.pixels_alpha(self.bg)[:] = a[:, None].astype(np.uint8)
        self.card = pygame.Surface((self.gw + 8, self.cell_h + 8), pygame.SRCALPHA)
        pygame.draw.rect(self.card, (50, 50, 55, 128), self.card.get_rect(), border_radius=hud.px(6))
        self._reset()

    def _pic(self, i):
        """Surfaces du picto i, ou None s'il n'est pas encore prêt (ou vide)."""
        if i not in self._cache:
            if i not in self._ready:
                return None
            r = self._ready[i]
            self._cache[i] = None if r is None else (_surface(r[0]), _surface(r[1]))
        return self._cache[i]

    def draw(self, surf, t):
        if self.rect is None or not len(self.kf):
            return
        self._t = t
        r = self.rect
        surf.blit(self.bg, r)
        clip = surf.get_clip()
        surf.set_clip(r)
        cur = self.order[min(max(int(t * self.fps), 0), len(self.order) - 1)]
        for j, c in enumerate(cur):           # couleur = coach qui occupe l'emplacement maintenant
            on = c in self.tracked
            col = self.colors[c] if on else _tint(self.colors[c], -0.25)
            pygame.draw.line(surf, col, (self.bars[j], r.y + hud.px(5)), (self.bars[j], r.bottom - hud.px(10)),
                             hud.px(3) if on else hud.px(2))
        base = r.bottom - hud.px(6)
        pygame.draw.line(surf, (90, 90, 90), (self.bar_x, base), (r.right, base), max(1, hud.px(1)))
        a0, a1 = np.searchsorted(self.kf_t, [t - HORIZON_PAST, t + self.horizon])
        for i, ti in zip(self.kf[a0:a1], self.kf_t[a0:a1]):
            pic = self._pic(i)
            if pic is None:
                continue
            dt = ti - t
            cx = self.bar_x + dt * self.speed
            x = int(cx - self.gw / 2)
            a = 1.0 if dt >= 0 else max(0.0, 1 + dt / HORIZON_PAST) * 0.5
            self.card.set_alpha(int(255 * a))
            surf.blit(self.card, (x - 4, self.ty - 4))
            img, glow = pic
            if abs(dt) < 0.15:                                       # « allumé » en passant la barre
                glow.set_alpha(int(150 * (1 - abs(dt) / 0.15)))
                surf.blit(glow, (x, self.ty))
            img.set_alpha(int(255 * a))
            surf.blit(img, (x, self.ty))
            pygame.draw.line(surf, (160, 160, 160), (int(cx), base - hud.px(4)), (int(cx), base), max(1, hud.px(1)))
        surf.set_clip(clip)
