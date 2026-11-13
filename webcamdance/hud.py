"""Rendu HUD pygame : panneaux joueurs, jugements animés, PiP webcam, résultats."""
import math

import cv2
import numpy as np
import pygame

from .skeleton import EDGES

COLORS = [(0, 229, 255), (255, 64, 160), (255, 214, 0), (118, 255, 3)]
JUDGE_COL = {"PERFECT": (255, 215, 64), "SUPER": (118, 255, 3), "GOOD": (0, 229, 255),
             "OK": (200, 200, 210), "MISS": (255, 70, 70)}
_fonts = {}
U = 1.0     # échelle d'interface = hauteur écran / 720 ; tailles du code en « pixels 720p »


def set_scale(height):
    """À appeler à chaque (re)création de la fenêtre : rendu net à la résolution native."""
    global U
    U = height / 720
    _fonts.clear()
    _tags.clear()


def px(v):
    return max(1, int(round(v * U))) if v > 0 else int(round(v * U))


def font(size, bold=True):
    if (size, bold) not in _fonts:
        _fonts[(size, bold)] = pygame.font.SysFont("segoeui,arial", px(size), bold=bold)
    return _fonts[(size, bold)]


def text(surf, s, size, color, pos, anchor="topleft", bold=True, shadow=True):
    img = font(size, bold).render(s, True, color)
    r = img.get_rect(**{anchor: pos})
    if shadow:
        surf.blit(font(size, bold).render(s, True, (0, 0, 0)), r.move(px(2), px(2)))
    surf.blit(img, r)
    return r


_tags = {}


def tag_surface(s, size, color):
    """Étiquette lisible sur n'importe quel fond : lettres couleur joueur, contour noir,
    halo blanc flouté léger. Mise en cache (le flou coûte ~1 ms)."""
    key = (s, size, color)
    if key in _tags:
        return _tags[key]
    f = font(size)
    fill, black, white = f.render(s, True, color), f.render(s, True, (0, 0, 0)), f.render(s, True, (255, 255, 255))
    pad, o = px(max(6, size // 3)), px(max(2, size // 12))
    w, h = fill.get_size()
    out = pygame.Surface((w + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
    halo = pygame.Surface(out.get_size(), pygame.SRCALPHA)
    for dx in range(-o * 2, o * 2 + 1, o):
        for dy in range(-o * 2, o * 2 + 1, o):
            halo.blit(white, (pad + dx, pad + dy))
    halo = pygame.transform.box_blur(halo, max(2, pad // 2))
    halo.fill((255, 255, 255, 130), special_flags=pygame.BLEND_RGBA_MULT)
    out.blit(halo, (0, 0))
    for dx in (-o, 0, o):
        for dy in (-o, 0, o):
            if dx or dy:
                out.blit(black, (pad + dx, pad + dy))
    out.blit(fill, (pad, pad))
    _tags[key] = out
    return out


def tag(surf, s, size, color, pos, anchor="midbottom", alpha=255, clip=None):
    img = tag_surface(s, size, color)
    if alpha < 255:
        img = img.copy()
        img.set_alpha(alpha)
    r = img.get_rect(**{anchor: pos})
    if clip is not None:
        r.clamp_ip(clip)
    surf.blit(img, r)
    return r


def panel(surf, rect, alpha=150, radius=14):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (12, 12, 24, alpha), s.get_rect(), border_radius=px(radius))
    surf.blit(s, rect)


def star(surf, cx, cy, r, color):
    """r en pixels écran (déjà mis à l'échelle par l'appelant)."""
    pts = [(cx + (r if i % 2 == 0 else r * 0.45) * math.sin(i * math.pi / 5),
            cy - (r if i % 2 == 0 else r * 0.45) * math.cos(i * math.pi / 5)) for i in range(10)]
    pygame.draw.polygon(surf, color, pts)


def bgr_surface(frame):
    h, w = frame.shape[:2]
    return pygame.image.frombuffer(frame.tobytes(), (w, h), "BGR")


def skeleton(surf, kp, color, thr, scale, off, width=3):
    # float() : pygame-ce refuse les numpy.float32 comme coordonnées
    pts = [(off[0] + float(x) * scale, off[1] + float(y) * scale) for x, y, _ in kp]
    for a, b in EDGES:
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            pygame.draw.line(surf, color, pts[a], pts[b], width)
    for i in range(5, 17):
        if kp[i, 2] >= thr:
            pygame.draw.circle(surf, (255, 255, 255), pts[i], width)


def webcam(surf, frame, poses, rect, thr):
    """Vignette webcam (déjà miroir) + squelettes colorés par slot."""
    if frame is None:
        return
    surf.blit(bgr_surface(cv2.resize(frame, rect.size)), rect)
    s = rect.w / frame.shape[1]
    for i, kp in enumerate(poses):
        if kp is not None:
            skeleton(surf, kp, COLORS[i % 4], thr, s, rect.topleft, max(2, rect.w // 200))
            x, y = rect.x + float(kp[5:7, 0].mean()) * s, rect.y + float(kp[5:7, 1].min()) * s
            tag(surf, f"P{i + 1}", max(18, rect.w // 20), COLORS[i % 4], (x, y - rect.h * 0.12), "midbottom")
    pygame.draw.rect(surf, (255, 255, 255), rect, px(2), border_radius=px(6))


def progress(surf, frac):
    W = surf.get_width()
    pygame.draw.rect(surf, (40, 40, 60), (0, 0, W, px(6)))
    pygame.draw.rect(surf, (255, 255, 255), (0, 0, int(W * frac), px(6)))


def results(surf, scores, records=()):
    W, H = surf.get_size()
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 190))
    surf.blit(dim, (0, 0))
    text(surf, "RÉSULTATS", 56, (255, 255, 255), (W // 2, px(60)), "midtop")
    best = max(s.total for s in scores)
    n = len(scores)
    for i, sc in enumerate(scores):
        cx, col = int((i + 0.5) * W / n), COLORS[i % 4]
        text(surf, f"P{i + 1}" + ("  ♛" if sc.total == best and n > 1 else ""), 40, col, (cx, px(190)), "midtop")
        text(surf, f"{sc.total:,}".replace(",", " "), 64, (255, 255, 255), (cx, px(240)), "midtop")
        if i < len(records) and records[i]:
            tag(surf, "Nouveau record !", 24, (255, 215, 64), (cx, px(312)), "midtop")
        for k in range(5):
            star(surf, cx + (k - 2) * px(44), px(350), px(18), (255, 215, 64) if sc.stars > k else (90, 90, 110))
        y = px(400)
        for label, cnt in sc.counts.items():
            text(surf, f"{label}  {cnt}", 22, JUDGE_COL[label], (cx, y), "midtop", bold=False)
            y += px(30)
        text(surf, f"meilleur combo x{sc.best_combo}", 20, (220, 220, 230), (cx, y + px(8)), "midtop", bold=False)
    text(surf, "Entrée : retour", 20, (200, 200, 210), (W // 2, H - px(40)), "midbottom", bold=False)


# ---------------------------------------------------------------- disposition empilée
NEUTRAL = (150, 150, 165)


def coach_colors(n_coach, cmap):
    """Couleur et libellé par coach d'après cmap (joueur p -> coach cmap[p]) ; gris si personne."""
    out = []
    for c in range(n_coach):
        ps = [p for p, cc in enumerate(cmap) if cc == c]
        out.append((COLORS[ps[0] % 4], " ".join(f"P{p + 1}" for p in ps)) if ps else (NEUTRAL, ""))
    return out


def coach_overlay(surf, coach, idx, rect, scale, y0, thr, colors, skel=True, alpha=70):
    """Squelettes très légers sur les coachs + étiquette joueur au-dessus de la tête.
    Coach absent : étiquette « flottante » semi-transparente là où il va (ré)apparaître,
    avec le compte à rebours, sinon à sa dernière position connue."""
    kpts = coach.kpts[idx]
    to_scr = lambda x, y: (rect.x + float(x) * scale, rect.y + (float(y) - y0) * scale)
    if skel:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        for c, kp in enumerate(kpts):
            ok = kp[:, 2] >= thr
            if ok.sum() < 5:
                continue
            pts = [(float(x) * scale, (float(y) - y0) * scale) for x, y, _ in kp]
            for a, b in EDGES:
                if ok[a] and ok[b]:
                    pygame.draw.line(layer, (*colors[c][0], alpha), pts[a], pts[b], px(3))
        surf.blit(layer, rect)
    for c, (col, label) in enumerate(colors):
        if not label:
            continue
        if np.isfinite(coach.head[idx, c]).all():
            x, y = to_scr(*coach.head[idx, c])
            tag(surf, label, 30, col, (x, y - px(4)), "midbottom", clip=rect)
            continue
        nxt, last = coach.next_idx[idx, c], coach.last_idx[idx, c]
        ref = nxt if nxt >= 0 else last
        if ref < 0:
            continue
        x, y = to_scr(*coach.head[ref, c])
        r = tag(surf, label, 30, col, (x, y - px(4)), "midbottom", 170, clip=rect.inflate(0, -px(40)).move(0, px(20)))
        if nxt >= 0:
            secs = (nxt - idx) / coach.fps
            if secs <= 10:
                tag(surf, f"arrive dans {secs:.0f} s" if secs >= 1 else "arrive !", 20, col,
                    (r.centerx, r.bottom - px(10)), "midtop", 200, clip=rect)


def coach_thumbs_row(surf, thumbs, fx, colors, area):
    """Formation de départ : vignette de chaque coach à sa place (x normalisé), cadre et
    libellé de la couleur du joueur qui le suit."""
    n = len(thumbs)
    order = np.argsort(fx)
    tw = max(t.get_width() for t in thumbs) if thumbs else 0
    # positions proportionnelles à la formation, espacement minimal = largeur vignette
    xs = np.array([area.x + area.w * fx[c] for c in range(n)])
    for k in range(1, n):
        a, b = order[k - 1], order[k]
        xs[b] = max(xs[b], xs[a] + tw + px(12))
    xs += area.centerx - (xs.min() + xs.max()) / 2
    for c in range(n):
        col, label = colors[c]
        r = thumbs[c].get_rect(midtop=(int(xs[c]), area.y))
        surf.blit(thumbs[c], r)
        pygame.draw.rect(surf, col, r.inflate(px(8), px(8)), px(4) if label else 1, border_radius=px(8))
        if label:
            tag(surf, label, 30, col, (r.centerx, r.bottom - px(4)), "midtop")


def judge_pop(surf, label, age, pos, size=40, clip=None):
    """Jugement animé : pop 1.6 -> 1, montée, fondu ; gerbe d'étoiles sur PERFECT."""
    if not 0 <= age < 0.9:
        return
    col = JUDGE_COL[label]
    img = font(size).render(label, True, col)
    img = pygame.transform.rotozoom(img, 0, 1.6 - 0.6 * min(age / 0.12, 1.0))
    img.set_alpha(int(255 * (1 - max(0.0, (age - 0.55) / 0.35))))
    x, y = pos[0], pos[1] - px(18) * age
    r = img.get_rect(midbottom=(x, y))
    if clip is not None:
        r.clamp_ip(clip)
    surf.blit(img, r)
    if label == "PERFECT" and age < 0.6:
        for k in range(10):
            ang = k * math.pi / 5 + 0.3
            d = px(20 + 160 * age)
            star(surf, r.centerx + d * math.cos(ang), r.centery + 0.6 * d * math.sin(ang),
                 max(2, px(8 * (1 - age / 0.6))), col)


def star_gauge(surf, g, col, sc):
    """Jauge fine + 5 petites étoiles (seuils tous les 16 %)."""
    pygame.draw.rect(surf, (60, 60, 80), g, border_radius=g.h // 2)
    pygame.draw.rect(surf, col, (g.x, g.y, int(g.w * min(1.0, sc.ratio / 0.8)), g.h), border_radius=g.h // 2)
    for k in range(1, 6):
        star(surf, g.x + g.w * k / 5 - px(2), g.centery, px(7), (255, 215, 64) if sc.stars >= k else (110, 110, 130))


_crops = {}     # joueur -> [cx, cy, h] cadrage lissé dans la webcam (px source)


def reset_crops():
    _crops.clear()


def _crop_box(i, kp, thr, fh, aspect=0.75, margin=1.3, alpha=0.25):
    """Cadrage portrait lissé autour du squelette du joueur i. -> ([cx, cy, h] ou None, vu ?)."""
    st = _crops.get(i)
    if kp is not None:
        ok = kp[:, 2] >= thr
        if ok.sum() >= 5:
            (x0, y0), (x1, y1) = kp[ok, :2].min(0), kp[ok, :2].max(0)
            h = max((y1 - y0) * margin, (x1 - x0) * margin / aspect, 0.3 * fh)
            tgt = ((x0 + x1) / 2, (y0 + y1) / 2 - 0.03 * h, h)       # un peu d'air au-dessus de la tête
            st = list(tgt) if st is None else [a + alpha * (b - a) for a, b in zip(st, tgt)]
            _crops[i] = st
            return st, True
    return st, False


def _extract(frame, box, size, aspect=0.75):
    """Recadre frame sur box (complète en sombre hors image) et redimensionne.
    -> (image, (x0, y0, échelle)) pour convertir les keypoints."""
    cx, cy, h = box
    w = h * aspect
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x1, y1 = x0 + max(2, int(round(w))), y0 + max(2, int(round(h)))
    fh, fw = frame.shape[:2]
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        sub = np.full((y1 - y0, x1 - x0, 3), (22, 10, 10), np.uint8)
    else:
        sub = cv2.copyMakeBorder(frame[iy0:iy1, ix0:ix1], iy0 - y0, y1 - iy1, ix0 - x0, x1 - ix1,
                                 cv2.BORDER_CONSTANT, value=(22, 10, 10))
    interp = cv2.INTER_AREA if size[0] < sub.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(sub, size, interpolation=interp), (x0, y0, size[0] / (x1 - x0))


def _halo(surf, rect, pts, ok, col):
    """Halo lumineux : membres épais à mi-résolution, flou boîte, blit additif."""
    half = pygame.Surface((max(1, rect.w // 2), max(1, rect.h // 2)), pygame.SRCALPHA)
    w = max(2, rect.h // 70)                           # léger : le joueur doit rester bien visible
    for a, b in EDGES:
        if ok[a] and ok[b]:
            pygame.draw.line(half, (*col, 55), ((pts[a][0] - rect.x) / 2, (pts[a][1] - rect.y) / 2),
                             ((pts[b][0] - rect.x) / 2, (pts[b][1] - rect.y) / 2), w)
    half = pygame.transform.box_blur(half, max(2, w))
    surf.blit(pygame.transform.smoothscale(half, rect.size), rect, special_flags=pygame.BLEND_RGB_ADD)


def player_strip(surf, rect, frame, poses, scores, t, thr, waiting=None):
    """Bande du bas : une mini-fenêtre par joueur, recadrée sur lui (halo + squelette),
    jugement au-dessus, score / combo / étoiles en dessous."""
    pygame.draw.rect(surf, (10, 10, 22), rect)
    n = len(scores)
    if n == 0:
        return
    score_h, top = px(52), px(8)
    vh = rect.h - score_h - top - px(4)
    vw = int(vh * 0.75)
    col_w = rect.w / n
    if vw > col_w - px(24):
        vw = int(col_w - px(24))
        vh = int(vw / 0.75)
    for i, sc in enumerate(scores):
        col = COLORS[i % 4]
        vr = pygame.Rect(int(rect.x + (i + 0.5) * col_w - vw / 2), rect.y + top, vw, vh)
        kp = poses[i] if i < len(poses) else None
        box, live = (None, False) if frame is None else _crop_box(i, kp, thr, frame.shape[0])
        if box is not None:
            img, (ox, oy, s) = _extract(frame, box, vr.size)
            if not live:
                img = (img * 0.35).astype(np.uint8)
            surf.blit(bgr_surface(img), vr)
            if live:
                pts = [(vr.x + (float(x) - ox) * s, vr.y + (float(y) - oy) * s) for x, y, _ in kp]
                ok = kp[:, 2] >= thr
                _halo(surf, vr, pts, ok, col)
                for a, b in EDGES:
                    if ok[a] and ok[b]:
                        pygame.draw.line(surf, col, pts[a], pts[b], px(2))
        else:
            pygame.draw.rect(surf, (24, 24, 40), vr, border_radius=px(8))
        pygame.draw.rect(surf, col, vr, px(3), border_radius=px(8))
        if not live:
            tag(surf, "?", 40, col, vr.center, "center")
        # combo / attente : en surimpression en bas de la mini-fenêtre
        if waiting and waiting[i]:
            tag(surf, "en attente du coach…", 15, (220, 220, 235), (vr.centerx, vr.bottom - px(4)), "midbottom")
        elif sc.combo >= 3:
            tag(surf, f"combo x{sc.combo}", 18, (255, 215, 64), (vr.centerx, vr.bottom - px(2)), "midbottom")
        # ligne de score + jauge d'étoiles
        y = vr.bottom + px(2)
        tag(surf, f"P{i + 1}", 24, col, (vr.x - px(8), y - px(6)), "topleft")
        text(surf, f"{sc.total:,}".replace(",", " "), 30, (255, 255, 255), (vr.right, y), "topright")
        star_gauge(surf, pygame.Rect(vr.x, rect.bottom - px(9), vr.w, px(6)), col, sc)
        # jugement juste au-dessus de la mini-fenêtre
        if sc.last and 0 <= t - sc.last[1] < 0.9:
            judge_pop(surf, sc.last[0], t - sc.last[1], (vr.centerx, vr.y + px(6)), 34, surf.get_rect())
