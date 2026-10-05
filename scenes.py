"""Procedural pixel-art scenes for the Flora0world HUB cinematic preview.

Every scene is a function f(local_t) -> float32 array (H, W, 3), 0..255,
drawn at a low internal resolution (480x270) and upscaled with nearest
neighbour at the end for the chunky pixel look.
"""
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 480, 270
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_S = ImageFont.truetype(FONT_PATH, 9)
FONT_M = ImageFont.truetype(FONT_PATH, 11)


# ----------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def lerpc(c1, c2, t):
    return tuple(lerp(c1[i], c2[i], t) for i in range(3))


def vgrad(w, h, stops):
    ys = np.linspace(0, 1, h)
    pos = [s[0] for s in stops]
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(ys, pos, [s[1][c] for s in stops])[:, None]
    return out


def ramp(t, stops):
    pos = [s[0] for s in stops]
    out = np.empty(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, [s[1][c] for s in stops])
    return out


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def to_arr(i):
    return np.asarray(i, dtype=np.float32).copy()


def glow(a, cx, cy, r, col, k=1.0):
    x0, x1 = int(max(0, cx - r)), int(min(a.shape[1], cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(a.shape[0], cy + r + 1))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    f = np.clip(1 - d, 0, 1) ** 2 * k
    a[y0:y1, x0:x1] += f[..., None] * np.array(col, np.float32)


def px(a, x, y, col, k=1.0):
    x, y = int(x), int(y)
    if 0 <= x < a.shape[1] and 0 <= y < a.shape[0]:
        a[y, x] = a[y, x] * (1 - k) + np.array(col, np.float32) * k


def zoom_crop(a, z, cx, cy):
    if z <= 1.001:
        return a
    vw = int(round(W / z))
    vh = int(round(vw * H / W))
    x0 = int(clamp(cx - vw / 2, 0, W - vw))
    y0 = int(clamp(cy - vh / 2, 0, H - vh))
    img = to_img(a).crop((x0, y0, x0 + vw, y0 + vh)).resize((W, H), Image.NEAREST)
    return to_arr(img)


def vnoise(w, h, seed, octs=5, base=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octs):
        n = base * 2 ** o
        g = rng.random((n + 1, int(n * w / h) + 2))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(im, np.float32) / 255 * amp
        tot += amp
        amp *= 0.5
    return out / tot


def ridge(xs, base, amp, freq, phase, shift):
    return (base + amp * np.sin(xs * freq + phase)
            + amp * 0.5 * np.sin(xs * freq * 2.3 + phase * 1.7 + shift)
            + amp * 0.25 * np.sin(xs * freq * 5.1 + shift * 2))


# ----------------------------------------------------------------- sprites
BUTTERFLIES = [
    ((255, 140, 30), (70, 30, 20)),
    ((70, 150, 255), (20, 55, 160)),
    ((250, 230, 60), (130, 85, 20)),
    ((235, 75, 115), (95, 20, 55)),
    ((110, 225, 125), (20, 95, 55)),
    ((240, 240, 250), (120, 130, 165)),
]


def butterfly(d, x, y, ph, c1, c2, s=1.0):
    f = abs(math.cos(ph)) * 0.85 + 0.15
    wx = 9 * s * f
    wy = 7 * s
    for sg in (-1, 1):
        d.polygon([(x, y), (x + sg * wx, y - wy), (x + sg * wx * 1.1, y - wy * 0.2), (x, y + 1)], fill=c1)
        d.polygon([(x, y + 1), (x + sg * wx * 0.8, y + wy * 0.9), (x + sg * wx * 0.3, y + wy * 1.1), (x, y + 3 * s)], fill=c2)
        d.point((x + sg * wx * 0.7, y - wy * 0.55), fill=(255, 255, 255))
    d.line([(x, y - 2 * s), (x, y + 4 * s)], fill=(25, 18, 18))
    d.point((x - 1, y - 3 * s), fill=(25, 18, 18))
    d.point((x + 1, y - 3 * s), fill=(25, 18, 18))


def tiny_moth(d, x, y, ph, col, s=1.0):
    f = abs(math.sin(ph)) * 0.8 + 0.2
    col = tuple(int(c) for c in col)
    for sg in (-1, 1):
        d.polygon([(x, y), (x + sg * (4 * s * f + 1), y - 2 * s), (x + sg * 3 * s * f, y + 2.5 * s)], fill=col)
    d.point((x, y), fill=(60, 45, 35))


def hero_moth(d, x, y, ph, s):
    f = 0.6 + 0.4 * abs(math.cos(ph))
    for sg in (-1, 1):
        fw = [(x, y - 2 * s), (x + sg * 22 * s * f, y - 14 * s), (x + sg * 31 * s * f, y - 6 * s),
              (x + sg * 26 * s * f, y + 6 * s), (x + sg * 6 * s * f, y + 4 * s)]
        d.polygon(fw, fill=(165, 92, 48))
        hw = [(x, y + 2 * s), (x + sg * 24 * s * f, y + 6 * s), (x + sg * 21 * s * f, y + 19 * s), (x + sg * 8 * s * f, y + 17 * s)]
        d.polygon(hw, fill=(135, 70, 40))
        d.line([fw[1], fw[3]], fill=(70, 35, 20), width=max(1, int(s * 0.7)))
        d.line([fw[0], fw[2]], fill=(210, 150, 90))
        ex, ey = x + sg * 19 * s * f, y - 3 * s
        d.ellipse([ex - 3.5 * s, ey - 3.5 * s, ex + 3.5 * s, ey + 3.5 * s], fill=(250, 238, 200))
        d.ellipse([ex - 2 * s, ey - 2 * s, ex + 2 * s, ey + 2 * s], fill=(35, 18, 12))
        for k in range(4):
            d.line([(x + sg * 1.5 * s, y - 8 * s), (x + sg * (4 + k * 2) * s, y - 15 * s - k * s)], fill=(60, 40, 25))
    d.ellipse([x - 2.3 * s, y - 8 * s, x + 2.3 * s, y + 14 * s], fill=(100, 62, 38))
    d.ellipse([x - 1.5 * s, y - 11 * s, x + 1.5 * s, y - 7 * s], fill=(80, 50, 30))


def person(d, x, y, phase, theta, walking, col=(24, 20, 30), net_len=36):
    bob = abs(math.sin(phase)) * 1.5 if walking else 0
    ly = y - bob
    s = math.sin(phase) * 4 if walking else 0
    d.line([(x - 1, ly - 14), (x - 1 + s, ly)], fill=col, width=3)
    d.line([(x + 1, ly - 14), (x + 1 - s, ly)], fill=col, width=3)
    d.rectangle([x - 4, ly - 30, x + 3, ly - 13], fill=col)
    d.rectangle([x - 9, ly - 29, x - 4, ly - 17], fill=(62, 46, 34))
    d.rectangle([x - 9, ly - 31, x - 4, ly - 29], fill=(120, 40, 40))
    d.ellipse([x - 3, ly - 37, x + 3, ly - 31], fill=(150, 105, 88))
    d.rectangle([x - 6, ly - 38, x + 6, ly - 37], fill=(48, 36, 44))
    d.rectangle([x - 3, ly - 42, x + 3, ly - 38], fill=(48, 36, 44))
    hx, hy = x + 5, ly - 25
    d.line([(x + 2, ly - 28), (hx, hy)], fill=col, width=2)
    cx_, cy_ = math.cos(theta), math.sin(theta)
    d.line([(hx, hy), (hx + cx_ * (net_len - 6), hy + cy_ * (net_len - 6))], fill=(130, 92, 52), width=1)
    hc = (hx + cx_ * net_len, hy + cy_ * net_len)
    # hoop as a polygon rotated with the pole
    hoop = []
    for k in range(14):
        a = k / 14 * 2 * math.pi
        u, v = math.cos(a) * 3.0, math.sin(a) * 8.0
        hoop.append((hc[0] + u * cx_ - v * cy_, hc[1] + u * cy_ + v * cx_))
    tip = (hc[0] - cx_ * 6, hc[1] + 17)
    for hp in hoop[::2]:
        d.line([hp, tip], fill=(185, 185, 175), width=1)
    d.polygon(hoop, outline=(235, 235, 225))
    return hc


# ================================================================ CABINET
BW, BH = 560, 315


def _build_cabinet():
    a = np.zeros((BH, BW, 3), np.float32)
    a[:] = (62, 40, 32)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for x in range(0, BW, 28):
        d.line([(x, 0), (x, 215)], fill=(50, 32, 25))
    d.rectangle([0, 0, BW, 10], fill=(40, 25, 20))
    d.rectangle([0, 200, BW, 216], fill=(80, 52, 37))
    d.rectangle([0, 216, BW, BH], fill=(92, 60, 40))
    rng = random.Random(3)
    for y in range(222, BH, 9):
        d.line([(0, y), (BW, y)], fill=(70, 45, 30))
        for _ in range(6):
            x = rng.randrange(BW)
            d.line([(x, y), (x, y + 9)], fill=(70, 45, 30))
    d.ellipse([120, 250, 420, 312], fill=(110, 40, 40))
    d.ellipse([135, 258, 405, 304], outline=(190, 150, 90))
    d.ellipse([160, 266, 380, 296], fill=(88, 30, 36))
    a = to_arr(img)
    a[34:168, 336:444] = vgrad(108, 134, [(0, (70, 50, 120)), (0.45, (232, 112, 92)), (0.75, (255, 170, 80)), (1, (255, 212, 125))])
    glow(a, 400, 150, 60, (255, 230, 150), 0.5)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    xs = np.arange(336, 444)
    hill = ridge(xs, 150, 6, 0.05, 1.0, 2.0)
    d.polygon([(336, 168)] + [(int(x), int(y)) for x, y in zip(xs, hill)] + [(443, 168)], fill=(38, 26, 48))
    for tx in (350, 368, 424, 436):
        d.polygon([(tx, 148), (tx - 7, 168), (tx + 7, 168)], fill=(26, 20, 38))
        d.polygon([(tx, 136), (tx - 5, 154), (tx + 5, 154)], fill=(26, 20, 38))
    d.rectangle([330, 28, 450, 174], outline=(28, 17, 12), width=7)
    d.rectangle([388, 30, 392, 172], fill=(28, 17, 12))
    d.rectangle([332, 98, 448, 102], fill=(28, 17, 12))
    d.rectangle([322, 172, 458, 183], fill=(112, 76, 50))
    d.polygon([(326, 24), (352, 24), (344, 176), (326, 176)], fill=(110, 28, 38))
    d.polygon([(436, 24), (462, 24), (462, 176), (444, 176)], fill=(110, 28, 38))
    for fx in (332, 338, 344):
        d.line([(fx, 28), (fx - 1, 174)], fill=(80, 18, 28))
    for fx in (448, 454, 458):
        d.line([(fx, 28), (fx + 1, 174)], fill=(80, 18, 28))
    d.rectangle([318, 18, 466, 26], fill=(52, 34, 24))
    # drawer cabinet
    d.rectangle([22, 36, 132, 214], fill=(76, 50, 36), outline=(30, 18, 12))
    d.rectangle([18, 30, 136, 38], fill=(60, 40, 28), outline=(30, 18, 12))
    for r in range(7):
        for c in range(4):
            x0, y0 = 26 + c * 26, 42 + r * 24
            d.rectangle([x0, y0, x0 + 24, y0 + 22], fill=(88, 58, 40), outline=(40, 24, 16))
            d.rectangle([x0 + 6, y0 + 4, x0 + 18, y0 + 8], fill=(210, 190, 140))
            d.rectangle([x0 + 10, y0 + 13, x0 + 14, y0 + 16], fill=(190, 150, 70))
    # framed butterflies
    frames = [(150, 44, 44, 36), (202, 44, 44, 36), (254, 44, 44, 36),
              (156, 92, 34, 28), (198, 92, 34, 28), (240, 92, 34, 28), (282, 92, 34, 28)]
    for i, (fx, fy, fw, fh) in enumerate(frames):
        d.rectangle([fx, fy, fx + fw, fy + fh], fill=(34, 20, 12))
        d.rectangle([fx + 3, fy + 3, fx + fw - 3, fy + fh - 3], fill=(228, 214, 174))
        c1, c2 = BUTTERFLIES[i % len(BUTTERFLIES)]
        if i == 4:
            d.ellipse([fx + 11, fy + 8, fx + 23, fy + 20], fill=(30, 90, 60))
            d.line([(fx + 17, fy + 6), (fx + 17, fy + 22)], fill=(20, 60, 40))
        else:
            butterfly(d, fx + fw / 2, fy + fh / 2 - 1, 0, c1, c2, 0.55 + 0.35 * (fw > 40))
        d.line([(fx + 4, fy + 4), (fx + fw - 6, fy + 10)], fill=(245, 238, 210))
    # desk
    d.rectangle([146, 172, 330, 182], fill=(120, 78, 46))
    d.rectangle([146, 182, 330, 186], fill=(70, 44, 26))
    d.rectangle([152, 186, 324, 224], fill=(86, 54, 32))
    for x in (212, 268):
        d.line([(x, 188), (x, 222)], fill=(56, 34, 20))
    for x in (182, 240, 296):
        d.rectangle([x - 5, 200, x + 5, 204], fill=(190, 150, 70))
    # specimen box on desk
    d.rectangle([158, 158, 212, 172], fill=(150, 110, 70), outline=(60, 38, 22))
    d.rectangle([161, 161, 209, 169], fill=(230, 220, 190))
    for k, col in enumerate(((255, 140, 30), (70, 150, 255), (110, 225, 125), (235, 75, 115), (250, 230, 60), (240, 240, 250))):
        d.rectangle([164 + k * 7, 163 + (k % 2) * 2, 167 + k * 7, 166 + (k % 2) * 2], fill=col)
    # jar
    d.rectangle([222, 154, 238, 172], fill=(120, 150, 140), outline=(200, 225, 215))
    d.rectangle([221, 150, 239, 154], fill=(150, 110, 60))
    d.rectangle([226, 163, 234, 170], fill=(60, 110, 70))
    # microscope
    d.rectangle([244, 167, 262, 172], fill=(30, 30, 36))
    d.line([(250, 167), (250, 148)], fill=(40, 40, 48), width=3)
    d.line([(250, 150), (262, 156)], fill=(60, 60, 70), width=3)
    d.rectangle([258, 150, 264, 164], fill=(70, 70, 80))
    # books
    d.rectangle([266, 165, 284, 172], fill=(60, 80, 120))
    d.rectangle([268, 159, 286, 165], fill=(130, 50, 50))
    # lamp
    d.ellipse([292, 168, 310, 174], fill=(30, 40, 36))
    d.line([(300, 170), (294, 146)], fill=(50, 55, 50), width=2)
    d.line([(294, 146), (308, 136)], fill=(50, 55, 50), width=2)
    d.polygon([(302, 126), (322, 134), (316, 148), (298, 140)], fill=(30, 100, 74))
    d.ellipse([306, 140, 312, 146], fill=(255, 235, 170))
    # nets leaning on the right wall
    for (x0, y0, x1, y1, hx, hy) in ((520, 300, 466, 100, 466, 82), (500, 300, 522, 120, 522, 100)):
        d.line([(x0, y0), (x1, y1)], fill=(150, 108, 60), width=3)
        d.ellipse([hx - 19, hy - 14, hx + 19, hy + 14], outline=(185, 185, 180), width=2)
        for k in range(7):
            ax = hx - 17 + k * 5.6
            d.line([(ax, hy + 6), (hx + (ax - hx) * 0.2, hy + 44)], fill=(120, 116, 104))
        for k in range(1, 5):
            yy = hy + 6 + k * 8
            d.line([(hx - 16 + k * 3, yy), (hx + 16 - k * 3, yy)], fill=(120, 116, 104))
    # boxes
    d.rectangle([436, 198, 514, 236], fill=(160, 124, 80), outline=(80, 58, 36))
    d.rectangle([444, 168, 506, 198], fill=(140, 105, 66), outline=(80, 58, 36))
    d.rectangle([452, 142, 500, 168], fill=(172, 136, 90), outline=(80, 58, 36))
    for (bx, by, bw) in ((436, 198, 78), (444, 168, 62), (452, 142, 48)):
        d.rectangle([bx + bw // 2 - 3, by, bx + bw // 2 + 3, by + 6], fill=(200, 175, 120))
    d.rectangle([446, 212, 470, 224], fill=(235, 225, 195))
    d.rectangle([454, 178, 474, 188], fill=(235, 225, 195))
    d.line([(449, 216), (466, 216)], fill=(80, 40, 40))
    d.rectangle([0, 238, 72, 292], fill=(150, 116, 74), outline=(80, 58, 36))
    d.rectangle([14, 208, 66, 238], fill=(134, 100, 64), outline=(80, 58, 36))
    d.rectangle([6, 250, 30, 262], fill=(235, 225, 195))
    d.rectangle([32, 238, 40, 246], fill=(200, 175, 120))
    m = Image.new("L", (BW, BH), 0)
    md = ImageDraw.Draw(m)
    md.polygon([(338, 38), (392, 38), (150, 300), (40, 300)], fill=255)
    md.polygon([(396, 38), (444, 38), (300, 300), (190, 300)], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(7))
    shaft = np.asarray(m, np.float32) / 255
    return np.asarray(img, np.uint8), shaft


CAB, SHAFT = _build_cabinet()
_rngd = np.random.default_rng(1)
DUST = _rngd.random((110, 4))


def cabinet(t):
    a = CAB.astype(np.float32)
    k = 0.92 + 0.08 * math.sin(t * 0.7)
    a += SHAFT[..., None] * np.array((255, 170, 90), np.float32) * 0.24 * k
    fl = 1 + 0.04 * math.sin(t * 23) + 0.03 * math.sin(t * 7.1)
    glow(a, 308, 148, 100, (255, 185, 95), 0.6 * fl)
    glow(a, 400, 150, 130, (255, 150, 60), 0.3)
    for u in DUST:
        x = 40 + (u[0] * 300 - t * (2 + 4 * u[2])) % 300 + 5 * math.sin(t * 0.5 + u[3] * 6)
        y = 60 + (u[1] * 190 + t * (1 + 3 * u[2])) % 190
        s = int(clamp(y, 0, BH - 1)) , int(clamp(x, 0, BW - 1))
        lit = 0.25 + SHAFT[s] * 0.9
        br = (0.35 + 0.65 * math.sin(t * 1.3 + u[3] * 9) ** 2) * lit
        px(a, x, y, (255, 225, 170), clamp(br))
        if u[2] > 0.7:
            px(a, x + 1, y, (255, 225, 170), clamp(br * 0.6))
    img = to_img(a)
    d = ImageDraw.Draw(img)
    # moth circling the lamp
    mx = 309 + 24 * math.cos(1.9 * t) + 6 * math.sin(5 * t)
    my = 142 + 12 * math.sin(2.6 * t)
    tiny_moth(d, mx, my, t * 40, (235, 210, 170), 1.1)
    # living morpho drifting across the room
    bx = 250 + 175 * math.sin(0.23 * t + 1.0)
    by = 118 + 46 * math.sin(0.41 * t) + 8 * math.sin(2.7 * t)
    butterfly(d, bx, by, t * 13, (70, 150, 255), (20, 55, 160), 1.2)
    a = to_arr(img)
    glow(a, bx, by, 14, (90, 170, 255), 0.35)
    img = to_img(a)
    # camera: slow drift, then a deep push into the window
    e = smooth((t - 41) / 5)
    ox = 20 + 30 * t / 46
    oy = 18 - 8 * t / 46
    cx = lerp(ox + 240, 390, e)
    cy = lerp(oy + 135, 100, e)
    z = 1 + e * 3.0
    vw = int(round(W / z))
    vh = int(round(vw * H / W))
    x0 = int(clamp(cx - vw / 2, 0, BW - vw))
    y0 = int(clamp(cy - vh / 2, 0, BH - vh))
    return to_arr(img.crop((x0, y0, x0 + vw, y0 + vh)).resize((W, H), Image.NEAREST))


# ================================================================ NIGHT
def _build_night():
    a = vgrad(W, H, [(0, (4, 8, 28)), (0.5, (12, 36, 52)), (1, (8, 26, 30))])
    img = to_img(a)
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    for _ in range(140):
        x, y = rng.randrange(W), rng.randrange(0, 120)
        b = rng.randrange(120, 255)
        d.point((x, y), fill=(b, b, min(255, b + 20)))
    d.ellipse([74, 40, 106, 72], fill=(236, 240, 226))
    for cx, cy, r in ((84, 50, 4), (97, 60, 3), (88, 64, 2)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(205, 212, 200))
    a = to_arr(img)
    glow(a, 90, 56, 90, (120, 150, 200), 0.55)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for layer, (col, wmin, wmax, n) in enumerate([((12, 36, 46), 5, 10, 16), ((6, 20, 26), 10, 20, 9)]):
        for _ in range(n):
            x = rng.randrange(-10, W + 10)
            w = rng.randrange(wmin, wmax)
            d.rectangle([x, 30 + layer * 20, x + w, H], fill=col)
            for _ in range(5):
                by = rng.randrange(40, 200)
                d.line([(x + w / 2, by), (x + w / 2 + rng.choice((-1, 1)) * rng.randrange(14, 40), by - rng.randrange(8, 20))], fill=col, width=2)
        d.polygon([(0, 20 + layer * 20)] + [(x, 14 + layer * 20 + 8 * math.sin(x * 0.06 + layer)) for x in range(0, W + 10, 10)] + [(W, 20 + layer * 20)], fill=col)
    d.rectangle([0, 226, W, H], fill=(6, 18, 20))
    for _ in range(40):
        x = rng.randrange(W)
        h = rng.randrange(8, 20)
        d.polygon([(x, 228), (x - 4, 228 - h), (x + 5, 228 - h * 0.6)], fill=(10, 34, 32))
    return np.asarray(img, np.uint8)


NIGHT = _build_night()
_rf = np.random.default_rng(4)
FIREFLY = _rf.random((22, 5))


def fireflies(a, t, n=22):
    for u in FIREFLY[:n]:
        x = u[0] * W + 14 * math.sin(t * 0.6 + u[2] * 8)
        y = 120 + u[1] * 110 + 8 * math.sin(t * 0.9 + u[3] * 8)
        br = max(0.0, math.sin(t * 1.6 + u[4] * 10)) ** 3
        if br > 0.05:
            glow(a, x, y, 7, (200, 255, 110), 0.9 * br)
            px(a, x, y, (240, 255, 170), br)


def title_bg(t):
    a = NIGHT.astype(np.float32)
    # a few pixel butterflies glowing as they rise
    fireflies(a, t)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for i in range(6):
        x = (60 + i * 75 + 18 * math.sin(t * 0.7 + i)) % W
        y = 220 - ((t * 9 + i * 41) % 190)
        c1, c2 = BUTTERFLIES[i % 6]
        butterfly(d, x, y, t * 9 + i, c1, c2, 0.9)
    a = to_arr(img)
    for i in range(6):
        x = (60 + i * 75 + 18 * math.sin(t * 0.7 + i)) % W
        y = 220 - ((t * 9 + i * 41) % 190)
        glow(a, x, y, 18, BUTTERFLIES[i % 6][0], 0.25)
    glow(a, 240, 135, 200, (60, 160, 140), 0.18)
    return zoom_crop(a, 1 + 0.04 * t, 240, 135)


_rm = random.Random(21)
MOTHS = []
for _i in range(46):
    MOTHS.append(dict(
        ta=_rm.uniform(2, 21), sx=_rm.uniform(290, 384), sy=_rm.uniform(86, 206),
        r=_rm.uniform(30, 120), w=_rm.uniform(0.8, 2.2) * _rm.choice((-1, 1)), ph=_rm.uniform(0, 6.28),
        s=_rm.choice((1, 1, 1.4)), col=_rm.choice(((200, 170, 120), (235, 230, 215), (170, 140, 95), (210, 195, 150)))))


def night(t):
    a = NIGHT.astype(np.float32)
    lamp = (337, 66)
    glow(a, 337, 130, 150, (150, 190, 255), 0.55)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    # tent with a warm lantern
    d.polygon([(10, 236), (66, 172), (124, 236)], fill=(26, 38, 52))
    d.polygon([(56, 236), (66, 190), (78, 236)], fill=(255, 170, 84))
    # sheet on a rope
    d.line([(278, 74), (396, 74)], fill=(120, 100, 70))
    d.rectangle([276, 74, 280, 230], fill=(30, 22, 18))
    d.rectangle([394, 74, 398, 230], fill=(30, 22, 18))
    d.rectangle([282, 76, 392, 214], fill=(205, 214, 225))
    for yy in range(90, 214, 12):
        d.line([(282, yy), (392, yy)], fill=(190, 199, 212))
    a = to_arr(img)
    glow(a, 337, 135, 95, (255, 255, 255), 0.35)
    glow(a, 337, 66, 60, (200, 225, 255), 1.0)
    # beam
    img = to_img(a)
    d = ImageDraw.Draw(img)
    d.ellipse([333, 62, 341, 70], fill=(255, 255, 255))
    d.line([(337, 40), (337, 62)], fill=(40, 40, 40))
    # seated collector with notebook
    d.ellipse([214, 213, 222, 221], fill=(150, 105, 88))
    d.rectangle([209, 211, 224, 214], fill=(48, 36, 44))
    d.rectangle([212, 221, 222, 238], fill=(24, 20, 30))
    d.polygon([(210, 238), (226, 238), (238, 246), (210, 246)], fill=(24, 20, 30))
    d.rectangle([224, 228, 232, 234], fill=(225, 215, 180))
    # moths
    for m in MOTHS:
        orb = (lamp[0] + m["r"] * math.cos(m["w"] * t + m["ph"]), lamp[1] + 0.6 * m["r"] * math.sin(m["w"] * t + m["ph"] * 1.3))
        if t < m["ta"]:
            x, y = orb
            near = clamp(1.25 - math.hypot(x - lamp[0], y - lamp[1]) / 135)
            if near > 0.05:
                col = tuple(c * near for c in m["col"])
                tiny_moth(d, x, y, t * 30 + m["ph"] * 9, col, m["s"])
        else:
            u = smooth((t - m["ta"]) / 2.2)
            x, y = lerp(orb[0], m["sx"], u), lerp(orb[1], m["sy"], u)
            flap = (t * 30 + m["ph"] * 9) if u < 1 else (m["ph"] * 9 + (t * 20 if int(t * 0.5 + m["ph"] * 3) % 4 == 0 else 0))
            tiny_moth(d, x, y, flap if u < 1 else 1.4 + 0.1 * math.sin(flap), tuple(c * (0.55 + 0.45 * u) for c in m["col"]), m["s"] * (1 + 0.15 * u))
    # hero moth
    hu = smooth((t - 12) / 3)
    if hu > 0:
        hx = lerp(420, 340, hu)
        hy = lerp(40, 148, hu) + (1 - hu) * 18 * math.sin(t * 3)
        hero_moth(d, hx, hy, (t * 22) if hu < 1 else 0.3 + 0.6 * math.sin(t * 0.8), 0.8 + 0.2 * hu)
    a = to_arr(img)
    glow(a, 68, 220, 70, (255, 160, 70), 0.85)
    glow(a, 226, 232, 36, (255, 190, 100), 0.45)
    fireflies(a, t)
    # drifting insects in the lamp beam
    for i in range(26):
        u = DUST[i]
        x = 337 + 70 * math.sin(t * (0.5 + u[2]) + u[0] * 6)
        y = 70 + 130 * ((u[1] + t * 0.03 * (0.5 + u[2])) % 1)
        px(a, x, y, (255, 255, 255), 0.5)
    ez = smooth((t - 6) / 12) * 0.5 + smooth((t - 18) / 6) * 1.6
    return zoom_crop(a, 1 + ez, lerp(300, 345, smooth((t - 16) / 6)), lerp(135, 150, smooth((t - 16) / 6)))


# ================================================================ MEADOW
_rg = np.random.default_rng(9)
FLOWERS = [(int(_rg.random() * W), 226 + int(_rg.random() * 40), _rg.integers(0, 5), _rg.random()) for _ in range(70)]
FLCOL = [(255, 255, 255), (255, 235, 80), (255, 130, 170), (190, 140, 255), (255, 150, 60)]
BLADE = _rg.random((260, 3))
AMBIENT = [(i, BUTTERFLIES[(i + 1) % 6]) for i in range(6)]


def meadow(t):
    a = vgrad(W, H, [(0, (68, 120, 195)), (0.38, (238, 190, 150)), (0.62, (255, 215, 140)), (1, (255, 232, 172))])
    glow(a, 350, 120, 170, (255, 200, 110), 0.8)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    d.ellipse([340, 109, 360, 129], fill=(255, 246, 214))
    for i in range(5):
        cx = (i * 130 - t * 3 * (1 + i % 2 * 0.5)) % (W + 100) - 50
        cy = 28 + i * 15
        for k in range(5):
            d.ellipse([cx + k * 9 - 6, cy - 4 - (k % 2) * 3, cx + k * 9 + 14, cy + 5], fill=(255, 225, 200))
    xs = np.arange(W)
    for (base, amp, fr, col, sh, sp) in ((152, 20, 0.012, (150, 162, 205), 2, 1.5), (176, 17, 0.02, (112, 142, 156), 5, 3.0),
                                         (200, 12, 0.03, (74, 124, 94), 8, 6.0), (222, 7, 0.045, (46, 96, 62), 12, 10.0)):
        ys = ridge(xs, base, amp, fr, t * sp * fr, sh)
        d.polygon([(0, H)] + [(int(x), int(y)) for x, y in zip(xs, ys)] + [(W, H)], fill=col)
    d.rectangle([0, 228, W, H], fill=(52, 104, 56))
    for k in range(0, W, 6):
        d.line([(k, 232), (k - 4, 270)], fill=(60, 118, 62))
    # collector walks in, stops, swings the net, holds, releases
    px_ = -30 + 255 * smooth(t / 12.5)
    walking = t < 12.5
    phase = px_ * 0.45
    if t < 13.3:
        th = math.radians(-72)
    elif t < 14.0:
        th = math.radians(lerp(-72, -125, smooth((t - 13.3) / 0.7)))
    elif t < 14.5:
        th = math.radians(lerp(-125, -33, smooth((t - 14.0) / 0.5)))
    elif t < 20:
        th = math.radians(-33 + 2 * math.sin(t * 3))
    else:
        th = math.radians(lerp(-33, -52, smooth((t - 20) / 2)))
    hc = person(d, px_, 232, phase, th, walking)
    # hero butterfly
    if t < 14.4:
        hx_, hy_ = 262 + 7 * math.sin(3 * t), 184 + 6 * math.sin(4.3 * t)
        if t > 13.3:
            hx_, hy_ = lerp(hx_, hc[0], 0.3), lerp(hy_, hc[1], 0.3)
        butterfly(d, hx_, hy_, t * 12, (255, 226, 70), (50, 36, 24), 1.5)
    elif t < 23:
        butterfly(d, hc[0] + 1, hc[1] - 2, t * (4 if int(t) % 2 else 8), (255, 226, 70), (50, 36, 24), 1.2)
    else:
        u = t - 23
        butterfly(d, hc[0] + u * 22 + 4 * math.sin(u * 5), hc[1] - u * 26, t * 14, (255, 226, 70), (50, 36, 24), 1.5)
    for i, (c1, c2) in AMBIENT:
        x = (i * 90 + 30 * t * (0.5 + i * 0.1)) % (W + 60) - 30
        y = 130 + 40 * math.sin(t * 0.7 + i * 2) + 10 * math.sin(t * 2.3 + i)
        butterfly(d, x, y, t * 12 + i, c1, c2, 0.8)
    for fx, fy, c, u in FLOWERS:
        sw = math.sin(t * 1.5 + u * 9) * 1.5
        d.line([(fx, fy + 8), (fx + sw, fy)], fill=(40, 100, 50))
        d.rectangle([fx + sw - 1, fy - 1, fx + sw + 1, fy + 1], fill=FLCOL[c])
        d.point((fx + sw, fy), fill=(255, 200, 60))
    for u in BLADE:
        x = int(u[0] * W)
        h = 14 + int(u[1] * 28)
        sw = math.sin(t * 1.8 + u[2] * 12 + x * 0.05) * 3
        d.line([(x, 272), (x + sw, 272 - h)], fill=(int(30 + 30 * u[1]), int(80 + 60 * u[2]), 40), width=2)
    a = to_arr(img)
    glow(a, 350, 120, 200, (255, 200, 120), 0.18)
    for i in range(46):
        u = DUST[i]
        x = (u[0] * W + t * (6 + 10 * u[2])) % W
        y = 90 + u[1] * 150 + 6 * math.sin(t + u[3] * 7)
        px(a, x, y, (255, 240, 190), 0.35 + 0.5 * math.sin(t * 1.5 + u[3] * 9) ** 2)
    ze = smooth((t - 13.2) / 1.3) * (1 - smooth((t - 19) / 2.0))
    z2 = smooth((t - 22) / 1.5) * (1 - smooth((t - 27) / 2))
    return zoom_crop(a, 1 + 1.3 * ze + 0.5 * z2, lerp(W / 2, hc[0], max(ze, z2)), lerp(H / 2, hc[1], max(ze, z2)))


# ================================================================ MAP
MAP_NAMES = ["ЛУГА", "ГОРНЫЙ ЛЕС", "ПОБЕРЕЖЬЕ", "ТРОПИКИ", "САВАННА", "ОСТРОВА"]


def _build_map():
    n = vnoise(W, H, 11, 6, 3)
    n2 = vnoise(W, H, 12, 4, 6)
    h = n * 0.8 + n2 * 0.2
    yy, xx = np.mgrid[0:H, 0:W]
    edge = np.minimum(np.minimum(xx, W - 1 - xx) / 70, np.minimum(yy, H - 1 - yy) / 45).clip(0, 1)
    h = h * (0.5 + 0.5 * edge)
    sea = float(np.percentile(h, 52))
    land = h > sea
    top = float(h.max())
    sea_t = ((sea - h).clip(0, 0.25) / 0.25)
    arr = ramp(sea_t, [(0, (36, 92, 120)), (0.3, (24, 66, 98)), (1, (10, 26, 52))])
    lt = ((h - sea) / (top - sea)).clip(0, 1)
    land_col = ramp(lt, [(0, (214, 200, 140)), (0.07, (92, 150, 76)), (0.4, (52, 108, 58)), (0.65, (146, 126, 88)), (0.85, (206, 200, 190)), (1, (250, 250, 250))])
    arr = np.where(land[..., None], land_col, arr)
    cont = (np.floor(h * 22) != np.floor(np.roll(h, 1, 1) * 22))
    arr[cont & land] *= 0.82
    coast = land & ~(np.roll(land, 1, 0) & np.roll(land, -1, 0) & np.roll(land, 1, 1) & np.roll(land, -1, 1))
    arr[coast] = (236, 228, 190)
    near = ~land & (np.roll(land, 1, 0) | np.roll(land, -1, 0) | np.roll(land, 1, 1) | np.roll(land, -1, 1))
    arr[near] = (60, 130, 160)
    arr = (arr // 14) * 14 + 7
    for g in range(0, W, 40):
        arr[:, g] = arr[:, g] * 0.85 + 255 * 0.15
    for g in range(0, H, 40):
        arr[g, :] = arr[g, :] * 0.85 + 255 * 0.15
    # route points by farthest point sampling on land
    rng = np.random.default_rng(5)
    cand = []
    for _ in range(6000):
        x, y = int(rng.integers(40, W - 40)), int(rng.integers(50, H - 50))
        if h[y, x] > sea + 0.03:
            cand.append((x, y))
    pts = [min(cand, key=lambda p: p[0])]
    while len(pts) < 6:
        pts.append(max(cand, key=lambda c: min(math.hypot(c[0] - p[0], c[1] - p[1]) for p in pts)))
    order = [pts[0]]
    rest = pts[1:]
    while rest:
        nxt = min(rest, key=lambda p: math.hypot(p[0] - order[-1][0], p[1] - order[-1][1]))
        order.append(nxt)
        rest.remove(nxt)
    return arr, order


MAP, ROUTE_PTS = _build_map()
SEG_N = 48


def _bezier_route():
    segs = []
    for i in range(len(ROUTE_PTS) - 1):
        p0, p1 = ROUTE_PTS[i], ROUTE_PTS[i + 1]
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        sgn = 1 if i % 2 == 0 else -1
        c = (mx - dy * 0.22 * sgn, my + dx * 0.22 * sgn)
        pts = []
        for k in range(SEG_N + 1):
            u = k / SEG_N
            pts.append(((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * c[0] + u * u * p1[0],
                        (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * c[1] + u * u * p1[1]))
        segs.append(pts)
    return segs


ROUTE = _bezier_route()


def mapscene(t):
    tt = t - 2.5
    if tt < 0:
        seg, frac, r = 0, 0.0, 0.0
    else:
        seg = int(tt // 5.4)
        r = tt - seg * 5.4
        frac = clamp(r / 4.2)
    done = seg >= 5
    if done:
        seg, frac = 4, 1.0
    u = smooth(frac)
    k = min(SEG_N, int(u * SEG_N))
    pos = ROUTE[seg][k]
    visited = 1 + seg + (1 if frac >= 1 else 0)
    visited = min(visited, 6)
    zin = smooth(t / 2.5)
    zout = smooth((t - 29.5) / 5)
    z = lerp(1, 2.4, zin) * (1 - zout) + zout * 1.0
    if zout > 0:
        z = lerp(2.4, 1.0, zout)
    ccx = lerp(W / 2, pos[0], zin)
    ccy = lerp(H / 2, pos[1], zin)
    ccx = lerp(ccx, W / 2, zout)
    ccy = lerp(ccy, H / 2, zout)
    vw = W / z
    vh = H / z
    x0 = clamp(ccx - vw / 2, 0, W - vw)
    y0 = clamp(ccy - vh / 2, 0, H - vh)
    img = to_img(MAP).crop((int(x0), int(y0), int(x0 + vw), int(y0 + vh))).resize((W, H), Image.NEAREST)
    sx, sy = W / vw, H / vh

    def S(p):
        return ((p[0] - int(x0)) * sx, (p[1] - int(y0)) * sy)

    a = to_arr(img)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    # route
    pts_all = []
    for si in range(min(seg + 1, 5)):
        lim = SEG_N + 1 if si < seg else k + 1
        pts_all += ROUTE[si][:lim]
    if len(pts_all) > 1:
        for i, p in enumerate(pts_all):
            s = S(p)
            if i % 6 < 4:
                d.rectangle([s[0] - 1, s[1] - 1, s[0] + 1, s[1] + 1], fill=(255, 214, 96))
    # pins
    for i in range(visited):
        s = S(ROUTE_PTS[i])
        if not (6 < s[0] < W - 6 and 40 < s[1] < H - 40):
            continue
        arrive = 0.5 if i == 0 else 2.5 + (i - 1) * 5.4 + 4.2
        age = t - arrive
        if 0 < age < 1.6:
            rr = 4 + age * 18
            d.ellipse([s[0] - rr, s[1] - rr, s[0] + rr, s[1] + rr], outline=(255, 235, 150))
        d.ellipse([s[0] - 4, s[1] - 4, s[0] + 4, s[1] + 4], fill=(240, 80, 64), outline=(255, 255, 255))
        d.rectangle([s[0] - 1, s[1] - 1, s[0] + 1, s[1] + 1], fill=(255, 255, 255))
        if age > 0.4:
            label = MAP_NAMES[i]
            tw = d.textlength(label, font=FONT_S)
            lx = min(max(s[0] + 8, 6), W - tw - 14)
            ly = clamp(s[1] - 16, 40, H - 70)
            d.rectangle([lx - 3, ly - 2, lx + tw + 3, ly + 11], fill=(10, 20, 30), outline=(255, 214, 96))
            d.text((lx, ly), label, font=FONT_S, fill=(255, 240, 200))
    # butterfly marker on route head
    if not (done and frac >= 1):
        s = S(pos)
        butterfly(d, s[0], s[1] - 2, t * 14, (255, 140, 30), (80, 30, 20), 1.1)
    # HUD
    d.text((50, 196), "КАРТА НАХОДОК", font=FONT_M, fill=(255, 236, 190))
    d.text((50, 209), "ТОЧЕК: %d / 6" % visited, font=FONT_S, fill=(200, 220, 230))
    cx_, cy_ = 436, 206
    d.ellipse([cx_ - 14, cy_ - 14, cx_ + 14, cy_ + 14], outline=(255, 236, 190))
    d.polygon([(cx_, cy_ - 13), (cx_ - 4, cy_), (cx_ + 4, cy_)], fill=(240, 80, 64))
    d.polygon([(cx_, cy_ + 13), (cx_ - 4, cy_), (cx_ + 4, cy_)], fill=(230, 230, 230))
    d.text((cx_ - 3, cy_ - 27), "N", font=FONT_S, fill=(255, 236, 190))
    a = to_arr(img)
    glow(a, 240, 135, 300, (30, 60, 90), 0.0)
    return a


# ================================================================ TRAVEL
SKY_KEYS = [
    (0.0, (70, 130, 220), (190, 225, 255)),
    (0.5, (120, 150, 220), (255, 222, 172)),
    (0.8, (60, 50, 130), (255, 122, 92)),
    (1.0, (12, 14, 45), (120, 52, 92)),
]
SIGNS = [(700, "ГОРЫ"), (1180, "ЛЕС"), (1660, "РЕКА")]


def sky_at(p):
    for i in range(len(SKY_KEYS) - 1):
        a, b = SKY_KEYS[i], SKY_KEYS[i + 1]
        if p <= b[0]:
            u = (p - a[0]) / (b[0] - a[0])
            return lerpc(a[1], b[1], u), lerpc(a[2], b[2], u)
    return SKY_KEYS[-1][1], SKY_KEYS[-1][2]


def travel(t):
    p = t / 20
    top, bot = sky_at(p)
    a = vgrad(W, H, [(0, top), (0.75, bot), (1, bot)])
    sunx = 40 + 400 * p
    suny = 200 - 150 * math.sin(math.pi * clamp(p * 0.95))
    glow(a, sunx, suny, 190, (255, 200, 120), 0.8 * (1 - smooth((p - 0.9) * 10) * 0.6))
    img = to_img(a)
    d = ImageDraw.Draw(img)
    rng = random.Random(5)
    if p > 0.82:
        for _ in range(120):
            x, y = rng.randrange(W), rng.randrange(0, 140)
            b = int(255 * smooth((p - 0.82) * 6) * rng.random())
            d.point((x, y), fill=(b, b, b))
    d.ellipse([sunx - 9, suny - 9, sunx + 9, suny + 9], fill=(255, 244, 214))
    for i in range(4):
        cx = (i * 150 - t * 4 * (1 + i % 2 * 0.4)) % (W + 120) - 60
        cy = 34 + i * 14
        cc = lerpc((255, 255, 255), bot, 0.35 + 0.4 * p)
        for k in range(5):
            d.ellipse([cx + k * 10 - 7, cy - 4 - (k % 2) * 3, cx + k * 10 + 15, cy + 6], fill=tuple(int(c) for c in cc))
    xs = np.arange(W)
    layers = [(140, 42, 0.011, (62, 74, 112), 3, 4, 0.62), (170, 34, 0.017, (48, 70, 96), 6, 9, 0.46),
              (198, 24, 0.026, (40, 74, 80), 9, 18, 0.28), (222, 12, 0.04, (30, 62, 50), 13, 34, 0.12)]
    for base, amp, fr, col, sh, sp, haze in layers:
        c = lerpc(col, bot, haze)
        ys = ridge(xs, base, amp, fr, t * sp * fr, sh)
        d.polygon([(0, H)] + [(int(x), int(y)) for x, y in zip(xs, ys)] + [(W, H)], fill=tuple(int(v) for v in c))
        if base == 140:
            for x, y in list(zip(xs, ys))[::3]:
                if y < 128:
                    d.point((x, y + 1), fill=(240, 240, 250))
    ground = lerpc((52, 100, 54), (20, 16, 40), p * 0.7)
    d.rectangle([0, 238, W, H], fill=tuple(int(v) for v in ground))
    d.rectangle([0, 244, W, 258], fill=(112, 88, 62) if p < 0.7 else (60, 46, 52))
    sc = 70
    for wx in range(-100, W + 200, 90):
        x = (wx - t * sc) % (W + 200) - 100
        d.rectangle([x, 224, x + 2, 244], fill=(58, 40, 30))
        d.line([(x, 228), (x - 90, 230)], fill=(58, 40, 30))
    for wx, label in SIGNS:
        x = wx - t * sc
        if -40 < x < W + 40:
            d.rectangle([x, 208, x + 3, 244], fill=(80, 56, 38))
            d.polygon([(x + 3, 210), (x + 38, 210), (x + 44, 216), (x + 38, 222), (x + 3, 222)], fill=(190, 150, 90), outline=(60, 40, 24))
            d.fontmode = "1"
            d.text((x + 6, 211), label, font=FONT_S, fill=(60, 36, 20))
    person(d, 170, 250, t * 7.5, math.radians(-82), True, col=(20, 16, 26))
    for i in range(5):
        bx = (i * 110 - t * 22) % (W + 80) - 40
        by = 60 + 20 * math.sin(t * 0.8 + i * 1.7)
        w = math.sin(t * 9 + i) * 3
        d.line([(bx - 5, by + w), (bx, by), (bx + 5, by + w)], fill=(40, 36, 52))
    for i in range(70):
        gx = (i * 37 - t * 100) % (W + 40) - 20
        d.line([(gx, 272), (gx + math.sin(t * 2 + i) * 2, 262 - (i * 7) % 10)], fill=(34, 80, 44))
    a = to_arr(img)
    glow(a, sunx, suny, 120, (255, 190, 110), 0.15)
    if p > 0.8:
        fireflies(a, t, 14)
    return a


# ================================================================ MACRO
_rb = np.random.default_rng(31)
BOKEH = [(_rb.random() * W, _rb.random() * H, 8 + _rb.random() * 26, _rb.random(), _rb.random()) for _ in range(26)]


def macro(t):
    a = vgrad(W, H, [(0, (18, 52, 40)), (0.6, (60, 112, 62)), (1, (30, 70, 40))])
    glow(a, 360, 40, 230, (255, 226, 140), 0.7)
    bok = Image.new("RGB", (W, H), (0, 0, 0))
    bd = ImageDraw.Draw(bok)
    for (x, y, r, c, sp) in BOKEH:
        xx = (x - t * (3 + 6 * sp)) % (W + 60) - 30
        col = lerpc((160, 230, 120), (255, 232, 140), c)
        k = 0.25 + 0.4 * sp
        bd.ellipse([xx - r, y - r, xx + r, y + r], fill=tuple(int(v * k) for v in col))
    bok = bok.filter(ImageFilter.GaussianBlur(3))
    a += np.asarray(bok, np.float32)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    # leaf with scrolling veins
    d.polygon([(0, 196), (150, 178), (330, 176), (480, 198), (480, 270), (0, 270)], fill=(48, 112, 52))
    d.polygon([(0, 214), (160, 200), (330, 198), (480, 216), (480, 270), (0, 270)], fill=(40, 98, 46))
    off = (t * 10) % 40
    for k in range(-2, 14):
        x = k * 40 - off
        d.line([(x, 200), (x + 70, 270)], fill=(78, 146, 74))
    d.line([(0, 206), (480, 208)], fill=(120, 178, 100), width=2)
    for (dx, dy, r) in ((90, 236, 6), (390, 228, 5), (240, 252, 7), (440, 256, 4)):
        d.ellipse([dx - r, dy - r, dx + r, dy + r], fill=(160, 215, 235), outline=(220, 245, 255))
        d.point((dx - r // 2, dy - r // 2), fill=(255, 255, 255))
    cx, cy = 235 + 3 * math.sin(t * 0.6), 150 + 2 * math.sin(t * 1.1)
    sw = t * 3.0
    # legs
    for i, lx in enumerate((-34, -8, 22)):
        for side in (0, 1):
            ph = sw + i * 1.9 + side * math.pi
            fx = cx + lx + 14 * math.sin(ph) + side * 6
            fy = cy + 54 + 3 * max(0, math.cos(ph))
            kx = cx + lx + 18 + side * 4
            ky = cy + 30 - side * 4
            d.line([(cx + lx, cy + 22), (kx, ky), (fx, fy)], fill=(34, 16, 10), width=3)
            d.line([(kx, ky), (fx, fy)], fill=(62, 30, 18), width=1)
    # horn + head
    d.polygon([(cx + 70, cy - 8), (cx + 108, cy - 38 + 2 * math.sin(t)), (cx + 116, cy - 33), (cx + 84, cy - 2)], fill=(40, 20, 12))
    d.polygon([(cx + 46, cy - 22), (cx + 66, cy - 46), (cx + 72, cy - 42), (cx + 60, cy - 18)], fill=(46, 22, 14))
    d.ellipse([cx + 54, cy - 8, cx + 86, cy + 18], fill=(42, 20, 12))
    d.ellipse([cx + 74, cy - 2, cx + 80, cy + 4], fill=(230, 190, 140))
    d.point((cx + 77, cy + 1), fill=(10, 10, 10))
    ax = cx + 84
    for sgn in (0, 1):
        wv = math.sin(t * 2 + sgn) * 5
        d.line([(ax, cy + 4), (ax + 14, cy + 14 + wv), (ax + 24, cy + 8 + wv)], fill=(30, 14, 8), width=2)
        d.polygon([(ax + 24, cy + 8 + wv), (ax + 31, cy + 3 + wv), (ax + 31, cy + 13 + wv)], fill=(70, 34, 20))
    # pronotum + elytra with metallic sheen
    d.ellipse([cx + 18, cy - 28, cx + 70, cy + 26], fill=(36, 18, 12))
    for i in range(9):
        k = i / 8
        col = lerpc((40, 20, 12), (176, 96, 46), k ** 1.4)
        d.ellipse([cx - 72 + i * 3.4, cy - 34 + i * 1.6, cx + 32 - i * 5.5, cy + 32 - i * 3.2 - 4], fill=tuple(int(v) for v in col))
    d.line([(cx - 20, cy - 32), (cx - 20, cy + 20)], fill=(30, 14, 8))
    for i in range(8):
        k = i / 7
        col = lerpc((50, 26, 16), (200, 126, 70), k ** 1.5)
        d.ellipse([cx + 22 + i * 2, cy - 24 + i * 1.5, cx + 68 - i * 4, cy + 22 - i * 3], fill=tuple(int(v) for v in col))
    d.ellipse([cx - 52, cy - 26, cx - 12, cy - 18], fill=(255, 232, 200))
    d.ellipse([cx + 28, cy - 18, cx + 46, cy - 12], fill=(255, 232, 200))
    a = to_arr(img)
    glow(a, cx - 30, cy - 22, 60, (255, 210, 140), 0.25)
    for i in range(30):
        u = DUST[i]
        x = (u[0] * W - t * (4 + 8 * u[2])) % W
        y = (u[1] * H + t * 3) % H
        px(a, x, y, (255, 245, 200), 0.3 + 0.4 * math.sin(t + u[3] * 9) ** 2)
    return zoom_crop(a, 1 + 0.12 * t / 9, cx, cy)


SCENES = {
    "cabinet": cabinet, "title": title_bg, "meadow": meadow, "night": night,
    "map": mapscene, "travel": travel, "macro": macro,
}
