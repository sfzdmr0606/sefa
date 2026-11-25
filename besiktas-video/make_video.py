"""Animated 9:16 short (1080x1920, 30 fps, 16 s) from one family photo.

Usage: python3 make_video.py aile.png build/music.wav out.mp4
Frames are drawn with Pillow/numpy and piped straight into ffmpeg.
"""
import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

SRC, MUSIC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREVIEW = sys.argv[4:]  # optional list of times -> write PNGs instead of video
W, H, FPS, DUR = 1080, 1920, 30, 16.0
BEAT = 0.5
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
rng = np.random.default_rng(1903)

src = Image.open(SRC).convert("RGB")
SW, SH = src.size


# ------------------------------------------------------------------ helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_back(x, s=2.2):
    x = clamp(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def beat_pulse(t, start, decay=9.0):
    """1 right on each beat after `start`, decaying to 0."""
    if t < start:
        return 0.0
    since = (t - start) % BEAT
    return math.exp(-since * decay)


def cartoonize(img):
    """Cartoon look: edge-preserving smoothing + colour quantization + ink lines."""
    import cv2

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    s = bgr
    for _ in range(2):
        s = cv2.bilateralFilter(s, 9, 60, 9)
    s = cv2.edgePreservingFilter(s, flags=1, sigma_s=50, sigma_r=0.35)
    Z = s.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(1903)
    _, lab, cen = cv2.kmeans(Z, 28, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    q = cen[lab.flatten()].reshape(s.shape).astype(np.uint8)
    q = cv2.addWeighted(q, 0.7, s, 0.3, 0)
    g = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7)
    e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 8)
    e = cv2.medianBlur(e, 3)
    out = cv2.bitwise_and(q, q, mask=e)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = (hsv[..., 1] * 1.25).clip(0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(out)


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    r = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


toon = cartoonize(src)


def make_bg(img):
    b = cover(img, W, H).filter(ImageFilter.GaussianBlur(38))
    b = ImageEnhance.Color(b).enhance(0.35)
    return ImageEnhance.Brightness(b).enhance(0.42)


BG = {"photo": np.array(make_bg(src)).astype(np.float32), "toon": np.array(make_bg(toon)).astype(np.float32)}

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ANG = np.arctan2(yy - H * 0.5, xx - W * 0.5)
VIGN = (1 - 0.55 * (((xx - W / 2) / (W * 0.8)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2)).clip(0.25, 1)[..., None]


def stripes_bg(t, speed=180, width=135, dark=0.0):
    x = (xx[0] + t * speed) // width % 2  # one row, broadcast
    row = np.where(x == 0, 245.0, 14.0)
    img = np.broadcast_to(row[None, :, None], (H, W, 3)).copy()
    img = img * VIGN
    if dark:
        img *= 1 - dark
    return img


def rays_layer(base, t, strength=0.18):
    sector = np.floor((ANG + t * 0.6) / (2 * np.pi / 28)) % 2
    return base * (1 - strength * sector[..., None]) + 255 * strength * 0.35 * sector[..., None]


_font_cache = {}
_text_cache = {}


def font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT, size)
    return _font_cache[size]


def text_img(txt, size, fill=(255, 255, 255), stroke=(0, 0, 0), sw=None, shadow=True):
    key = (txt, size, fill, stroke, sw, shadow)
    if key in _text_cache:
        return _text_cache[key]
    sw = sw if sw is not None else max(4, size // 18)
    f = font(size)
    l, tp, r, b = f.getbbox(txt, stroke_width=sw)
    pad = 40
    im = Image.new("RGBA", (r - l + pad * 2, b - tp + pad * 2), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l + 8, pad - tp + 12), txt, font=f, fill=(0, 0, 0, 170), stroke_width=sw, stroke_fill=(0, 0, 0, 170))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(10)))
    ImageDraw.Draw(im).text((pad - l, pad - tp), txt, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke)
    _text_cache[key] = im
    return im


def paste(frame, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if alpha <= 0.01 or scale <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 0.999:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def pop_text(frame, txt, size, cx, cy, t, t0, dur=0.35, bounce_from=None, **kw):
    if t < t0:
        return
    p = (t - t0) / dur
    s = 0.2 + 0.8 * ease_back(p) if p < 1 else 1.0
    if bounce_from is not None:
        s *= 1 + 0.06 * beat_pulse(t, bounce_from)
    paste(frame, text_img(txt, size, **kw), cx, cy, s, clamp(p * 3))


def star(draw, cx, cy, r, fill, outline=None):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    draw.polygon(pts, fill=fill, outline=outline, width=6 if outline else 0)


def stars(frame, cx, cy, t, t0, r=46, gap=130):
    d = ImageDraw.Draw(frame)
    for i in (-1, 0, 1):
        p = (t - t0 - (i + 1) * 0.1) / 0.3
        if p <= 0:
            continue
        rr = r * (0.3 + 0.7 * ease_back(p))
        star(d, cx + i * gap, cy + 5, rr * 1.18, (0, 0, 0, 200))
        star(d, cx + i * gap, cy, rr, (255, 255, 255, 255), (0, 0, 0, 255))


def view(img, cx, cy, zoom, ox=0.0, oy=0.0):
    """Full-screen 9:16 window of `img` centred near (cx, cy). zoom=1 -> full image height."""
    ch = SH / zoom
    cw = ch * W / H
    x0 = clamp(cx - cw / 2 + ox, 0, SW - cw)
    y0 = clamp(cy - ch / 2 + oy, 0, SH - ch)
    return img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)


CARD_W = 1000
CARD_H = int(CARD_W * SH / SW)
BORDER = 14
_shadow = Image.new("RGBA", (CARD_W + 2 * BORDER + 120, CARD_H + 2 * BORDER + 120), (0, 0, 0, 0))
ImageDraw.Draw(_shadow).rectangle((60, 75, 60 + CARD_W + 2 * BORDER, 75 + CARD_H + 2 * BORDER), fill=(0, 0, 0, 200))
SHADOW = _shadow.filter(ImageFilter.GaussianBlur(24))


def card(img, kb):
    """Photo card with white border; kb = ken-burns zoom (>=1)."""
    cw, ch = SW / kb, SH / kb
    x0, y0 = (SW - cw) / 2, (SH - ch) / 2 - (SH - ch) * 0.2
    ph = img.transform((CARD_W, CARD_H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
    c = Image.new("RGBA", (CARD_W + 2 * BORDER, CARD_H + 2 * BORDER), (255, 255, 255, 255))
    c.paste(ph, (BORDER, BORDER))
    return c


def place_card(frame, c, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    paste(frame, SHADOW, cx, cy + 10, scale, 0.9 * alpha, rot)
    paste(frame, c, cx, cy, scale, alpha, rot)


def wipe_mix(a, b, p):
    """Diagonal wipe from image a to b, p in 0..1 (PIL RGB images)."""
    if p <= 0:
        return a
    if p >= 1:
        return b
    h, w = a.size[1], a.size[0]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    edge = (gx * 0.35 + gy) / (w * 0.35 + h)
    m = ((p * 1.2 - edge) * 25).clip(0, 1)[..., None]
    out = np.array(a, dtype=np.float32) * (1 - m) + np.array(b, dtype=np.float32) * m
    return Image.fromarray(out.astype(np.uint8))


# confetti
NCONF = 140
conf = {
    "x": rng.uniform(0, W, NCONF),
    "y0": rng.uniform(-H * 1.1, -20, NCONF),
    "vy": rng.uniform(260, 520, NCONF),
    "sway": rng.uniform(20, 70, NCONF),
    "ph": rng.uniform(0, 6.28, NCONF),
    "rot": rng.uniform(0, 6.28, NCONF),
    "vr": rng.uniform(-6, 6, NCONF),
    "w": rng.uniform(14, 28, NCONF),
    "h": rng.uniform(7, 14, NCONF),
    "c": rng.integers(0, 10, NCONF),
}


def confetti(frame, t, t0):
    if t < t0:
        return
    dt = t - t0
    d = ImageDraw.Draw(frame)
    for i in range(NCONF):
        y = conf["y0"][i] + conf["vy"][i] * dt
        if y < -40 or y > H + 40:
            continue
        x = conf["x"][i] + conf["sway"][i] * math.sin(dt * 3 + conf["ph"][i])
        a = conf["rot"][i] + conf["vr"][i] * dt
        squash = abs(math.cos(dt * 5 + conf["ph"][i]))  # flipping paper
        w, h = conf["w"][i], conf["h"][i] * (0.25 + 0.75 * squash)
        ca, sa = math.cos(a), math.sin(a)
        pts = [(x + px * ca - py * sa, y + px * sa + py * ca) for px, py in ((-w, -h), (w, -h), (w, h), (-w, h))]
        k = conf["c"][i]
        col = (255, 255, 255, 255) if k < 5 else (10, 10, 10, 255) if k < 9 else (227, 10, 23, 255)
        d.polygon(pts, fill=col, outline=(0, 0, 0, 255) if k < 5 else (255, 255, 255, 255))


def flash(frame, t, at, length=0.25, strength=1.0):
    if at <= t < at + length:
        a = strength * (1 - (t - at) / length) ** 2
        frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(255 * a))))


def band(frame, cy, h, alpha=230):
    d = ImageDraw.Draw(frame)
    d.rectangle((0, cy - h // 2, W, cy + h // 2), fill=(0, 0, 0, alpha))
    d.rectangle((0, cy - h // 2, W, cy - h // 2 + 8), fill=(255, 255, 255, 255))
    d.rectangle((0, cy + h // 2 - 8, W, cy + h // 2), fill=(255, 255, 255, 255))
    d.rectangle((0, cy - h // 2 + 14, W, cy - h // 2 + 19), fill=(227, 10, 23, 255))
    d.rectangle((0, cy + h // 2 - 19, W, cy + h // 2 - 14), fill=(227, 10, 23, 255))


def shake(t, start, amp=16):
    p = beat_pulse(t, start, 14)
    return (amp * p * math.sin(t * 97.0), amp * p * math.cos(t * 71.0))


# faces in source photo (x, y)
GIRL, BOY, DAD = (300, 520), (830, 560), (1120, 520)


# ------------------------------------------------------------------ frame
def render(t):
    # ---------------- A: intro 0-2s
    if t < 2.0:
        frame = Image.fromarray(stripes_bg(t, 220).astype(np.uint8)).convert("RGBA")
        bh = int(300 * ease_out(t / 0.35))
        if bh > 4:
            band(frame, 860, bh)
        word = "BEŞİKTAŞ"
        f = font(168)
        total = f.getlength(word)
        x = W / 2 - total / 2
        for i, ch in enumerate(word):
            w = f.getlength(ch)
            t0 = 0.25 + i * 0.125
            if t >= t0:
                p = (t - t0) / 0.25
                paste(frame, text_img(ch, 168, sw=6), x + w / 2, 860 + (1 - ease_out(p)) * -120, 0.3 + 0.7 * ease_back(p), clamp(p * 3))
            x += w
        stars(frame, W / 2, 610, t, 1.3)
        pop_text(frame, "1903", 120, W / 2, 1120, t, 1.55, 0.3)
        flash(frame, t, 1.92, 0.08, 0.6)
        flash(frame, t, 2.0, 0.0, 0)
        return frame

    # ---------------- B: drop, family card 2-4s
    if t < 4.0:
        lt = t - 2.0
        base = rays_layer(BG["photo"], t, 0.10)
        frame = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        sx, sy = shake(t, 2.0, 18)
        s = 1.35 - 0.35 * ease_out(lt / 0.45)
        s *= 1 + 0.025 * beat_pulse(t, 2.0)
        place_card(frame, card(src, 1.0 + 0.08 * lt / 2), W / 2 + sx, 1060 + sy, s)
        pop_text(frame, "AİLECEK", 140, W / 2, 360, t, 2.05, bounce_from=2.0)
        pop_text(frame, "KARTAL", 190, W / 2, 545, t, 3.0, bounce_from=3.0)
        flash(frame, t, 2.0, 0.35)
        return frame

    # ---------------- C: close-ups 4-10s
    if t < 10.0:
        idx = int((t - 4.0) // 2)
        lt = (t - 4.0) - idx * 2
        (cx, cy), word = [(GIRL, "SİYAH"), (BOY, "BEYAZ"), (DAD, "TEK AŞK")][idx]
        zoom = 1.06 + 0.10 * lt / 2 + 0.05 * beat_pulse(t, 4.0, 10)
        pan = (-40 + 80 * lt / 2) * (1 if idx != 1 else -1)
        img = view(src, cx, cy, zoom, ox=pan)
        # whip blur on entry
        if lt < 0.14 and idx > 0:
            a = np.array(img, dtype=np.float32)
            k = int(70 * (1 - lt / 0.14))
            acc = sum(np.roll(a, s, axis=1) for s in range(-k, k + 1, max(1, k // 6)))
            img = Image.fromarray((acc / len(range(-k, k + 1, max(1, k // 6)))).astype(np.uint8))
        frame = img.convert("RGBA")
        # jersey-stripe side accents
        d = ImageDraw.Draw(frame)
        for i in range(4):
            col = (255, 255, 255, 255) if i % 2 == 0 else (0, 0, 0, 255)
            grow = ease_out((lt - i * 0.05) / 0.3)
            d.rectangle((i * 22, 0, i * 22 + 22, int(H * grow)), fill=col)
            d.rectangle((W - (i + 1) * 22, H - int(H * grow), W - i * 22, H), fill=col)
        band_h = int(230 * ease_out((lt - 0.05) / 0.25))
        if band_h > 4:
            band(frame, 1380, band_h, 215)
        pop_text(frame, word, 170, W / 2, 1380, t, 4.0 + idx * 2 + 0.1, bounce_from=4.0)
        flash(frame, t, 4.0 + idx * 2, 0.18, 0.8 if idx == 0 else 0.5)
        return frame

    # ---------------- D: cartoon / celebration 10-14s
    if t < 14.0:
        lt = t - 10.0
        p_toon = clamp((t - 10.0) / 0.3) - clamp((t - 12.5) / 0.3)
        bg_key = "toon" if p_toon > 0.5 else "photo"
        base = rays_layer(BG[bg_key], t * 2, 0.22)
        frame = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        sx, sy = shake(t, 10.0, 12)
        kb = 1.0 + 0.10 * lt / 4
        c_photo, c_toon = card(src, kb), card(toon, kb)
        if p_toon <= 0:
            c = c_photo
        elif p_toon >= 1:
            c = c_toon
        else:
            a, b = (c_photo, c_toon) if t < 12 else (c_toon, c_photo)
            c = wipe_mix(a.convert("RGB"), b.convert("RGB"), p_toon if t < 12 else 1 - p_toon).convert("RGBA")
        rot = 2.5 * math.sin(lt * math.pi / 2)
        s = 0.96 + 0.03 * beat_pulse(t, 10.0)
        place_card(frame, c, W / 2 + sx, 1060 + sy, s, rot)
        pop_text(frame, "EN BÜYÜK", 150, W / 2, 330, t, 10.0, bounce_from=10.0)
        pop_text(frame, "BEŞİKTAŞ", 175, W / 2, 520, t, 11.0, bounce_from=11.0)
        confetti(frame, t, 10.0)
        flash(frame, t, 10.0, 0.2, 0.7)
        flash(frame, t, 12.5, 0.15, 0.4)
        return frame

    # ---------------- E: finale 14-16s
    lt = t - 14.0
    frame = Image.fromarray(stripes_bg(t, 60, dark=0.35).astype(np.uint8)).convert("RGBA")
    s = 1.18 - 0.2 * ease_out(lt / 0.5)
    place_card(frame, card(src, 1.12 - 0.08 * ease_out(lt / 2)), W / 2, 1010, s, -2.0)
    band(frame, 345, 330)
    pop_text(frame, "AİLECEK", 110, W / 2, 275, t, 14.05)
    pop_text(frame, "BEŞİKTAŞLIYIZ", 104, W / 2, 400, t, 14.3)
    d_y = 1010 + CARD_H // 2 + 150
    stars(frame, W / 2, d_y, t, 14.6, r=40, gap=110)
    pop_text(frame, "1903", 110, W / 2, d_y + 120, t, 14.9)
    confetti(frame, t, 13.2)
    flash(frame, t, 14.0, 0.4)
    return frame


# ------------------------------------------------------------------ output
if PREVIEW:
    for ts in PREVIEW:
        render(float(ts)).convert("RGB").save(f"build/preview_{ts}.png")
    sys.exit()

ff = subprocess.Popen(
    [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", MUSIC,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high", "-level", "4.2",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart", OUT,
    ],
    stdin=subprocess.PIPE,
)
n = int(DUR * FPS)
for i in range(n):
    ff.stdin.write(render(i / FPS).convert("RGB").tobytes())
    if i % 60 == 0:
        print(f"frame {i}/{n}", flush=True)
ff.stdin.close()
ff.wait()
print("wrote", OUT)
