"""2D puppet renderer: template characters drawn in code and animated by the voice track.

Used when no generative image/video model is available (no API key): the rest of the pipeline
(verbatim lines, TTS, timing, captions, QA) is unchanged. Characters are picked from a small template
library by keywords in name/appearance (cabbage, beet, potato, carrot, tomato; anything else -> blob).
The mouth opens with the loudness of the speaker's line, eyes blink, brows follow the line's `delivery`.
"""

from __future__ import annotations

import hashlib
import math
import random
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .media import FFMPEG, Caption, MediaError
from .models import Character, ScriptPlan, ShotPlan

SS = 2  # supersampling factor for antialiased shapes
SPRITE_W, SPRITE_H = 520, 640
GROUND_Y = 1460

KINDS = {
    "cabbage": ("cabbage", "капуст"),
    "beet": ("beet", "буряк"),
    "potato": ("potato", "картопл"),
    "carrot": ("carrot", "морк"),
    "tomato": ("tomato", "помідор", "томат"),
    "cucumber": ("cucumber", "огір", "огур"),
}
ACCESSORIES = {
    "apron": ("apron", "фартух"),
    "headscarf": ("headscarf", "scarf", "хустк"),
    "glasses": ("glasses", "окуляр"),
    "bowtie": ("bow tie", "bowtie", "метелик"),
    "bow": ("hair bow", "pink bow", "бант"),
}
SCENES = {
    "garden": ("garden", "город", "грядк", "vegetable patch"),
    "kitchen": ("kitchen", "кухн", "борщ", "pot", "каструл"),
}


def _has(text: str, keys: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(k in t for k in keys)


def kind_of(c: Character) -> str:
    # the name wins: "Огірок" with appearance "cabbage template styled as a cucumber" is a cucumber
    for text in (c.name, c.appearance):
        kind = next((k for k, keys in KINDS.items() if _has(text, keys)), None)
        if kind:
            return kind
    return "blob"


def accessories_of(c: Character) -> set[str]:
    found = {a for a, keys in ACCESSORIES.items() if _has(c.appearance, keys)}
    return found - {"bow"} if "bowtie" in found else found


# ------------------------------------------------------------------ expressions
@dataclass(frozen=True)
class Expr:
    brow: float  # +1 angry (inner ends down), -1 sad (inner ends up)
    eye_open: float
    smile: float  # -1 frown .. +1 smile


_EXPRS = [
    (("angry", "furious", "rage", "shout", "yell", "betray", "лют", "зл", "крич"), Expr(0.9, 1.0, -0.7)),
    (
        ("sad", "tear", "cry", "guilt", "sigh", "whisper", "ashamed", "сум", "плач", "винн"),
        Expr(-0.8, 0.72, -0.4),
    ),
    (("laugh", "tease", "happy", "joy", "playful", "grin", "смі", "рад"), Expr(-0.15, 0.85, 0.95)),
    (("proud", "grump", "stubborn", "boast", "бурч", "горд"), Expr(0.5, 0.85, -0.15)),
    (("surpris", "shock", "здив"), Expr(-0.5, 1.15, 0.0)),
]
NEUTRAL = Expr(0.0, 1.0, 0.25)


def expr_of(delivery: str) -> Expr:
    return next((e for keys, e in _EXPRS if _has(delivery, keys)), NEUTRAL)


# ------------------------------------------------------------------ drawing helpers
def _ellipse_poly(cx, cy, rx, ry, angle=0.0, n=72, wobble=None):
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = 1.0 + (wobble(t) if wobble else 0.0)
        x, y = rx * r * math.cos(t), ry * r * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


@dataclass
class Face:
    cx: float  # in sprite 1x coordinates (relative to sprite box)
    cy: float
    s: float  # face scale (1.0 = eye radius 22px at sprite scale 1)


def _draw_body(kind: str, acc: set[str], scale: float, seed: int) -> tuple[Image.Image, Face]:
    """Body sprite (RGBA, 1x after downsampling) + where the face goes."""
    W, H = int(SPRITE_W * scale * SS), int(SPRITE_H * scale * SS)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rnd = random.Random(seed)
    k = scale * SS
    cx = W / 2
    ow = int(5 * k)  # outline width
    face = Face(0.5, 0.56, 1.0)

    if kind == "cabbage":
        cy, r = H * 0.60, W * 0.40
        for ang in (-0.9, 0.9, -0.35, 0.35):  # outer leaves
            lx = cx + math.sin(ang) * r * 0.75
            d.polygon(
                _ellipse_poly(lx, cy - r * 0.05, r * 0.55, r * 0.85, ang), fill="#5f9e3c", outline="#3d6e25"
            )
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#9fd36a", outline="#4f8a30", width=ow)
        for i, off in enumerate((-0.55, -0.15, 0.3)):  # leaf veins / layers
            x0 = cx + off * r
            d.arc(
                (x0 - r * 0.9, cy - r * 0.95, x0 + r * 0.9, cy + r * 0.95),
                200 + i * 8,
                330,
                fill="#6fae46",
                width=int(4 * k),
            )
        face = Face(0.5, 0.58, 1.05)
    elif kind == "beet":
        cy, r = H * 0.57, W * 0.36
        for ang in (-0.45, 0.0, 0.45):  # leaves with red stems
            tx, ty = cx + math.sin(ang) * r * 1.1, cy - r * 1.55
            d.line((cx, cy - r * 0.8, tx, ty + r * 0.3), fill="#a3184a", width=int(7 * k))
            d.polygon(_ellipse_poly(tx, ty, r * 0.28, r * 0.55, ang), fill="#4e9a3a", outline="#2f6b22")
        d.polygon(
            [(cx - r * 0.72, cy + r * 0.55), (cx + r * 0.72, cy + r * 0.55), (cx, H * 0.985)], fill="#8e1f4f"
        )
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#8e1f4f", outline="#5c0f31", width=ow)
        d.ellipse((cx - r * 0.65, cy - r * 0.75, cx - r * 0.15, cy - r * 0.3), fill="#b23a6b")
        face = Face(0.5, 0.56, 0.95)
    elif kind == "potato":
        cy, rx, ry = H * 0.60, W * 0.41, H * 0.33
        bumps = [(rnd.uniform(0, 6.28), rnd.uniform(0.02, 0.05), rnd.randint(2, 4)) for _ in range(3)]

        def wob(t):
            return sum(a * math.sin(f * t + p) for p, a, f in bumps)

        d.polygon(_ellipse_poly(cx, cy, rx, ry, 0.05, wobble=wob), fill="#c9965a", outline="#8a5d2c")
        for _ in range(7):
            sx, sy = cx + rnd.uniform(-0.75, 0.75) * rx, cy + rnd.uniform(-0.7, 0.75) * ry
            if abs(sx - cx) < rx * 0.45 and sy < cy + ry * 0.25:
                continue  # keep the face area clean
            sr = rnd.uniform(5, 9) * k
            d.ellipse((sx - sr, sy - sr * 0.7, sx + sr, sy + sr * 0.7), fill="#9c6a37")
        face = Face(0.5, 0.57, 1.0)
    elif kind == "carrot":
        top, r = H * 0.30, W * 0.30
        for ang in (-0.5, -0.15, 0.2, 0.55):  # leaf tuft
            d.polygon(
                _ellipse_poly(cx + math.sin(ang) * r * 0.6, top - r * 0.65, r * 0.16, r * 0.7, ang),
                fill="#4f9b38",
                outline="#2f6b22",
            )
        d.ellipse(
            (cx - r, top - r * 0.45, cx + r, top + r * 0.45), fill="#f28a1e", outline="#b85d0a", width=ow
        )
        cone = [(cx - r, top), (cx + r, top), (cx + r * 0.08, H * 0.985), (cx - r * 0.08, H * 0.985)]
        d.polygon(cone, fill="#f28a1e")
        d.line([cone[0], cone[3]], fill="#b85d0a", width=ow)
        d.line([cone[1], cone[2]], fill="#b85d0a", width=ow)
        for y in (0.62, 0.74, 0.86):
            yy = H * y
            hw = r * (1 - (yy - top) / (H - top)) * 0.7
            d.line((cx - hw, yy, cx - hw * 0.3, yy + 6 * k), fill="#c96a10", width=int(4 * k))
        face = Face(0.5, 0.40, 1.0)
    elif kind == "tomato":
        cy, r = H * 0.62, W * 0.40
        d.ellipse((cx - r, cy - r * 0.9, cx + r, cy + r * 0.9), fill="#e0412f", outline="#9c2215", width=ow)
        d.ellipse((cx - r * 0.6, cy - r * 0.7, cx - r * 0.2, cy - r * 0.35), fill="#f07a63")
        star = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r * (0.42 if i % 2 == 0 else 0.16)
            star.append((cx + rr * math.cos(a), cy - r * 0.86 + rr * math.sin(a) * 0.55))
        d.polygon(star, fill="#3f8a2e")
        face = Face(0.5, 0.63, 1.0)
    elif kind == "cucumber":
        cy, rx, ry = H * 0.60, W * 0.27, H * 0.38
        d.polygon(_ellipse_poly(cx, cy, rx, ry, 0.0), fill="#5c9e3a", outline="#2f6420")
        d.ellipse((cx - rx * 0.55, cy - ry * 0.8, cx - rx * 0.1, cy + ry * 0.6), fill="#74b54f")
        for _ in range(14):  # bumps
            a = rnd.uniform(0, 6.28)
            bx, by = (
                cx + math.cos(a) * rx * rnd.uniform(0.55, 0.92),
                cy + math.sin(a) * ry * rnd.uniform(0.5, 0.92),
            )
            if abs(bx - cx) < rx * 0.6 and abs(by - (cy - ry * 0.12)) < ry * 0.3:
                continue
            d.ellipse((bx - 4 * k, by - 4 * k, bx + 4 * k, by + 4 * k), fill="#3e7a26")
        d.ellipse((cx - 18 * k, cy - ry - 10 * k, cx + 18 * k, cy - ry + 14 * k), fill="#c8b45a")
        face = Face(0.5, 0.50, 0.8)
    else:
        h = int(hashlib.sha256(str(seed).encode()).hexdigest()[:6], 16)
        col = (80 + h % 150, 80 + (h >> 8) % 150, 80 + (h >> 16) % 150)
        d.ellipse((cx - W * 0.4, H * 0.28, cx + W * 0.4, H * 0.97), fill=col, outline="#333333", width=ow)
        face = Face(0.5, 0.58, 1.0)

    fx, fy = face.cx * W, face.cy * H
    if "apron" in acc:
        aw, ay = W * 0.2, fy + 100 * k * face.s
        ab = min(H * 0.95, ay + H * 0.24)
        d.polygon(
            [(fx - aw * 0.8, ay), (fx + aw * 0.8, ay), (fx + aw * 1.15, ab), (fx - aw * 1.15, ab)],
            fill="#fbf2f2",
            outline="#e0a8a8",
        )
        for x in range(-3, 4):  # polka dots
            for y in (0.3, 0.65):
                px, py = fx + x * aw * 0.3, ay + (ab - ay) * y
                d.ellipse((px - 5 * k, py - 5 * k, px + 5 * k, py + 5 * k), fill="#e36b5d")
        d.line((fx - aw * 1.25, ay + 4 * k, fx + aw * 1.25, ay + 4 * k), fill="#e36b5d", width=int(7 * k))
    if "headscarf" in acc:
        top = fy - 150 * k * face.s
        d.polygon(
            [(fx - W * 0.36, top + 40 * k), (fx + W * 0.36, top + 40 * k), (fx, top - 70 * k)],
            fill="#d23b3b",
            outline="#8f1f1f",
        )
        for i in range(5):
            dx = (i - 2) * W * 0.11
            d.ellipse((fx + dx - 7 * k, top + 5 * k, fx + dx + 7 * k, top + 19 * k), fill="#fbe9c7")
        d.polygon(
            [(fx + W * 0.3, top + 40 * k), (fx + W * 0.42, top + 80 * k), (fx + W * 0.36, top + 20 * k)],
            fill="#d23b3b",
        )
    if "bowtie" in acc:
        tx, ty = fx, fy + 120 * k * face.s
        d.polygon([(tx, ty), (tx - 40 * k, ty - 24 * k), (tx - 40 * k, ty + 24 * k)], fill="#1f2a44")
        d.polygon([(tx, ty), (tx + 40 * k, ty - 24 * k), (tx + 40 * k, ty + 24 * k)], fill="#1f2a44")
        d.ellipse((tx - 10 * k, ty - 10 * k, tx + 10 * k, ty + 10 * k), fill="#34456b")
    if "bow" in acc:
        bx, by = fx + W * 0.18, fy - 160 * k * face.s
        d.polygon([(bx, by), (bx - 45 * k, by - 30 * k), (bx - 45 * k, by + 30 * k)], fill="#ff5fa2")
        d.polygon([(bx, by), (bx + 45 * k, by - 30 * k), (bx + 45 * k, by + 30 * k)], fill="#ff5fa2")
        d.ellipse((bx - 12 * k, by - 12 * k, bx + 12 * k, by + 12 * k), fill="#e23b82")

    small = im.resize((W // SS, H // SS), Image.LANCZOS)
    return small, face


def _draw_face(
    size: tuple[int, int],
    face: Face,
    scale: float,
    e: Expr,
    mouth_open: float,
    blink: float,
    look: float,
    glasses: bool,
) -> Image.Image:
    """Face layer for one frame, same size as the 1x sprite. Drawn supersampled, then downsized."""
    W, H = size[0] * SS, size[1] * SS
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = scale * SS * face.s
    fx, fy = face.cx * W, face.cy * H
    er, dx, ey = 22 * k, 52 * k, fy - 18 * k

    for side in (-1, 1):  # cheeks
        d.ellipse(
            (
                fx + side * 80 * k - 20 * k,
                fy + 22 * k - 11 * k,
                fx + side * 80 * k + 20 * k,
                fy + 22 * k + 11 * k,
            ),
            fill=(255, 120, 130, 90),
        )
    open_ = max(0.08, e.eye_open * blink)
    for side in (-1, 1):
        ex = fx + side * dx
        d.ellipse(
            (ex - er, ey - er * 1.15 * open_, ex + er, ey + er * 1.15 * open_),
            fill="white",
            outline="#222",
            width=int(3 * k / face.s),
        )
        if open_ > 0.3:
            pr = er * 0.52
            px, py = ex + look * er * 0.35, ey + er * 0.1
            d.ellipse(
                (px - pr, py - pr * min(1, open_ * 1.1), px + pr, py + pr * min(1, open_ * 1.1)),
                fill="#1d1d1f",
            )
            d.ellipse(
                (px - pr * 0.25 - 4 * k, py - pr * 0.5, px - pr * 0.25 + 4 * k, py - pr * 0.5 + 8 * k),
                fill="white",
            )
        # brows: inner end goes down when angry, up when sad
        inner_x, outer_x = ex - side * er * 0.9, ex + side * er * 1.1
        base = ey - er * 1.75
        d.line(
            (inner_x, base + e.brow * er * 0.55, outer_x, base - e.brow * er * 0.25 - er * 0.1),
            fill="#2a1d14",
            width=int(7 * k),
        )
        if glasses:
            d.ellipse(
                (ex - er * 1.45, ey - er * 1.35, ex + er * 1.45, ey + er * 1.35),
                outline="#3b3b3b",
                width=int(5 * k),
            )
    if glasses:
        d.line((fx - dx + er * 1.45, ey, fx + dx - er * 1.45, ey), fill="#3b3b3b", width=int(5 * k))

    my, mw = fy + 42 * k, 46 * k
    if mouth_open > 0.08:
        mh = 46 * k * mouth_open
        w = mw * (0.7 + 0.3 * mouth_open)
        lift = e.smile * min(8 * k, mh * 0.3)  # never flips the ellipse on a barely open mouth
        d.ellipse(
            (fx - w, my - mh * 0.45 - lift, fx + w, my + mh * 0.55),
            fill="#4a1414",
            outline="#2a0a0a",
            width=int(3 * k / face.s),
        )
        if mh > 18 * k:
            d.ellipse((fx - w * 0.5, my + mh * 0.1, fx + w * 0.5, my + mh * 0.5), fill="#e4626a")
    else:
        pts = [
            (fx + x * mw, my - e.smile * 14 * k * (x * x) + e.smile * 4 * k)
            for x in [i / 8 - 1 for i in range(17)]
        ]
        d.line(pts, fill="#3a1a12", width=int(6 * k), joint="curve")
    return im.resize(size, Image.LANCZOS)


# ------------------------------------------------------------------ scenes
def scene_of(text: str) -> str:
    return next((s for s, keys in SCENES.items() if _has(text, keys)), "kitchen")


def _background(scene: str, w: int, h: int, seed: int) -> Image.Image:
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    rnd = random.Random(seed)

    def grad(y0, y1, c0, c1):
        for y in range(y0, y1):
            t = (y - y0) / max(1, y1 - y0)
            d.line((0, y, w, y), fill=tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3)))

    if scene == "garden":
        grad(0, 1345, (22, 30, 74), (92, 66, 120))
        for _ in range(70):
            x, y, r = rnd.randint(0, w), rnd.randint(0, 900), rnd.choice((1, 2, 2, 3))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 250, 225))
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((700, 140, 980, 420), fill=(255, 240, 190, 120))
        im.paste(glow.filter(ImageFilter.GaussianBlur(60)), (0, 0), glow.filter(ImageFilter.GaussianBlur(60)))
        d.ellipse((760, 200, 920, 360), fill=(250, 240, 200))
        for x in range(-20, w, 95):  # fence
            d.polygon(
                [(x, 1360), (x, 1060), (x + 35, 1025), (x + 70, 1060), (x + 70, 1360)], fill=(52, 40, 50)
            )
        d.rectangle((0, 1140, w, 1170), fill=(45, 34, 44))
        grad(1330, h, (92, 58, 36), (60, 36, 22))
        for y in range(1420, h, 120):
            d.line((0, y, w, y - 30), fill=(70, 42, 26), width=10)
        for _ in range(16):
            x, y = rnd.randint(0, w), rnd.randint(1500, h - 40)
            d.polygon(_ellipse_poly(x, y, 14, 30, rnd.uniform(-0.6, 0.6), 24), fill=(70, 130, 60))
    else:  # kitchen
        grad(0, 1100, (250, 222, 186), (238, 192, 146))
        for y in range(1100, 1400, 70):  # tiles
            d.line((0, y, w, y), fill=(222, 214, 200), width=4)
        d.rectangle((0, 1100, w, 1400), fill=(245, 240, 230))
        for y in range(1100, 1400, 75):
            d.line((0, y, w, y), fill=(214, 205, 190), width=4)
        for x in range(0, w, 75):
            d.line((x, 1100, x, 1400), fill=(214, 205, 190), width=4)
        d.rectangle((90, 180, 470, 640), fill=(150, 200, 235), outline=(255, 255, 255), width=22)  # window
        d.line((280, 180, 280, 640), fill=(255, 255, 255), width=14)
        d.line((90, 410, 470, 410), fill=(255, 255, 255), width=14)
        d.rectangle((600, 420, 1040, 440), fill=(150, 100, 60))  # shelf + jars
        for i, col in enumerate(((220, 120, 60), (240, 200, 90), (130, 170, 90), (200, 80, 80))):
            x = 630 + i * 105
            d.rounded_rectangle((x, 300, x + 70, 420), radius=14, fill=col, outline=(120, 80, 50), width=4)
        d.rectangle((0, 1400, w, h), fill=(160, 104, 62))  # table
        d.rectangle((0, 1400, w, 1430), fill=(190, 132, 84))
        for _ in range(9):
            y = rnd.randint(1460, h - 20)
            d.line(
                (rnd.randint(0, w // 2), y, rnd.randint(w // 2, w), y + rnd.randint(-8, 8)),
                fill=(142, 90, 52),
                width=3,
            )
        d.ellipse((900, 1240, 1180, 1300), fill=(150, 40, 40))  # pot of borscht, half off-screen
        d.rectangle((900, 1270, 1180, 1440), fill=(185, 50, 48))
        d.ellipse((900, 1220, 1180, 1290), fill=(140, 30, 40))
        d.ellipse((930, 1235, 1150, 1280), fill=(170, 30, 50))
    return im.filter(ImageFilter.GaussianBlur(2.5))


# ------------------------------------------------------------------ animation
def _envelope(wav: Path, fps: int) -> list[float]:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        samples = array("h", w.readframes(w.getnframes()))
    step = sr / fps
    rms = []
    for i in range(int(len(samples) / step) + 1):
        chunk = samples[int(i * step) : int((i + 1) * step)]
        rms.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)) if chunk else 0.0)
    peak = sorted(rms)[int(len(rms) * 0.95)] if rms else 1.0
    out, cur = [], 0.0
    for v in rms:
        target = min(1.0, v / max(peak, 1.0))
        target = 0.0 if target < 0.12 else target
        cur = target if target > cur else cur * 0.55 + target * 0.45  # fast attack, slower release
        out.append(cur)
    return out


@dataclass
class Spoken:
    start: float
    end: float
    speaker: str
    delivery: str


def _layout(n: int) -> list[tuple[float, float]]:
    return {1: [(540, 1.55)], 2: [(300, 1.1), (790, 1.1)], 3: [(200, 0.9), (540, 0.9), (880, 0.9)]}.get(
        n, [(540, 1.0)]
    )


def _camera(camera: str, p: float) -> tuple[float, float]:
    return {
        "zoom_in": (1.0 + 0.12 * p, 0.0),
        "zoom_out": (1.12 - 0.12 * p, 0.0),
        "pan_left": (1.1, 40 - 80 * p),
        "pan_right": (1.1, -40 + 80 * p),
    }.get(camera, (1.03 + 0.03 * p, 0.0))


class ShotRenderer:
    def __init__(self, plan: ScriptPlan, shot: ShotPlan, scene_text: str, w: int, h: int, fps: int):
        self.w, self.h, self.fps, self.shot = w, h, fps, shot
        seed = int(hashlib.sha256(plan.title.encode()).hexdigest()[:8], 16)
        self.bg = _background(scene_of(scene_text), w, h, seed).convert("RGBA")
        chars = {c.name: c for c in plan.characters}
        names = [n for n in shot.on_screen if n in chars] or [
            a.speaker for a in shot.lines if a.speaker in chars
        ][:1]
        self.actors = []
        default_expr = {a.speaker: expr_of(a.delivery) for s in plan.shots for a in s.lines}
        for (x, scale), name in zip(_layout(len(names)), names, strict=False):
            c = chars[name]
            body, face = _draw_body(kind_of(c), accessories_of(c), scale, seed + len(name))
            self.actors.append(
                {
                    "name": name,
                    "x": x,
                    "scale": scale,
                    "body": body,
                    "face": face,
                    "glasses": "glasses" in accessories_of(c),
                    "rest": default_expr.get(name, NEUTRAL),
                    "phase": random.Random(name).uniform(0, 6.28),
                    "blinks": random.Random(name + "b"),
                }
            )
        for a in self.actors:
            rb, t, times = a["blinks"], 0.6, []
            while t < 60:
                times.append(t)
                t += rb.uniform(2.2, 4.2)
            a["blink_times"] = times

    def frame(
        self, t: float, dur: float, speaking: Spoken | None, level: float, caption: Image.Image | None
    ) -> Image.Image:
        im = self.bg.copy()
        for i, a in enumerate(self.actors):
            talking = speaking is not None and speaking.speaker == a["name"]
            e = expr_of(speaking.delivery) if talking else a["rest"]
            mouth = level if talking else 0.0
            blink = 0.1 if any(0 <= t - bt < 0.13 for bt in a["blink_times"]) else 1.0
            look = 0.0
            if len(self.actors) > 1:
                look = 1.0 if i == 0 else -1.0
            body, face = a["body"], a["face"]
            bob = math.sin(t * 2 * math.pi * 0.5 + a["phase"]) * 5 + mouth * 12
            bx = int(a["x"] - body.width / 2)
            by = int(GROUND_Y + 30 - body.height - bob)
            im.alpha_composite(body, (bx, by))
            f = _draw_face(body.size, face, a["scale"], e, mouth, blink, look, a["glasses"])
            im.alpha_composite(f, (bx, by))
        z, dx = _camera(self.shot.camera, t / max(dur, 0.01))
        cw, ch = self.w / z, self.h / z
        cx = min(max(self.w / 2 + dx, cw / 2), self.w - cw / 2)
        cy = min(max(self.h * 0.55, ch / 2), self.h - ch / 2)
        im = im.resize(
            (self.w, self.h), Image.BILINEAR, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        )
        if caption is not None:
            im.alpha_composite(caption)
        return im


def render_puppet_shot(
    plan: ScriptPlan,
    shot: ShotPlan,
    scene_text: str,
    audio: Path,
    spoken: list[Spoken],
    captions: list[Caption],
    duration: float,
    dst: Path,
    w: int,
    h: int,
    fps: int,
) -> None:
    r = ShotRenderer(plan, shot, scene_text, w, h, fps)
    env = _envelope(audio, fps)
    caps = [(Image.open(c.png).convert("RGBA"), c.start, c.end) for c in captions]
    frames = round(duration * fps)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-t",
        f"{duration:.3f}",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(frames):
            t = i / fps
            sp = next((s for s in spoken if s.start <= t < s.end), None)
            cap = next((im for im, a, b in caps if a <= t < b), None)
            level = env[i] if i < len(env) else 0.0
            proc.stdin.write(r.frame(t, duration, sp, level, cap).convert("RGB").tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise MediaError(f"ffmpeg (puppet) failed: {err[-1500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()


def render_still(plan: ScriptPlan, shot: ShotPlan, scene_text: str, dst: Path, w: int, h: int) -> None:
    ShotRenderer(plan, shot, scene_text, w, h, 30).frame(0.0, 1.0, None, 0.0, None).convert("RGB").save(dst)


def render_portrait(c: Character, dst: Path) -> None:
    body, face = _draw_body(kind_of(c), accessories_of(c), 1.0, 7)
    f = _draw_face(body.size, face, 1.0, NEUTRAL, 0.0, 1.0, 0.0, "glasses" in accessories_of(c))
    out = Image.new("RGBA", body.size, (236, 238, 242, 255))
    out.alpha_composite(body)
    out.alpha_composite(f)
    out.convert("RGB").save(dst)
