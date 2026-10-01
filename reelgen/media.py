"""ffmpeg helpers (bundled binary via imageio-ffmpeg - no system install needed)."""

from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 24000


class MediaError(RuntimeError):
    pass


def _ffmpeg(*args: str) -> str:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise MediaError(f"ffmpeg failed ({p.returncode}): {' '.join(args)[:300]}\n{p.stderr[-1500:]}")
    return p.stderr


# ---------- audio ----------
def to_wav(src: Path, dst: Path) -> None:
    """Normalize any audio to mono 24 kHz s16 WAV so segments can be concatenated in Python."""
    _ffmpeg("-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def build_timeline_wav(segments: list[Path | float], dst: Path) -> list[tuple[float, float]]:
    """Concatenate WAV files and silences (floats, seconds). Returns (start, end) of every WAV segment."""
    frames, spans, t = [], [], 0.0
    for seg in segments:
        if isinstance(seg, (int, float)):
            n = int(seg * SAMPLE_RATE)
            frames.append(b"\x00\x00" * n)
            t += n / SAMPLE_RATE
            continue
        with wave.open(str(seg), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise MediaError(f"{seg} is not mono/24kHz/s16 - run to_wav first")
            data = w.readframes(w.getnframes())
        dur = len(data) / 2 / SAMPLE_RATE
        frames.append(data)
        spans.append((t, t + dur))
        t += dur
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(b"".join(frames))
    return spans


# ---------- captions ----------
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int):
    for f in _FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def render_caption(text: str, width: int, height: int, dst: Path) -> None:
    """Transparent full-frame PNG with the line as a subtitle in the lower third."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(int(width * 0.055))
    max_w = int(width * 0.84)
    words, lines, cur = text.split(), [], ""
    for wd in words:
        cand = f"{cur} {wd}".strip()
        if d.textlength(cand, font=font) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    line_h = int(font.size * 1.3)
    block_h = line_h * len(lines)
    y0 = int(height * 0.83) - block_h // 2  # lower band: below faces, above the phone UI
    pad = int(width * 0.03)
    widest = max(d.textlength(ln, font=font) for ln in lines)
    d.rounded_rectangle(
        ((width - widest) / 2 - pad, y0 - pad, (width + widest) / 2 + pad, y0 + block_h + pad // 2),
        radius=pad,
        fill=(0, 0, 0, 150),
    )
    for i, ln in enumerate(lines):
        x = (width - d.textlength(ln, font=font)) / 2
        d.text((x, y0 + i * line_h), ln, font=font, fill=(255, 255, 255, 255))
    img.save(dst)


# ---------- video ----------
def _zoompan(camera: str, frames: int, w: int, h: int, fps: int) -> str:
    n = max(frames - 1, 1)
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    z, x, y = {
        "zoom_in": (f"1+0.15*on/{n}", cx, cy),
        "zoom_out": (f"1.15-0.15*on/{n}", cx, cy),
        "pan_left": ("1.12", f"(iw-iw/zoom)*(1-on/{n})", cy),
        "pan_right": ("1.12", f"(iw-iw/zoom)*on/{n}", cy),
        "static": (f"1+0.03*on/{n}", cx, cy),
    }[camera]
    # Upscale 2x before zoompan: avoids the well-known zoompan sub-pixel jitter.
    return (
        f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"
    )


@dataclass
class Caption:
    png: Path
    start: float
    end: float


def render_shot(
    image: Path,
    audio: Path,
    captions: list[Caption],
    camera: str,
    duration: float,
    dst: Path,
    w: int,
    h: int,
    fps: int,
) -> None:
    frames = round(duration * fps)
    inputs = ["-i", str(image), "-i", str(audio)]
    graph = [f"[0:v]{_zoompan(camera, frames, w, h, fps)}[v0]"]
    for i, c in enumerate(captions):
        inputs += ["-i", str(c.png)]
        graph.append(f"[v{i}][{i + 2}:v]overlay=0:0:enable='between(t,{c.start:.3f},{c.end:.3f})'[v{i + 1}]")
    graph.append(f"[v{len(captions)}]format=yuv420p[vout]")
    _ffmpeg(
        *inputs,
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[vout]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-t",
        f"{duration:.3f}",
        str(dst),
    )


def concat(clips: list[Path], dst: Path) -> None:
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(dst))
    lst.unlink(missing_ok=True)


# ---------- probing ----------
@dataclass
class Probe:
    duration: float
    width: int
    height: int
    has_audio: bool
    mean_volume_db: float | None


def probe(path: Path) -> Probe:
    err = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stderr
    dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    vm = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", err)
    mv = re.search(r"mean_volume: (-?[\d.]+) dB", err)
    if not dm:
        raise MediaError(f"cannot probe {path}:\n{err[-800:]}")
    hh, mm, ss = dm.groups()
    return Probe(
        duration=int(hh) * 3600 + int(mm) * 60 + float(ss),
        width=int(vm.group(1)) if vm else 0,
        height=int(vm.group(2)) if vm else 0,
        has_audio=bool(re.search(r"Stream #.*Audio:", err)),
        mean_volume_db=float(mv.group(1)) if mv else None,
    )


def extract_audio(video: Path, dst_wav: Path) -> None:
    _ffmpeg("-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(dst_wav))
