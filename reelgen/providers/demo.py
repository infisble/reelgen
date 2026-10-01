"""Keyless providers so the whole pipeline runs without paid APIs.

- TemplateWriter: deterministic plan (establishing shot + dialogue shot).
- EdgeTTS: free Microsoft Edge neural voices (needs internet, no key). Speaks the text as given.
- PlaceholderImages: procedural gradient frames (no generative model).
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from ..media import to_wav
from ..models import Character, LineAssignment, ParsedInput, ScriptPlan, ShotPlan

EDGE_VOICES = {
    "uk": {"female": "uk-UA-PolinaNeural", "male": "uk-UA-OstapNeural"},
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
}


class TemplateWriter:
    def write(self, parsed: ParsedInput, feedback: list[str], context: str = "") -> ScriptPlan:
        idea = parsed.idea.strip()
        for line in parsed.lines:  # keep dialogue out of image prompts (validator rule)
            idea = idea.replace(line, "")
        idea = re.sub(r'["«»“”„]', "", idea).strip() or "A cinematic scene"
        if parsed.lines_source == "whole_input":
            chars, speakers = [], ["narrator"] * len(parsed.lines)
        else:  # naive dialogue: alternate two voices
            chars = [
                Character(name="A", appearance="", voice="female_calm"),
                Character(name="B", appearance="", voice="male_calm"),
            ]
            speakers = ["A" if i % 2 == 0 else "B" for i in range(len(parsed.lines))]
        shots = [
            ShotPlan(
                shot_id=1,
                lines=[],
                on_screen=[],
                visual_prompt=f"Establishing wide shot. {idea}",
                camera="zoom_in",
            ),
            ShotPlan(
                shot_id=2,
                lines=[
                    LineAssignment(line_index=i, speaker=sp, delivery="") for i, sp in enumerate(speakers)
                ],
                on_screen=[],
                visual_prompt=f"Close-up of the main character, dramatic light. {idea}",
                camera="pan_right",
            ),
        ]
        return ScriptPlan(
            title=idea[:40], logline=idea, visual_style="cinematic", characters=chars, shots=shots
        )


VOICE_PITCH = {"female_bright": 18, "female_calm": 0, "male_calm": 0, "male_deep": -14, "narrator": 0}
# edge-tts has no emotion styles; prosody (rate, pitch, volume) is the lever we have
PROSODY = [
    (("angry", "furious", "shout", "yell", "betray", "лют", "крич"), 10, 6, 15),
    (("sad", "tear", "cry", "guilt", "sigh", "whisper", "ashamed", "сум", "винн"), -14, -8, -12),
    (("laugh", "tease", "happy", "joy", "playful", "смі", "рад"), 8, 12, 5),
    (("proud", "grump", "stubborn", "boast", "бурч"), -4, -4, 5),
]


def _prosody(voice: str, delivery: str) -> tuple[int, int, int]:
    d = delivery.lower()
    rate, pitch, volume = next(
        ((r, p, v) for keys, r, p, v in PROSODY if any(k in d for k in keys)), (0, 0, 0)
    )
    return rate, pitch + VOICE_PITCH.get(voice, 0), volume


class EdgeTTS:
    def __init__(self, counters: dict):
        self.k = counters

    def synthesize(
        self, text: str, voice: str, delivery: str, lang: str, out_wav: Path, attempt: int
    ) -> None:
        # edge-tts has no emotion control: `delivery` is ignored in demo mode
        import edge_tts

        gender = "male" if voice.startswith("male") else "female"
        name = EDGE_VOICES.get(lang, EDGE_VOICES["en"])[gender]
        rate, pitch, volume = _prosody(voice, delivery)
        rate -= [0, 8, 15][min(attempt, 2)]
        mp3 = out_wav.with_suffix(".mp3")
        comm = edge_tts.Communicate(
            text, name, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{volume:+d}%"
        )
        asyncio.run(comm.save(str(mp3)))
        self.k["tts_calls"] = self.k.get("tts_calls", 0) + 1
        to_wav(mp3, out_wav)
        mp3.unlink(missing_ok=True)


class PlaceholderImages:
    def generate(self, prompt: str, out_png: Path, attempt: int, refs: list[Path] | None = None) -> None:
        seed = int(hashlib.sha256(f"{prompt}|{attempt}".encode()).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        w, h = 1024, 1536
        top = tuple(rnd.randint(20, 120) for _ in range(3))
        bottom = tuple(rnd.randint(120, 240) for _ in range(3))
        column = Image.new("RGB", (1, h))
        for y in range(h):
            t = y / h
            column.putpixel((0, y), tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)))
        img = column.resize((w, h))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for _ in range(9):
            r = rnd.randint(60, 260)
            x, y = rnd.randint(0, w), rnd.randint(0, h)
            col = tuple(rnd.randint(150, 255) for _ in range(3)) + (rnd.randint(40, 110),)
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)
        layer = layer.filter(ImageFilter.GaussianBlur(30))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        img.save(out_png)


class FilePlanWriter:
    """The script step done outside the pipeline (by a coding agent or a human) and handed in as JSON.

    The plan still goes through the same validator as an LLM plan, so it cannot touch the line text.
    """

    def __init__(self, path: Path):
        self.path = path

    def write(self, parsed: ParsedInput, feedback: list[str], context: str = "") -> ScriptPlan:
        if feedback:
            raise ValueError(f"plan file {self.path} is invalid: {feedback}")
        return ScriptPlan.model_validate_json(self.path.read_text(encoding="utf-8"))
