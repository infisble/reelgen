"""OpenAI-backed providers: structured script, TTS, STT verification, images, vision judge."""

from __future__ import annotations

import base64
import os
import urllib.request
from pathlib import Path

from ..config import Settings
from ..media import to_wav
from ..models import ImageReview, ParsedInput, ScriptPlan
from ..prompts import script_system, script_user
from . import Providers

VOICES = {
    "female_calm": "sage",
    "female_bright": "coral",
    "male_calm": "ash",
    "male_deep": "onyx",
    "narrator": "alloy",
}

TTS_INSTRUCTIONS = (
    "You are a voice actor in a vertical drama. Performance: {delivery}. "
    "Say every word exactly as written - do not add, drop or change words."
)

JUDGE_PROMPT = """You are a strict QA reviewer for frames of a vertical AI video.
Rate how well the image matches the prompt (1-5) and flag defects: visible text/watermarks,
anatomy defects (extra fingers, broken faces). Prompt:
{prompt}"""


class OpenAIWriter:
    def __init__(self, client, settings: Settings, counters: dict):
        self.c, self.s, self.k = client, settings, counters

    def write(self, parsed: ParsedInput, feedback: list[str]) -> ScriptPlan:
        resp = self.c.responses.parse(
            model=self.s.llm_model,
            instructions=script_system(self.s.max_shots, self.s.renderer),
            input=script_user(parsed, feedback),
            text_format=ScriptPlan,
        )
        _bump(self.k, "llm_calls")
        if resp.usage:
            _bump(self.k, "llm_tokens", resp.usage.total_tokens)
        if resp.output_parsed is None:
            raise ValueError("LLM returned no parsable ScriptPlan (refusal or truncation)")
        return resp.output_parsed


class OpenAITTS:
    def __init__(self, client, settings: Settings, counters: dict):
        self.c, self.s, self.k = client, settings, counters

    def synthesize(
        self, text: str, voice: str, delivery: str, lang: str, out_wav: Path, attempt: int
    ) -> None:
        # Retry strategy: slow down a bit on each attempt - mispronunciation is the usual failure.
        speed = [1.0, 0.92, 0.85][min(attempt, 2)]
        raw = out_wav.with_suffix(".raw.wav")
        with self.c.audio.speech.with_streaming_response.create(
            model=self.s.tts_model,
            voice=VOICES.get(voice, "alloy"),
            input=text,
            instructions=TTS_INSTRUCTIONS.format(delivery=delivery or "natural, matching the meaning"),
            response_format="wav",
            speed=speed,
        ) as r:
            r.stream_to_file(raw)
        _bump(self.k, "tts_calls")
        to_wav(raw, out_wav)
        raw.unlink(missing_ok=True)


class OpenAISTT:
    def __init__(self, client, settings: Settings, counters: dict):
        self.c, self.s, self.k = client, settings, counters

    def transcribe(self, wav: Path, lang: str) -> str:
        # Deliberately no `prompt` with the expected text: it would bias STT toward a false "match".
        with open(wav, "rb") as f:
            r = self.c.audio.transcriptions.create(model=self.s.stt_model, file=f, language=lang)
        _bump(self.k, "stt_calls")
        return r if isinstance(r, str) else r.text


class OpenAIImages:
    def __init__(self, client, settings: Settings, counters: dict):
        self.c, self.s, self.k = client, settings, counters

    def generate(self, prompt: str, out_png: Path, attempt: int, refs: list[Path] | None = None) -> None:
        common = dict(model=self.s.image_model, prompt=prompt, size="1024x1536", quality=self.s.image_quality)
        if refs:  # identity lock: the same character sheet goes into every shot
            files = [open(p, "rb") for p in refs]
            try:
                r = self.c.images.edit(image=files, **common)
            finally:
                for f in files:
                    f.close()
        else:
            r = self.c.images.generate(n=1, **common)
        _bump(self.k, "image_calls")
        d = r.data[0]
        if d.b64_json:
            out_png.write_bytes(base64.b64decode(d.b64_json))
        else:
            with urllib.request.urlopen(d.url, timeout=60) as resp:  # noqa: S310 - URL from the API
                out_png.write_bytes(resp.read())


class OpenAIJudge:
    def __init__(self, client, settings: Settings, counters: dict):
        self.c, self.s, self.k = client, settings, counters

    def review(self, png: Path, prompt: str) -> ImageReview:
        b64 = base64.b64encode(png.read_bytes()).decode()
        resp = self.c.responses.parse(
            model=self.s.llm_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": JUDGE_PROMPT.format(prompt=prompt)},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                    ],
                }
            ],
            text_format=ImageReview,
        )
        _bump(self.k, "judge_calls")
        if resp.output_parsed is None:
            raise ValueError("judge returned no parsable review")
        return resp.output_parsed


def _bump(counters: dict, key: str, n: int = 1) -> None:
    counters[key] = counters.get(key, 0) + n


def make_openai(settings: Settings, counters: dict) -> Providers:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Put it in .env / env, or run with --demo.")
    from openai import OpenAI

    # SDK handles transient errors (429/5xx/timeouts) with exponential backoff;
    # our own retry loops in the pipeline are for *quality* failures.
    client = OpenAI(max_retries=4, timeout=180)
    return Providers(
        writer=OpenAIWriter(client, settings, counters),
        tts=OpenAITTS(client, settings, counters),
        stt=OpenAISTT(client, settings, counters),
        images=OpenAIImages(client, settings, counters),
        judge=OpenAIJudge(client, settings, counters),
        counters=counters,
    )
