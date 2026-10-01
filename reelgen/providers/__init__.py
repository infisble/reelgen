"""Provider interfaces + factory. Real = OpenAI; demo = keyless fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..models import ImageReview, ParsedInput, ScriptPlan


class ScriptWriter(Protocol):
    def write(self, parsed: ParsedInput, feedback: list[str]) -> ScriptPlan: ...


class TTS(Protocol):
    def synthesize(
        self, text: str, voice: str, delivery: str, lang: str, out_wav: Path, attempt: int
    ) -> None: ...


class STT(Protocol):
    def transcribe(self, wav: Path, lang: str) -> str: ...


class ImageGen(Protocol):
    def generate(self, prompt: str, out_png: Path, attempt: int, refs: list[Path] | None = None) -> None:
        """`refs`: character reference images to keep identity consistent across shots."""
        ...


class ImageJudge(Protocol):
    def review(self, png: Path, prompt: str) -> ImageReview: ...


@dataclass
class Providers:
    writer: ScriptWriter
    tts: TTS
    stt: STT | None  # None -> verbatim check is reported as "unverified"
    images: ImageGen
    judge: ImageJudge | None
    counters: dict


def build_providers(
    settings: Settings, counters: dict[str, int], plan_file: str | None = None, writer: str | None = None
) -> Providers:
    """`counters` is shared with RunState so provider call counts land in state.json.

    Script step: `plan_file` (hand-written plan) > `writer="claude"` (Claude Code) > mode default
    (OpenAI LLM, or the deterministic template in demo mode).
    """
    if settings.demo:
        from .demo import EdgeTTS, PlaceholderImages, TemplateWriter

        providers = Providers(TemplateWriter(), EdgeTTS(counters), None, PlaceholderImages(), None, counters)
    else:
        from .openai_impl import make_openai

        providers = make_openai(settings, counters)

    if plan_file:
        from .demo import FilePlanWriter

        providers.writer = FilePlanWriter(Path(plan_file))
    elif writer == "claude":
        from .claude_code import ClaudeCodeWriter

        providers.writer = ClaudeCodeWriter(settings, counters)
    return providers
