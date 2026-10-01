"""Runtime settings. Everything model-specific is overridable via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    demo: bool
    images: str = "api"  # api: provider model (OpenAI) | local: open SDXL model on the local GPU
    renderer: str = "kenburns"  # kenburns: generated stills + camera | puppet: 2D template characters
    # OpenAI models (only used when demo=False)
    llm_model: str = _env("REELGEN_LLM_MODEL", "gpt-5-mini")
    tts_model: str = _env("REELGEN_TTS_MODEL", "gpt-4o-mini-tts")
    stt_model: str = _env("REELGEN_STT_MODEL", "gpt-transcribe")
    image_model: str = _env("REELGEN_IMAGE_MODEL", "gpt-image-2.5-flare")
    image_quality: str = _env("REELGEN_IMAGE_QUALITY", "medium")

    # Pipeline limits / quality gates
    min_video_s: float = 8.0
    max_video_s: float = 30.0
    max_shots: int = 4
    script_attempts: int = 3  # LLM repair loop
    tts_attempts: int = 3  # re-synthesize until STT matches the line verbatim
    image_attempts: int = 3  # regenerate if the vision judge rejects the frame
    min_image_score: int = 3  # judge score 1..5

    # Timeline
    lead_s: float = 0.35  # silence before the first line in a shot
    gap_s: float = 0.35  # silence between lines in one shot
    tail_s: float = 0.6  # silence after the last line in a shot
    silent_shot_s: float = 2.5  # duration of a shot without dialogue
    fps: int = 30
    width: int = 1080
    height: int = 1920


def load_settings(demo: bool, renderer: str = "kenburns", images: str = "api") -> Settings:
    return Settings(demo=demo, renderer=renderer, images=images)
