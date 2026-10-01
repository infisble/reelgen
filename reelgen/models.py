"""Data contracts between stages. Every stage reads/writes JSON that matches these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Voice = Literal["female_calm", "female_bright", "male_calm", "male_deep", "narrator"]
Camera = Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]


# ---------- stage: parse ----------
class ParsedInput(BaseModel):
    idea: str
    lines: list[str]  # verbatim spoken lines, extracted by code
    lines_source: Literal["quotes", "whole_input"]
    lang: str
    est_speech_s: float


# ---------- stage: script (LLM structured output) ----------
class Character(BaseModel):
    name: str
    appearance: str = Field(description="Concise visual description reused in every shot for consistency")
    voice: Voice


class LineAssignment(BaseModel):
    line_index: int = Field(description="Index into the provided list of lines. Never rewrite the line text.")
    speaker: str = Field(description="Character name, or 'narrator' for voice-over")
    delivery: str = Field(
        description="How the line is performed, from the scene context: emotion, volume, pace, subtext. "
        "E.g. 'exhausted, almost whispering, holding back tears'. English, one short phrase."
    )


class ShotPlan(BaseModel):
    shot_id: int
    lines: list[LineAssignment] = Field(description="Lines spoken during this shot, in order; may be empty")
    on_screen: list[str] = Field(
        description="Names of declared characters visible in this frame (their reference images are used)"
    )
    visual_prompt: str = Field(
        description="Image prompt for a vertical 9:16 frame. No dialogue text, no captions, no letters."
    )
    camera: Camera


class ScriptPlan(BaseModel):
    title: str
    logline: str
    visual_style: str = Field(description="Shared style string appended to every visual prompt")
    characters: list[Character]
    shots: list[ShotPlan]


# ---------- quality control ----------
class ImageReview(BaseModel):
    score: int = Field(description="1 = unusable, 5 = excellent match to the prompt")
    has_text_or_watermark: bool
    has_anatomy_defects: bool
    issues: list[str]


# ---------- run state ----------
StageStatus = Literal["pending", "running", "done", "failed"]


class StageState(BaseModel):
    status: StageStatus = "pending"
    attempts: int = 0
    duration_s: float | None = None
    error: str | None = None
    outputs: dict = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str
    idea: str
    demo: bool
    created_at: str
    status: Literal["running", "done", "needs_review", "failed"] = "running"
    options: dict = Field(default_factory=dict)  # renderer, plan file - reused on --resume
    stages: dict[str, StageState] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)  # provider calls, tokens, retries
