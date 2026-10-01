"""Code-side validation of the LLM's script plan. Errors are fed back to the LLM (repair loop)."""

from __future__ import annotations

from .models import ParsedInput, ScriptPlan
from .verbatim import normalize_words


def validate_plan(plan: ScriptPlan, parsed: ParsedInput, max_shots: int) -> list[str]:
    errors: list[str] = []
    if not 1 <= len(plan.shots) <= max_shots:
        errors.append(f"Need 1..{max_shots} shots, got {len(plan.shots)}.")

    used = [a.line_index for s in plan.shots for a in s.lines]
    expected = list(range(len(parsed.lines)))
    if used != expected:
        errors.append(
            f"Every line index {expected} must be used exactly once, in order across shots; got {used}."
        )

    names = {c.name for c in plan.characters} | {"narrator"}
    for s in plan.shots:
        for a in s.lines:
            if a.speaker not in names:
                errors.append(f"Shot {s.shot_id}: speaker '{a.speaker}' is not a declared character.")
        for n in s.on_screen:
            if n not in names - {"narrator"}:
                errors.append(f"Shot {s.shot_id}: on_screen '{n}' is not a declared character.")
        if len(s.visual_prompt.strip()) < 20:
            errors.append(f"Shot {s.shot_id}: visual_prompt is too short to be useful.")
        prompt_words = " ".join(normalize_words(s.visual_prompt))
        for i, line in enumerate(parsed.lines):
            lw = " ".join(normalize_words(line))
            if len(lw.split()) >= 3 and lw in prompt_words:
                errors.append(
                    f"Shot {s.shot_id}: visual_prompt contains dialogue line {i}; image models render it as "
                    "garbled text. Describe the scene instead."
                )
    if [s.shot_id for s in plan.shots] != list(range(1, len(plan.shots) + 1)):
        errors.append("shot_id values must be 1..N in order.")
    return errors
