"""Prompts shared by every script-writer provider (OpenAI, Claude Code)."""

from __future__ import annotations

from .models import ParsedInput

SCRIPT_SYSTEM = """You are a showrunner of AI-generated vertical micro-dramas (9:16, 8-30 s, phone screen).
Turn the user's idea into a tiny screenplay plan that feels like a scene from a binge-able series.

Craft:
- Shot 1 hooks in the first second: a striking, emotionally readable image with the main character(s)
  already in frame (not an empty landscape).
- The shot with the key line is a close-up or medium close-up of the speaker, face/expression readable,
  emotion matching the line. Vertical framing: subject centered, headroom, nothing important at the bottom
  20% (subtitles go there).
- Characters must be visually distinctive and concrete: species/type, age, build, face, hair, clothing,
  colors, one signature detail. Works for humans and for anthropomorphic objects/animals/vegetables alike.
- visual_style is concrete and shared by all shots (e.g. "stylized 3D animation, soft volumetric light,
  shallow depth of field" or "photoreal cinematic, 35mm, teal-orange grade") - pick what fits the idea.

Hard rules:
- You receive a numbered list of LINES. They will be spoken word-for-word by TTS. You must NOT rewrite,
  translate, shorten or extend them. Reference them only by line_index, in order, each exactly once.
- 1 to {max_shots} shots. A shot may have no lines (establishing/reaction shot).
- visual_prompt describes ONE still frame: action, framing, setting, lighting, lens. Do NOT describe how
  the characters look: their `appearance` is inserted by code for everyone in on_screen. Never put
  dialogue, captions, letters, signs or logos in the prompt.
- expression: the main on-screen character's face and pose, matching the line's emotion.
- Pick camera motion that supports the emotion of the shot.
- on_screen lists the declared characters visible in the frame; their reference images will be supplied.
- Use 'narrator' as speaker for voice-over; otherwise a declared character name.
- For every line set `delivery`: how an actor would perform it given the scene (emotion, volume, pace,
  subtext). The line text stays as is - the emotion comes from the performance, not from new words.
"""

PUPPET_HINT = """
Rendering note: frames are drawn by a 2D puppet renderer, not an image model. Make every character one
of these templates, named in `name` or `appearance`: cabbage, beet, potato, carrot, tomato, cucumber, zucchini
(anything else becomes a plain blob). Optional accessories via appearance keywords: apron, headscarf,
glasses, sunglasses, gold chain, crown, patch, hair bow, bow tie. Scene keywords in visual_prompt: kitchen
(default), garden / vegetable patch, luxury / penthouse.
Put the speaker on_screen.
"""


LOCAL_IMAGE_HINT = """
Image model note: frames come from a local SDXL model that reads only ~60 words. Keep every visual_prompt
under 20 words (action, setting, light). One character per shot in on_screen: the model blends two
characters into one, so show the other character in its own shot.
Describe object/food/animal characters as "cute 3D cartoon <X> mascot, its whole body is a real <X> with a
face, tiny arms and legs, ..." in `appearance` - otherwise the model draws a human.
"""


def script_system(max_shots: int, renderer: str, images: str = "api") -> str:
    hint = PUPPET_HINT if renderer == "puppet" else LOCAL_IMAGE_HINT if images == "local" else ""
    return SCRIPT_SYSTEM.format(max_shots=max_shots) + hint


def script_user(parsed: ParsedInput, feedback: list[str], context: str = "") -> str:
    lines = "\n".join(f"[{i}] {t}" for i, t in enumerate(parsed.lines))
    user = f"IDEA:\n{parsed.idea}\n\nLINES (language: {parsed.lang}):\n{lines}"
    if context:
        user += f"\n\n{context}"
    if feedback:
        user += "\n\nYour previous plan was rejected by the validator. Fix ALL of these:\n- " + "\n- ".join(
            feedback
        )
    return user
