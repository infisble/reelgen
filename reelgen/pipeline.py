"""The agent pipeline: a fixed DAG of stages with persisted state, quality gates and targeted retries.

parse -> script -> cast -> voice -> images -> render -> assemble -> qa

Each stage: reads previous artifacts from the run dir, writes its own, returns a small `outputs`
dict that goes into state.json. Expensive per-item artifacts (voice lines, shot images) are cached
by content hash, so re-running a stage only regenerates the items whose inputs changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from . import puppet
from .config import Settings
from .media import (
    Caption,
    build_timeline_wav,
    concat,
    extract_audio,
    probe,
    render_caption,
    render_shot,
    wav_duration,
)
from .models import ParsedInput, ScriptPlan, StageState
from .providers import Providers
from .script_rules import validate_plan
from .state import RunContext
from .verbatim import compare, detect_lang, estimate_speech_s, extract_lines

STAGE_ORDER = ["parse", "script", "cast", "voice", "images", "render", "assemble", "qa"]


class InputError(ValueError):
    """Bad user input - not retryable."""


class StageFailed(RuntimeError):
    """A stage exhausted its retries."""


def _h(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cached(ctx: RunContext, meta_rel: str, key: str, *files: str) -> dict | None:
    meta = ctx.dir / meta_rel
    if not meta.exists():
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    if data.get("key") == key and all((ctx.dir / f).exists() for f in files):
        return data
    return None


# ------------------------------------------------------------------ stages
def stage_parse(ctx: RunContext, s: Settings, p: Providers) -> dict:
    idea = ctx.state.idea.strip()
    if not idea:
        raise InputError("Empty idea")
    lines, source = extract_lines(idea), "quotes"
    if not lines:  # no quoted dialogue -> the whole input is narrated verbatim
        lines, source = [idea], "whole_input"
    est = sum(estimate_speech_s(t) for t in lines)
    budget = s.max_video_s - 2.0 - (s.lead_s + s.tail_s) * len(lines)
    if est > budget:
        raise InputError(f"Spoken text needs ~{est:.0f}s, does not fit into a {s.max_video_s:.0f}s video")
    parsed = ParsedInput(
        idea=idea,
        lines=lines,
        lines_source=source,
        lang=detect_lang(" ".join(lines)),
        est_speech_s=round(est, 1),
    )
    ctx.write_json("input.json", parsed)
    return {
        "lines": len(lines),
        "lines_source": source,
        "lang": parsed.lang,
        "est_speech_s": parsed.est_speech_s,
    }


def stage_script(ctx: RunContext, s: Settings, p: Providers) -> dict:
    parsed = ParsedInput.model_validate(ctx.read_json("input.json"))
    feedback: list[str] = []
    for attempt in range(1, s.script_attempts + 1):
        try:
            plan = p.writer.write(parsed, feedback)
            errors = validate_plan(plan, parsed, s.max_shots)
        except ValueError as e:  # refusal / unparsable output
            plan, errors = None, [str(e)]
        ctx.log("script", "attempt", n=attempt, ok=not errors, errors=errors)
        if not errors:
            break
        feedback = errors
        ctx.bump("script_repairs")
    else:
        raise StageFailed(f"Script plan still invalid after {s.script_attempts} attempts: {feedback}")

    ctx.write_json("script.json", plan)
    # Human-readable screenplay. Line text is inserted by code from input.json, never by the LLM.
    screenplay = [
        {
            "shot": sh.shot_id,
            "camera": sh.camera,
            "visual": sh.visual_prompt,
            "dialogue": [
                {"speaker": a.speaker, "delivery": a.delivery, "text": parsed.lines[a.line_index]}
                for a in sh.lines
            ],
        }
        for sh in plan.shots
    ]
    ctx.write_json("screenplay.json", {"title": plan.title, "logline": plan.logline, "shots": screenplay})
    return {"shots": len(plan.shots), "attempts": attempt, "title": plan.title}


def stage_voice(ctx: RunContext, s: Settings, p: Providers) -> dict:
    parsed = ParsedInput.model_validate(ctx.read_json("input.json"))
    plan = ScriptPlan.model_validate(ctx.read_json("script.json"))
    speaker_of = {a.line_index: a.speaker for sh in plan.shots for a in sh.lines}
    voice_of = {c.name: c.voice for c in plan.characters} | {"narrator": "narrator"}
    delivery_of = {a.line_index: a.delivery for sh in plan.shots for a in sh.lines}
    tts_id = type(p.tts).__name__ + (s.tts_model if not s.demo else "")

    summary = []
    for i, text in enumerate(parsed.lines):
        voice = voice_of.get(speaker_of[i], "narrator")
        delivery = delivery_of[i]
        key = _h(text, voice, delivery, parsed.lang, tts_id)
        wav_rel, meta_rel = f"voice/line_{i}.wav", f"voice/line_{i}.json"
        if (meta := _cached(ctx, meta_rel, key, wav_rel)) is not None:
            ctx.log("voice", "cached", line=i, status=meta["status"])
            summary.append(meta)
            continue

        wav, status, check, attempt = ctx.path(wav_rel), "unverified", None, 0
        for attempt in range(1, s.tts_attempts + 1):
            p.tts.synthesize(text, voice, delivery, parsed.lang, wav, attempt - 1)
            if p.stt is None:
                break
            check = compare(text, p.stt.transcribe(wav, parsed.lang))
            ctx.log(
                "voice",
                "verify",
                line=i,
                attempt=attempt,
                ok=check.ok,
                wer=round(check.wer, 3),
                heard=check.heard,
            )
            if check.ok:
                status = "verified"
                break
            status = "mismatch"
            ctx.bump("tts_retries")
        meta = {
            "key": key,
            "line": i,
            "text": text,
            "voice": voice,
            "status": status,
            "attempts": attempt,
            "duration_s": round(wav_duration(wav), 3),
            "check": check.to_dict() if check else None,
        }
        ctx.write_json(meta_rel, meta)
        summary.append(meta)

    return {"lines": [{k: m[k] for k in ("line", "status", "attempts", "duration_s")} for m in summary]}


def stage_cast(ctx: RunContext, s: Settings, p: Providers) -> dict:
    """Character reference sheets. Every shot with the character uses this image -> same face/look."""
    plan = ScriptPlan.model_validate(ctx.read_json("script.json"))
    img_id = type(p.images).__name__ + (s.image_model if not s.demo else "")
    cast = {}
    for i, c in enumerate(plan.characters):
        if not c.appearance.strip():
            continue
        prompt = (
            f"Character reference for an animated/filmed series: {c.appearance}. Style: {plan.visual_style}. "
            "Single character, head-to-knees, facing camera 3/4, neutral expression, plain light-grey "
            "studio background, even lighting. No text, no letters, no watermark."
        )
        key = _h(prompt, img_id, s.renderer)
        png_rel, meta_rel = f"cast/char_{i}.png", f"cast/char_{i}.json"
        if _cached(ctx, meta_rel, key, png_rel) is None:
            if s.renderer == "puppet":
                puppet.render_portrait(c, ctx.path(png_rel))
            else:
                p.images.generate(prompt, ctx.path(png_rel), 0)
            ctx.write_json(meta_rel, {"key": key, "name": c.name, "prompt": prompt})
            ctx.log("cast", "reference", character=c.name)
        cast[c.name] = png_rel
    return {"cast": cast}


def stage_images(ctx: RunContext, s: Settings, p: Providers) -> dict:
    plan = ScriptPlan.model_validate(ctx.read_json("script.json"))
    if s.renderer == "puppet":  # key frames for preview/report; the animation itself happens in render
        for sh in plan.shots:
            png = ctx.path(f"images/shot_{sh.shot_id}.png")
            puppet.render_still(plan, sh, _scene_text(ctx, plan, sh), png, s.width, s.height)
        return {"shots": [{"shot": sh.shot_id, "attempts": 1, "accepted": True} for sh in plan.shots]}
    img_id = type(p.images).__name__ + (s.image_model if not s.demo else "")
    out = []
    for sh in plan.shots:
        prompt = (
            f"{sh.visual_prompt} Style: {plan.visual_style}. Vertical 9:16 cinematic frame. "
            "No text, no captions, no letters, no watermark."
        )
        cast = ctx.stage("cast").outputs.get("cast", {})
        refs = [ctx.dir / cast[n] for n in sh.on_screen if n in cast]
        if refs:
            prompt += (
                " Keep the characters exactly as in the reference images (face, body, clothing, colors)."
            )
        ref_hashes = [hashlib.sha256(r.read_bytes()).hexdigest()[:12] for r in refs]
        key = _h(prompt, img_id, *ref_hashes)
        png_rel, meta_rel = f"images/shot_{sh.shot_id}.png", f"images/shot_{sh.shot_id}.json"
        if (meta := _cached(ctx, meta_rel, key, png_rel)) is not None:
            ctx.log("images", "cached", shot=sh.shot_id)
            out.append(meta)
            continue

        best: tuple[int, Path, dict | None] | None = None
        accepted, attempt = False, 0
        for attempt in range(1, s.image_attempts + 1):
            cand = ctx.path(f"images/shot_{sh.shot_id}_try{attempt}.png")
            p.images.generate(prompt, cand, attempt - 1, refs)
            if p.judge is None:
                best, accepted = (0, cand, None), True
                break
            try:
                r = p.judge.review(cand, prompt)
            except Exception as e:  # judge is advisory: its failure must not block the run
                ctx.log("images", "judge_error", shot=sh.shot_id, error=str(e))
                best, accepted = (0, cand, None), True
                break
            ok = r.score >= s.min_image_score and not r.has_text_or_watermark and not r.has_anatomy_defects
            ctx.log(
                "images", "review", shot=sh.shot_id, attempt=attempt, score=r.score, ok=ok, issues=r.issues
            )
            if best is None or r.score > best[0]:
                best = (r.score, cand, r.model_dump())
            if ok:
                accepted = True
                break
            ctx.bump("image_retries")
        assert best is not None
        shutil.copyfile(best[1], ctx.dir / png_rel)
        meta = {
            "key": key,
            "shot": sh.shot_id,
            "prompt": prompt,
            "attempts": attempt,
            "accepted": accepted,
            "review": best[2],
        }
        ctx.write_json(meta_rel, meta)
        out.append(meta)
    return {"shots": [{k: m[k] for k in ("shot", "attempts", "accepted")} for m in out]}


def _scene_text(ctx: RunContext, plan: ScriptPlan, sh) -> str:
    return f"{sh.visual_prompt} {plan.visual_style} {ctx.state.idea}"


def stage_render(ctx: RunContext, s: Settings, p: Providers) -> dict:
    parsed = ParsedInput.model_validate(ctx.read_json("input.json"))
    plan = ScriptPlan.model_validate(ctx.read_json("script.json"))

    # 1) build per-shot audio timelines (wav paths and silences)
    timelines: list[list[Path | float]] = []
    for sh in plan.shots:
        if not sh.lines:
            timelines.append([s.silent_shot_s])
            continue
        tl: list[Path | float] = [s.lead_s]
        for j, a in enumerate(sh.lines):
            tl.append(ctx.dir / f"voice/line_{a.line_index}.wav")
            tl.append(s.gap_s if j < len(sh.lines) - 1 else s.tail_s)
        timelines.append(tl)

    def dur(tl):
        return sum(x if isinstance(x, float) else wav_duration(x) for x in tl)

    total = sum(dur(tl) for tl in timelines)
    if total < s.min_video_s:  # pad the ending so the video is >= 8 s
        timelines[-1].append(s.min_video_s - total + 0.3)
        total = sum(dur(tl) for tl in timelines)
    if total > s.max_video_s:
        raise StageFailed(f"Timeline is {total:.1f}s > {s.max_video_s}s (TTS slower than estimated)")

    # 2) render each shot: Ken Burns over the frame + captions synced to the line spans
    clips = []
    for sh, tl in zip(plan.shots, timelines, strict=True):
        wav = ctx.path(f"render/shot_{sh.shot_id}.wav")
        spans = build_timeline_wav(tl, wav)
        caps = []
        for a, (st, en) in zip(sh.lines, spans, strict=True):
            png = ctx.path(f"render/cap_{a.line_index}.png")
            render_caption(parsed.lines[a.line_index], s.width, s.height, png)
            caps.append(Caption(png, st, en))
        clip = ctx.path(f"render/shot_{sh.shot_id}.mp4")
        if s.renderer == "puppet":
            spoken = [
                puppet.Spoken(st, en, a.speaker, a.delivery)
                for a, (st, en) in zip(sh.lines, spans, strict=True)
            ]
            puppet.render_puppet_shot(
                plan,
                sh,
                _scene_text(ctx, plan, sh),
                wav,
                spoken,
                caps,
                wav_duration(wav),
                clip,
                s.width,
                s.height,
                s.fps,
            )
            clips.append(clip)
            ctx.log(
                "render", "shot", shot=sh.shot_id, duration=round(wav_duration(wav), 2), renderer="puppet"
            )
            continue
        render_shot(
            ctx.dir / f"images/shot_{sh.shot_id}.png",
            wav,
            caps,
            sh.camera,
            wav_duration(wav),
            clip,
            s.width,
            s.height,
            s.fps,
        )
        clips.append(clip)
        ctx.log("render", "shot", shot=sh.shot_id, duration=round(wav_duration(wav), 2))
    return {
        "clips": [c.relative_to(ctx.dir).as_posix() for c in clips],
        "planned_duration_s": round(total, 2),
    }


def stage_assemble(ctx: RunContext, s: Settings, p: Providers) -> dict:
    clips = [ctx.dir / c for c in ctx.stage("render").outputs["clips"]]
    final = ctx.path("final.mp4")
    concat(clips, final)
    return {"final": "final.mp4"}


def stage_qa(ctx: RunContext, s: Settings, p: Providers) -> dict:
    parsed = ParsedInput.model_validate(ctx.read_json("input.json"))
    final = ctx.dir / "final.mp4"
    pr = probe(final)
    checks: dict[str, bool | str] = {
        "duration_8_30s": s.min_video_s <= pr.duration <= s.max_video_s + 0.25,
        "vertical_1080x1920": (pr.width, pr.height) == (s.width, s.height),
        "has_audio": pr.has_audio,
        "audio_not_silent": pr.mean_volume_db is not None and pr.mean_volume_db > -45,
    }
    heard = None
    if p.stt is not None:  # end-to-end: what is actually audible in the final file
        wav = ctx.path("qa/final_audio.wav")
        extract_audio(final, wav)
        heard = p.stt.transcribe(wav, parsed.lang)
        checks["verbatim_e2e"] = compare(" ".join(parsed.lines), heard).ok
    else:
        checks["verbatim_e2e"] = "unverified (demo mode has no STT)"

    voice = ctx.stage("voice").outputs.get("lines", [])
    images = ctx.stage("images").outputs.get("shots", [])
    warnings = [
        f"line {v['line']}: TTS/STT mismatch after {v['attempts']} attempts"
        for v in voice
        if v["status"] == "mismatch"
    ]
    warnings += [f"shot {i['shot']}: image below quality threshold" for i in images if not i["accepted"]]
    passed = all(v is True or isinstance(v, str) for v in checks.values()) and not warnings
    report = {
        "run_id": ctx.state.run_id,
        "passed": passed,
        "checks": checks,
        "warnings": warnings,
        "probe": pr.__dict__,
        "heard": heard,
        "lines": parsed.lines,
        "counters": ctx.state.counters,
        "stage_durations_s": {k: v.duration_s for k, v in ctx.state.stages.items()},
    }
    ctx.write_json("report.json", report)
    return {"passed": passed, "checks": checks, "warnings": warnings, "duration_s": round(pr.duration, 2)}


STAGES: dict[str, Callable[[RunContext, Settings, Providers], dict]] = {
    "parse": stage_parse,
    "script": stage_script,
    "cast": stage_cast,
    "voice": stage_voice,
    "images": stage_images,
    "render": stage_render,
    "assemble": stage_assemble,
    "qa": stage_qa,
}


# ------------------------------------------------------------------ runner
def run_pipeline(ctx: RunContext, s: Settings, p: Providers, from_stage: str | None = None) -> str:
    if from_stage:
        for name in STAGE_ORDER[STAGE_ORDER.index(from_stage) :]:
            ctx.state.stages[name] = StageState()
    ctx.state.status = "running"
    ctx.save()
    for name in STAGE_ORDER:
        st = ctx.stage(name)
        if st.status == "done":
            ctx.log(name, "skip (done in a previous run)")
            continue
        st.status, st.error = "running", None
        st.attempts += 1
        ctx.save()
        ctx.log(name, "start")
        t0 = time.monotonic()
        try:
            st.outputs = STAGES[name](ctx, s, p)
        except Exception as e:
            st.status, st.error = "failed", f"{type(e).__name__}: {e}"
            st.duration_s = round(time.monotonic() - t0, 2)
            ctx.state.status = "failed"
            ctx.save()
            ctx.log(name, "failed", error=st.error)
            raise
        st.status, st.duration_s = "done", round(time.monotonic() - t0, 2)
        ctx.save()
        ctx.log(name, "done", seconds=st.duration_s)
    ctx.state.status = "done" if ctx.stage("qa").outputs["passed"] else "needs_review"
    ctx.save()
    return ctx.state.status
