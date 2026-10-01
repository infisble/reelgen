"""End-to-end pipeline with fake providers (offline, deterministic, real ffmpeg)."""

import json
import math
import struct
import wave

import pytest

from reelgen.config import load_settings
from reelgen.media import probe
from reelgen.models import Character, ImageReview
from reelgen.pipeline import InputError, run_pipeline
from reelgen.providers import Providers
from reelgen.providers.demo import PlaceholderImages, TemplateWriter
from reelgen.state import RunContext

IDEA = 'A lighthouse keeper sees a ship in the storm. He whispers: "Not tonight, old friend, not tonight."'


class ToneTTS:
    """Writes a 440 Hz tone, ~0.4 s per word. Remembers what it was asked to say."""

    def __init__(self):
        self.said: list[str] = []

    def synthesize(self, text, voice, delivery, lang, out_wav, attempt):
        self.said.append(text)
        n = int(24000 * 0.4 * len(text.split()))
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(
                b"".join(
                    struct.pack("<h", int(8000 * math.sin(i / 24000 * 2 * math.pi * 440))) for i in range(n)
                )
            )


class EchoSTT:
    """'Hears' whatever TTS said last; optionally garbles the first N transcriptions."""

    def __init__(self, tts: ToneTTS, garble_first: int = 0, full_text: str = ""):
        self.tts, self.garble, self.full = tts, garble_first, full_text

    def transcribe(self, wav, lang):
        if self.garble > 0:
            self.garble -= 1
            return "something else entirely"
        return self.full if "final_audio" in str(wav) else self.tts.said[-1]


class StrictJudge:
    def __init__(self, reject_first: int = 0):
        self.reject = reject_first

    def review(self, png, prompt):
        if self.reject > 0:
            self.reject -= 1
            return ImageReview(
                score=2, has_text_or_watermark=True, has_anatomy_defects=False, issues=["text"]
            )
        return ImageReview(score=5, has_text_or_watermark=False, has_anatomy_defects=False, issues=[])


def make(tmp_path, garble=0, reject=0, idea=IDEA):
    ctx = RunContext.create(tmp_path, idea, demo=True, run_id="t")
    tts = ToneTTS()
    stt = EchoSTT(tts, garble, full_text="Not tonight, old friend, not tonight.")
    p = Providers(TemplateWriter(), tts, stt, PlaceholderImages(), StrictJudge(reject), ctx.state.counters)
    return ctx, p, tts


def test_end_to_end_produces_valid_video(tmp_path):
    ctx, p, tts = make(tmp_path)
    status = run_pipeline(ctx, load_settings(demo=True), p)
    assert status == "done"
    pr = probe(ctx.dir / "final.mp4")
    assert 8 <= pr.duration <= 30.25 and (pr.width, pr.height) == (1080, 1920) and pr.has_audio
    # the TTS received exactly the quoted line - nothing paraphrased it on the way
    assert tts.said == ["Not tonight, old friend, not tonight."]
    assert ctx.stage("qa").outputs["checks"]["verbatim_e2e"] is True


def test_stt_mismatch_triggers_tts_retry(tmp_path):
    ctx, p, tts = make(tmp_path, garble=1)
    run_pipeline(ctx, load_settings(demo=True), p)
    line = ctx.stage("voice").outputs["lines"][0]
    assert line["status"] == "verified" and line["attempts"] == 2
    assert ctx.state.counters["tts_retries"] == 1


def test_persistent_mismatch_goes_to_human_review(tmp_path):
    ctx, p, _ = make(tmp_path, garble=99)
    assert run_pipeline(ctx, load_settings(demo=True), p) == "needs_review"
    assert ctx.stage("qa").outputs["warnings"]


def test_rejected_image_is_regenerated(tmp_path):
    ctx, p, _ = make(tmp_path, reject=1)
    run_pipeline(ctx, load_settings(demo=True), p)
    shots = ctx.stage("images").outputs["shots"]
    assert shots[0]["attempts"] == 2 and shots[0]["accepted"]


def test_resume_skips_done_stages_and_partial_regen_uses_cache(tmp_path):
    ctx, p, tts = make(tmp_path)
    s = load_settings(demo=True)
    run_pipeline(ctx, s, p)
    ctx2 = RunContext.load(ctx.dir)
    run_pipeline(ctx2, s, p)  # nothing to do
    assert len(tts.said) == 1
    run_pipeline(ctx2, s, p, from_stage="voice")  # stage re-run, but line unchanged -> cache hit
    assert len(tts.said) == 1 and ctx2.state.status == "done"


def test_too_long_input_rejected(tmp_path):
    long_line = " ".join(["word"] * 120)
    ctx, p, _ = make(tmp_path, idea=f'He said: "{long_line}"')
    with pytest.raises(InputError):
        run_pipeline(ctx, load_settings(demo=True), p)
    assert ctx.state.status == "failed"


class CastWriter(TemplateWriter):
    """Template plan + one real character who is on screen in the dialogue shot."""

    def write(self, parsed, feedback, context=""):
        plan = super().write(parsed, feedback, context)
        plan.characters = [
            Character(name="A", appearance="old keeper, grey beard, yellow raincoat", voice="male_deep")
        ]
        plan.shots[1].on_screen = ["A"]
        for a in plan.shots[1].lines:
            a.speaker = "A"
        return plan


class RecordingImages(PlaceholderImages):
    def __init__(self):
        self.calls = []

    def generate(self, prompt, out_png, attempt, refs=None):
        self.calls.append((prompt, [r.name for r in refs or []]))
        super().generate(prompt, out_png, attempt, refs)


def test_character_reference_is_used_in_shots_with_that_character(tmp_path):
    ctx, p, _ = make(tmp_path)
    p.writer, p.images = CastWriter(), RecordingImages()
    assert run_pipeline(ctx, load_settings(demo=True), p) == "done"
    assert (ctx.dir / "cast/char_0.png").exists()
    cast_call, shot1, shot2 = p.images.calls
    assert "yellow raincoat" in cast_call[0] and cast_call[1] == []
    assert shot1[1] == []  # establishing shot: nobody on screen
    assert shot2[1] == ["char_0.png"] and "reference images" in shot2[0]


def test_series_bible_overrides_invented_looks(tmp_path):
    bible = {
        "visual_style": "glossy 3D",
        "characters": [{"name": "A", "appearance": "zucchini with a patch", "voice": "male_calm"}],
    }
    bible_file = tmp_path / "bible.json"
    bible_file.write_text(json.dumps(bible), encoding="utf-8")
    ctx, p, _ = make(tmp_path)
    ctx.state.options = {"series": str(bible_file)}
    p.writer = CastWriter()  # invents "old keeper, grey beard, yellow raincoat" for A
    run_pipeline(ctx, load_settings(demo=True), p)
    plan = json.loads((ctx.dir / "script.json").read_text(encoding="utf-8"))
    assert plan["characters"][0]["appearance"] == "zucchini with a patch"
    assert plan["characters"][0]["voice"] == "male_calm" and plan["visual_style"] == "glossy 3D"


class TieJudge:
    """Every frame fails with the same score; only the first one also has an anatomy defect."""

    def __init__(self):
        self.n = 0

    def review(self, png, prompt):
        self.n += 1
        return ImageReview(score=2, has_text_or_watermark=False, has_anatomy_defects=self.n == 1, issues=[])


def test_fallback_prefers_frame_without_defects(tmp_path):
    ctx, p, _ = make(tmp_path)
    p.judge = TieJudge()
    run_pipeline(ctx, load_settings(demo=True), p)
    meta = json.loads((ctx.dir / "images/shot_1.json").read_text(encoding="utf-8"))
    assert meta["accepted"] is False and meta["review"]["has_anatomy_defects"] is False
