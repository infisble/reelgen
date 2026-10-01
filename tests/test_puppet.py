"""Puppet renderer: template picking, expressions, and a full pipeline run with a hand-written plan."""

import json

import pytest

from reelgen import puppet
from reelgen.config import load_settings
from reelgen.media import probe
from reelgen.models import Character
from reelgen.pipeline import run_pipeline
from reelgen.providers import Providers
from reelgen.providers.demo import FilePlanWriter, PlaceholderImages
from reelgen.state import RunContext
from tests.test_pipeline import EchoSTT, ToneTTS

IDEA = "Мама-капуста на кухні. Вона кричить: «Ти зрадив мене з олів'є!» Буряк зітхає: «Мамо, це лише салат.»"
PLAN = {
    "title": "t",
    "logline": "l",
    "visual_style": "2D cartoon",
    "characters": [
        {"name": "Мама-Капуста", "appearance": "cabbage mom, apron, headscarf", "voice": "female_calm"},
        {"name": "Буряк", "appearance": "young beet", "voice": "male_calm"},
    ],
    "shots": [
        {
            "shot_id": 1,
            "lines": [{"line_index": 0, "speaker": "Мама-Капуста", "delivery": "furious"}],
            "on_screen": ["Мама-Капуста", "Буряк"],
            "visual_prompt": "Kitchen, cabbage mom turns to the beet son",
            "camera": "zoom_in",
        },
        {
            "shot_id": 2,
            "lines": [{"line_index": 1, "speaker": "Буряк", "delivery": "guilty, sighing"}],
            "on_screen": ["Буряк"],
            "visual_prompt": "Close-up of the beet son, ashamed",
            "camera": "static",
        },
    ],
}


@pytest.mark.parametrize(
    "name,appearance,kind",
    [
        ("Мама-Капуста", "", "cabbage"),
        ("Bob", "young beet", "beet"),
        ("Дід", "old potato", "potato"),
        ("Морквина", "", "carrot"),
        ("X", "a sad cloud", "blob"),
        ("Огірок", "cabbage template styled as a cucumber", "cucumber"),
    ],
)
def test_kind_from_name_or_appearance(name, appearance, kind):
    assert puppet.kind_of(Character(name=name, appearance=appearance, voice="narrator")) == kind


def test_expression_follows_delivery():
    assert puppet.expr_of("furious, betrayed").brow > 0
    assert puppet.expr_of("guilty, quiet").brow < 0
    assert puppet.expr_of("laughing").smile > 0.5
    assert puppet.expr_of("") == puppet.NEUTRAL


def test_puppet_pipeline_with_plan_file(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(PLAN, ensure_ascii=False), encoding="utf-8")
    ctx = RunContext.create(tmp_path, IDEA, demo=True, run_id="p")
    tts = ToneTTS()
    p = Providers(
        FilePlanWriter(plan_file),
        tts,
        EchoSTT(tts, full_text="Ти зрадив мене з олів'є! Мамо, це лише салат."),
        PlaceholderImages(),
        None,
        ctx.state.counters,
    )
    assert run_pipeline(ctx, load_settings(demo=True, renderer="puppet"), p) == "done"
    assert tts.said == ["Ти зрадив мене з олів'є!", "Мамо, це лише салат."]
    assert (ctx.dir / "cast/char_0.png").exists() and (ctx.dir / "images/shot_2.png").exists()
    pr = probe(ctx.dir / "final.mp4")
    assert 8 <= pr.duration <= 30.25 and (pr.width, pr.height) == (1080, 1920) and pr.has_audio


def test_scene_and_accessory_keywords_do_not_misfire():
    assert puppet.scene_of("kitchen studio, 50mm lens, shallow depth of field") == "kitchen"
    assert puppet.scene_of("night vegetable garden") == "garden"
    anchor = Character(name="Помідор", appearance="news anchor with a bow tie", voice="male_calm")
    assert puppet.accessories_of(anchor) == {"bowtie"}
