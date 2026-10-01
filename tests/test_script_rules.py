from reelgen.models import Character, LineAssignment, ParsedInput, ScriptPlan, ShotPlan
from reelgen.script_rules import validate_plan

PARSED = ParsedInput(
    idea="x",
    lines=["I will never come back here again", "Fine."],
    lines_source="quotes",
    lang="en",
    est_speech_s=3,
)


def plan(**over) -> ScriptPlan:
    base = dict(
        title="t",
        logline="l",
        visual_style="noir",
        characters=[Character(name="Ann", appearance="red coat", voice="female_calm")],
        shots=[
            ShotPlan(
                shot_id=1,
                lines=[LineAssignment(line_index=0, speaker="Ann", delivery="cold")],
                on_screen=["Ann"],
                visual_prompt="Ann in a red coat at a rainy bus stop, night, neon",
                camera="zoom_in",
            ),
            ShotPlan(
                shot_id=2,
                lines=[LineAssignment(line_index=1, speaker="narrator", delivery="flat")],
                on_screen=[],
                visual_prompt="Empty bus stop after the bus leaves, wet asphalt",
                camera="static",
            ),
        ],
    )
    base.update(over)
    return ScriptPlan(**base)


def test_valid_plan():
    assert validate_plan(plan(), PARSED, max_shots=3) == []


def test_line_order_and_coverage_enforced():
    p = plan()
    p.shots[0].lines, p.shots[1].lines = p.shots[1].lines, p.shots[0].lines
    assert any("exactly once" in e for e in validate_plan(p, PARSED, 3))


def test_unknown_speaker():
    p = plan()
    p.shots[0].lines[0].speaker = "Bob"
    assert any("Bob" in e for e in validate_plan(p, PARSED, 3))


def test_dialogue_in_visual_prompt_rejected():
    p = plan()
    p.shots[0].visual_prompt = "A sign that reads I will never come back here again"
    assert any("contains dialogue" in e for e in validate_plan(p, PARSED, 3))


def test_too_many_shots():
    p = plan()
    assert any("1..1 shots" in e for e in validate_plan(p, PARSED, 1))


def test_on_screen_must_be_declared():
    p = plan()
    p.shots[0].on_screen = ["Ghost"]
    assert any("Ghost" in e for e in validate_plan(p, PARSED, 3))
