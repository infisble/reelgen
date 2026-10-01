"""Script step via Claude Code in headless mode (`claude -p --json-schema`).

Uses the local Claude Code login, so the whole pipeline runs with one command and no API keys:
Claude writes the plan, the puppet renderer draws it, edge-tts voices it. Tools are disabled - the
model only returns JSON, which then goes through the same validator as any other plan.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..config import Settings
from ..models import ParsedInput, ScriptPlan
from ..prompts import script_system, script_user


def find_claude() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise SystemExit("Claude Code CLI not found. Install it or use --writer openai/template.")
    # On Windows `claude` is a .cmd shim; call the real binary so JSON args don't go through cmd.exe quoting.
    native = Path(exe).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    return str(native) if exe.lower().endswith((".cmd", ".bat")) and native.exists() else exe


class ClaudeCodeWriter:
    def __init__(self, settings: Settings, counters: dict):
        self.s, self.k = settings, counters
        self.exe = find_claude()
        self.model = os.environ.get("REELGEN_CLAUDE_MODEL", "sonnet")

    def write(self, parsed: ParsedInput, feedback: list[str], context: str = "") -> ScriptPlan:
        prompt = (
            script_system(self.s.max_shots, self.s.renderer, self.s.images)
            + "\n\n"
            + script_user(parsed, feedback, context)
        )
        schema = json.dumps(ScriptPlan.model_json_schema(), ensure_ascii=True)
        cmd = [
            self.exe,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--tools",
            "",
            "--no-session-persistence",
            "--model",
            self.model,
        ]
        r = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
        )
        self.k["llm_calls"] = self.k.get("llm_calls", 0) + 1
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(f"claude -p returned non-JSON (exit {r.returncode}): {r.stderr[-500:]}") from e
        if out.get("is_error") or "structured_output" not in out:
            raise ValueError(f"claude -p failed: {str(out.get('result'))[:500]}")
        cost = out.get("total_cost_usd")
        if cost:
            self.k["llm_cost_millicents"] = self.k.get("llm_cost_millicents", 0) + round(cost * 100000)
        return ScriptPlan.model_validate(out["structured_output"])


JUDGE_PROMPT = """You are a strict QA reviewer of frames for a vertical AI cartoon series.
Open the image file {path} with the Read tool and compare it with the prompt below.
Score 1-5 how well it matches (character, expression, setting). Flag text/watermarks and anatomy defects:
extra or missing eyes, extra limbs or fingers, duplicated faces, merged characters.
Prompt: {prompt}"""


class ClaudeCodeJudge:
    """Vision QA via Claude Code: reads the frame with the Read tool, returns an ImageReview."""

    def __init__(self, counters: dict):
        self.k, self.exe = counters, find_claude()
        self.model = os.environ.get("REELGEN_CLAUDE_JUDGE_MODEL", "sonnet")

    def review(self, png: Path, prompt: str):
        from ..models import ImageReview

        path = png.resolve()
        cmd = [
            self.exe,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ImageReview.model_json_schema()),
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--add-dir",
            str(path.parent),
            "--no-session-persistence",
            "--model",
            self.model,
        ]
        r = subprocess.run(
            cmd,
            input=JUDGE_PROMPT.format(path=path, prompt=prompt),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        self.k["judge_calls"] = self.k.get("judge_calls", 0) + 1
        out = json.loads(r.stdout)
        if out.get("is_error") or "structured_output" not in out:
            raise ValueError(f"claude judge failed: {str(out.get('result'))[:300]}")
        return ImageReview.model_validate(out["structured_output"])
