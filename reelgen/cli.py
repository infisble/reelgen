"""CLI: one command -> one video file.

python -m reelgen "Idea with a \"quoted line\" to be spoken."          # OpenAI
python -m reelgen --demo "..."                                       # no API key
python -m reelgen --resume 20261001-120000                           # continue a failed run
python -m reelgen --resume 20261001-120000 --from-stage images       # regenerate from a stage
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import STAGE_ORDER, InputError, run_pipeline
from .providers import build_providers
from .state import RunContext

EXIT = {"done": 0, "needs_review": 2, "failed": 1}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_once(
    idea: str | None,
    demo: bool,
    runs_dir: Path,
    resume: str | None = None,
    from_stage: str | None = None,
    run_id: str | None = None,
    options: dict | None = None,
) -> RunContext:
    if resume:
        ctx = RunContext.load(runs_dir / resume)
        demo = ctx.state.demo
    else:
        ctx = RunContext.create(runs_dir, idea or "", demo, run_id)
        ctx.state.options = {k: v for k, v in (options or {}).items() if v}
        ctx.save()
    opts = ctx.state.options
    settings = load_settings(demo, opts.get("renderer", "kenburns"))
    providers = build_providers(settings, ctx.state.counters, opts.get("plan"), opts.get("writer"))
    try:
        run_pipeline(ctx, settings, providers, from_stage)
    except InputError as e:
        print(f"Input rejected: {e}", file=sys.stderr)
    except Exception as e:  # already recorded in state.json; keep the CLI message short
        print(
            f"Run failed: {type(e).__name__}: {e}\nResume with: python -m reelgen --runs-dir "
            f"{runs_dir.as_posix()} --resume {ctx.state.run_id}",
            file=sys.stderr,
        )
    return ctx


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="reelgen", description="Idea -> vertical short video (8-30 s).")
    ap.add_argument(
        "idea", nargs="?", help='1-3 sentences; text in quotes ("..." or «...») is spoken verbatim'
    )
    ap.add_argument(
        "--demo", action="store_true", help="no API keys: template script, edge-tts, placeholders"
    )
    ap.add_argument(
        "--renderer",
        choices=["kenburns", "puppet"],
        default="kenburns",
        help="kenburns: generated stills + camera; puppet: 2D template characters animated in code",
    )
    ap.add_argument("--plan", help="script plan JSON written by an agent/human instead of the LLM step")
    ap.add_argument("--series", help="series bible JSON: fixed cast (name, appearance, voice) and style")
    ap.add_argument(
        "--writer",
        choices=["claude"],
        help="claude: script step via local Claude Code (claude -p), no API key",
    )
    ap.add_argument("--out", type=Path, help="copy the final video here")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--run-id", help="name for a new run (default: timestamp)")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue an existing run")
    ap.add_argument(
        "--from-stage", choices=STAGE_ORDER, help="with --resume: redo this stage and the following"
    )
    a = ap.parse_args(argv)
    if not a.idea and not a.resume:
        ap.error("give an idea or --resume RUN_ID")
    load_dotenv()

    opts = {"renderer": a.renderer, "plan": a.plan, "writer": a.writer, "series": a.series}
    ctx = run_once(a.idea, a.demo, a.runs_dir, a.resume, a.from_stage, a.run_id, opts)
    status = ctx.state.status
    final = ctx.dir / "final.mp4"
    if status in ("done", "needs_review") and final.exists():
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final, a.out)
            final = a.out
        qa = ctx.stage("qa").outputs
        print(f"\n{status.upper()}: {final}  ({qa['duration_s']}s)")
        for k, v in qa["checks"].items():
            print(f"  {k}: {v}")
        for w in qa["warnings"]:
            print(f"  WARNING: {w}")
        print(f"  report: {ctx.dir / 'report.json'}")
    return EXIT.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
