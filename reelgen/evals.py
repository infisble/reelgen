"""Regression eval: run the pipeline over a fixed idea set, aggregate metrics, compare two versions.

python -m reelgen.evals run --label v1 [--demo] [--only uk_cafe,en_lighthouse]
python -m reelgen.evals compare evals/results/v1.json evals/results/v2.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .cli import load_dotenv, run_once

IDEAS = Path("evals/ideas.jsonl")
RESULTS = Path("evals/results")


def _case_metrics(ctx, wall_s: float) -> dict:
    st = ctx.state
    qa = st.stages.get("qa")
    qa_out = qa.outputs if qa and qa.status == "done" else {}
    voice = st.stages.get("voice")
    lines = voice.outputs.get("lines", []) if voice else []
    failed_stage = next((n for n, s in st.stages.items() if s.status == "failed"), None)
    return {
        "status": st.status,
        "failed_stage": failed_stage,
        "wall_s": round(wall_s, 1),
        "video_s": qa_out.get("duration_s"),
        "verbatim_e2e": qa_out.get("checks", {}).get("verbatim_e2e"),
        "lines_verified_first_try": sum(1 for v in lines if v["status"] == "verified" and v["attempts"] == 1),
        "lines_total": len(lines),
        "script_repairs": st.counters.get("script_repairs", 0),
        "tts_retries": st.counters.get("tts_retries", 0),
        "image_retries": st.counters.get("image_retries", 0),
        "counters": st.counters,
        "run_dir": ctx.dir.as_posix(),
    }


def _summary(cases: dict[str, dict]) -> dict:
    vals = list(cases.values())
    n = len(vals) or 1
    walls = [c["wall_s"] for c in vals if c["status"] != "failed"]
    verb = [c["verbatim_e2e"] for c in vals if isinstance(c["verbatim_e2e"], bool)]
    lines = sum(c["lines_total"] for c in vals) or 1
    return {
        "cases": len(vals),
        "success_rate": round(sum(c["status"] == "done" for c in vals) / n, 3),
        "needs_review_rate": round(sum(c["status"] == "needs_review" for c in vals) / n, 3),
        "failure_rate": round(sum(c["status"] not in ("done", "needs_review") for c in vals) / n, 3),
        "verbatim_pass_rate": round(sum(verb) / len(verb), 3) if verb else None,
        "tts_first_try_rate": round(sum(c["lines_verified_first_try"] for c in vals) / lines, 3),
        "median_wall_s": round(statistics.median(walls), 1) if walls else None,
        "retries_per_run": round(
            sum(c["tts_retries"] + c["image_retries"] + c["script_repairs"] for c in vals) / n, 2
        ),
    }


def cmd_run(a) -> int:
    load_dotenv()
    only = set(a.only.split(",")) if a.only else None
    ideas = [json.loads(x) for x in IDEAS.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = {}
    for item in ideas:
        if only and item["id"] not in only:
            continue
        print(f"\n=== {item['id']} ===", file=sys.stderr)
        t0 = time.monotonic()
        ctx = run_once(item["idea"], a.demo, Path("runs") / f"eval-{a.label}", run_id=item["id"])
        cases[item["id"]] = _case_metrics(ctx, time.monotonic() - t0)
    result = {"label": a.label, "demo": a.demo, "summary": _summary(cases), "cases": cases}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{a.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved {out}")
    return 0


def cmd_compare(a) -> int:
    ra, rb = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b))
    print(f"{'metric':<22}{ra['label']:>12}{rb['label']:>12}")
    for k, va in ra["summary"].items():
        vb = rb["summary"].get(k)
        print(f"{k:<22}{va!s:>12}{vb!s:>12}")
    print("\nper case (status / video_s / verbatim):")
    for cid in sorted(set(ra["cases"]) | set(rb["cases"])):
        ca, cb = ra["cases"].get(cid, {}), rb["cases"].get(cid, {})

        def fmt(c):
            return f"{c.get('status')}/{c.get('video_s')}/{c.get('verbatim_e2e')}" if c else "-"

        flag = "  <-- changed" if fmt(ca) != fmt(cb) else ""
        print(f"  {cid:<26}{fmt(ca):>34}  {fmt(cb):>34}{flag}")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="reelgen.evals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--demo", action="store_true")
    r.add_argument("--only")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
