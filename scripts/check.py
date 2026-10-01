"""Single quality gate for humans, CI and coding agents: lint + format + tests.

Used as a Claude Code Stop hook (.claude/settings.json): exit code 2 blocks the agent from finishing
and feeds the failure output back to it, so "done" always means "checks are green".
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = [ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"]
PY = next((str(p) for p in VENV_PY if p.exists()), sys.executable)
STEPS = [
    [PY, "-m", "ruff", "check", "reelgen", "tests", "scripts"],
    [PY, "-m", "ruff", "format", "--check", "reelgen", "tests", "scripts"],
    [PY, "-m", "pytest", "-q", "-x"],
]


def main() -> int:
    hook_input = {}
    if not sys.stdin.isatty():
        try:
            hook_input = json.loads(sys.stdin.read() or "{}")
        except json.JSONDecodeError:
            pass
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for cmd in STEPS:
        name = " ".join(cmd[2:])
        r = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
        )
        if r.returncode != 0:
            out = (r.stdout + r.stderr)[-4000:]
            print(
                f"CHECK FAILED: {name}\n{out}\n"
                "Fix the cause (do not weaken tests), then re-run scripts/check.py.",
                file=sys.stderr,
            )
            # Second consecutive block from a Stop hook -> don't loop forever; report non-blocking.
            return 1 if hook_input.get("stop_hook_active") else 2
        print(f"ok: {name}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
