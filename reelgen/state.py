"""Run directory + persistent state (state.json) + event log (events.jsonl).

A run is a folder; every stage writes its artifacts there and records them in state.json.
Re-running with the same run id skips stages that are already `done` (resume after a crash or
after a manual fix), and `--from-stage X` invalidates X and everything after it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .models import RunState, StageState


class RunContext:
    def __init__(self, run_dir: Path, state: RunState):
        self.dir = run_dir
        self.state = state

    # ----- persistence -----
    @classmethod
    def create(cls, runs_root: Path, idea: str, demo: bool, run_id: str | None = None) -> RunContext:
        now = datetime.now(UTC)
        run_id = run_id or now.strftime("%Y%m%d-%H%M%S")
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = cls(run_dir, RunState(run_id=run_id, idea=idea, demo=demo, created_at=now.isoformat()))
        ctx.save()
        return ctx

    @classmethod
    def load(cls, run_dir: Path) -> RunContext:
        state = RunState.model_validate_json((run_dir / "state.json").read_text(encoding="utf-8"))
        return cls(run_dir, state)

    def save(self) -> None:
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        # atomic: a crash never leaves half-written state. On Windows the target can be briefly
        # locked by antivirus / file watchers (WinError 5) - retry a few times before giving up.
        for i in range(10):
            try:
                os.replace(tmp, self.dir / "state.json")
                return
            except PermissionError:
                if i == 9:
                    raise
                time.sleep(0.05 * (i + 1))

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, rel: str, data) -> Path:
        p = self.path(rel)
        if hasattr(data, "model_dump_json"):
            p.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def read_json(self, rel: str):
        return json.loads((self.dir / rel).read_text(encoding="utf-8"))

    # ----- observability -----
    def log(self, stage: str, event: str, **data) -> None:
        rec = {"ts": round(time.time(), 3), "stage": stage, "event": event, **data}
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra = " ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list)))
        print(f"[{stage}] {event} {extra}".rstrip(), file=sys.stderr, flush=True)

    def stage(self, name: str) -> StageState:
        return self.state.stages.setdefault(name, StageState())

    def bump(self, key: str, n: int = 1) -> None:
        self.state.counters[key] = self.state.counters.get(key, 0) + n
