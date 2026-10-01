# CLAUDE.md — rules for coding agents in this repo

This repo is a pipeline `idea -> script -> cast -> voice -> images -> render -> assemble -> qa` producing a
vertical 8–30 s video. You (the agent) are expected to take a task to "done" without a human.
"Done" has one definition: `python scripts/check.py` exits 0 (it also runs as a Stop hook).

## Invariants (never break; tests enforce them)
1. **Spoken text is never produced by an LLM.** Lines come from `verbatim.extract_lines` (quoted
   text) or the whole input. LLMs reference lines by `line_index` only. Do not add a field that
   lets a model emit dialogue text, and do not pass line text through any model except TTS/STT.
2. **Every stage is resumable.** A stage reads its inputs from the run dir and writes artifacts
   there; outputs in `state.json` must be JSON-serializable and small (paths, counts, flags).
3. **Per-item artifacts are content-hash cached** (`_cached` + `key`). If you add an input that
   changes an artifact, include it in the key.
4. **Quality gates are code, not vibes**: script → `validate_plan`; voice → STT + `compare` (WER must
   be 0); images → vision judge (advisory, must not block on judge errors); final → `stage_qa`.
5. **No paid API calls in tests.** Use fakes (see `tests/test_pipeline.py`) or `httpx.MockTransport`
   (see `tests/test_openai_providers.py`).

## How to work
- Contracts live in `reelgen/models.py`. Change the model first, then the stage, then tests.
- New stage = function `stage_x(ctx, settings, providers) -> dict`, register in `STAGES` and
  `STAGE_ORDER`, add an e2e assertion in `tests/test_pipeline.py`.
- New provider = class implementing a Protocol from `reelgen/providers/__init__.py` + a mocked test.
- A behaviour change to prompts/models must be evaluated: `python -m reelgen.evals run --label <new>`
  then `python -m reelgen.evals compare evals/results/<old>.json evals/results/<new>.json`.
  Do not merge a change that lowers `success_rate` or `verbatim_pass_rate`.
- Do not weaken a failing test to make it pass; fix the cause or explain why the test was wrong.
- Keep diffs small; one concern per change. Line length 110 (ruff).

## Commands
- `python scripts/check.py` — lint + format + tests (the gate)
- `python -m reelgen --demo "idea"` — keyless run; `python -m reelgen "idea"` — OpenAI run
- `python -m reelgen --resume <run_id> [--from-stage images]` — resume / partial regeneration
