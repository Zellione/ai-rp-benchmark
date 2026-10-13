# RPBench — initial runnable scaffold (v0.1)

A **minimal smoke-test runner**, not yet a validated RP quality benchmark.
It preserves every request and response so human reviewers can inspect a run.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Smoke test (no model/server needed)

```bash
rpbench --scenario scenarios/suspicious-archivist.json --config configs/mock.json
find runs -type f
```

## Try with llama.cpp

Start llama-server with the relevant model and port, then run:

```bash
rpbench --scenario scenarios/suspicious-archivist.json --config configs/llamacpp-chat.example.json
```

## Try with KoboldCpp native text completion

Start KoboldCpp on port 5001, then:

```bash
rpbench --scenario scenarios/suspicious-archivist.json --config configs/koboldcpp-native.example.json
```

**The demo prompt format is deliberately generic, not a validated template for your model.**
For meaningful tests, configure the real instruct/roleplay template and stop sequences.
Never mix this smoke test with scored benchmark results.

## Run files

Each execution receives its own folder under `runs/`:

- `manifest.json`: configuration, scenario, file hashes, status.
- `conversation.md`: entire readable conversation, appended and flushed after each generated turn.
- `events.jsonl`: request payloads (including exact submitted raw completion prompt or chat messages), response payloads, timing and errors, appended and fsynced per event.

Runs are retained even if generation fails midway. Files can contain sensitive content and should not be published without review. Set appropriate filesystem permissions. No hidden access tokens are intentionally written to logs.

## Current adapter status

- **Implemented basic generation routes**: llama.cpp native/OpenAI, KoboldCpp native/OpenAI, vLLM OpenAI; generic OpenAI-compatible HTTP parsing.
- **Planned only**: dedicated ninfer and Strata adapters. Placeholder names are explicitly guarded unless the user verifies an OpenAI-compatible server themselves.
- **Not yet implemented**: capabilities negotiation, tokenization and true context measurement, engine version capture, automatic template rendering, context-window truncation detection, resume, generation branching, judge, rubric, plots, deterministic fixtures and statistical uncertainty.

A generic JSON API does **not** imply all samplers are supported. This MVP forwards fields as configured; unsupported parameters should cause a server error rather than be silently replaced by benchmark code. Always verify what the engine actually applied.

## Study design

See `docs/protocol.md` for experimental controls, archival requirements and next milestones.
