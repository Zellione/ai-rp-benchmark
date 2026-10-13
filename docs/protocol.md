# Research protocol draft

## Questions

1. Base vs Instruct vs RP/creative-writing fine-tunes within a verified model lineage.
2. Persona fidelity, lore retention/application and justified vs unjustified character development.
3. Sensitivity to context size (4K, 8K, 16K, 32K, 64K, 128K, 256K where supported), position and token distance.
4. Chat vs text completion and template/stop-sequence differences.
5. llama.cpp vs KoboldCpp with exactly the same GGUF and controlled inference settings.
6. Model-optimized RP samplers versus controlled common presets.

## Separate suites

- **Snapshot suite**: frozen transcript prefix and one identical final user prompt; replicate across context lengths/positions; 5–10 independent samples.
- **Longitudinal suite**: 25/50/100/200/300/400+ turns of simulator-driven interactions, checkpointed and archived.
- **Prompt API suite**: chat template vs manual text-completion prompt, measuring serialized token sequences where obtainable.
- **Engine parity suite**: same exact model file hash, same prompt token IDs where possible, matching sampler algorithms/order and numerical settings.

Do not infer a causally pure engine difference where prompt tokens, quantization, sampler semantics or cache numerics diverge.

## Evaluation (future work)

- Narrative Quality: prose/style, dialogue, pacing, creativity.
- RP Fidelity: persona invariants, lore use, player agency, epistemics, relationship dynamics.
- RP Robustness: drift across turns, context length/position, run-to-run dispersion, error frequency.
- Per-turn annotations must point to transcript turn IDs and cite evidence snippets.
- Rule-based tests + blinded human ratings + calibrated LLM judges; use separate calibration and held-out scenarios.
- Do not interpret legitimate character growth as drift.

## Scientific controls

- Model lineage (base/instruct/fine-tune), checkpoint/revision, exact file hashes and quantization.
- Backend binary/version/commit, GPU and runtime configuration, context and KV type, rope settings.
- Temperature/min_p/top_p/top_k/repetition settings, sampler order, actual supported parameters, seed.
- Prompt template string and ultimately token IDs, stopping criteria and actual prompt lengths.
- Never silently truncate, summarize or substitute parameters. Label where engines differ.
- Multiple independent runs; uncertainty intervals and model/config-level leaderboard separation.
- Optimize hyperparameters on tuning scenarios; assess quality on held-out scenarios.

## Next engineering milestones

1. Backend capabilities and tokenizer interfaces with versioned adapter fixtures.
2. Scenario schema with explicit stable persona constraints, variable relations, lore facts, turns and probes.
3. Deterministic replay/snapshots, checkpoints and crash-safe resumption.
4. Human-readable HTML transcript explorer + annotation store.
5. Simulator, judge, blind manual review, exports and dashboard.

**MVP limitation:** sequential scripted turns in this scaffold do not demonstrate natural RP behavior or score anything. The mock backend only verifies logging and plumbing.
