# Eval plan — Phase 2 (English + code + agentic + contrastive binding)

Phase 1 was evaluated with a 6-task English log-likelihood suite (`benchmark/README.md`). That suite cannot see what Phase 2 is for: code, fill-in-the-middle, tool calling, reasoning-trace format, and entity-attribute binding. This plan adds evals for those, keeps the English suite as a forgetting gate, and fixes a cadence so every checkpoint is measured the same way.

All evals run on the **A10 eval host**, never on `fire01`. The only `fire01` interaction is copying a checkpoint, bandwidth-throttled (`rsync --bwlimit`) so training's NFS data reads are unaffected.

## Ground rules

- **Base model, no SFT.** ~211M active params, no instruction tuning. Every eval uses completion-style prompts (no chat-following assumed) and scores either log-likelihood of fixed candidates or the *form/content* of greedy generations.
- **Trends over absolutes.** Expect low absolute numbers on code/agentic. The signal is checkpoint-over-checkpoint movement vs. the Phase 1 baseline (`step_19073`), which should be near zero on code/agentic — confirming the evals are measuring something real.
- **Generation vs. scoring are different tests.** The binding work showed a forced two-way log-likelihood "win" can coexist with wrong free generation (marathon/sprint). Every binding claim needs both a scored probe *and* greedy generations.

## The suite

| Area | Eval | Measures | Scoring | Tier |
|---|---|---|---|---|
| English (forgetting gate) | `hellaswag, piqa, arc_easy, arc_challenge, winogrande, lambada_openai` via `scripts/run_benchmarks.py` | Did Phase 2 damage the English base? | log-likelihood | milestone |
| Binding | 100-case probe v2 (`probefinding-winogrande-20261121T185948.json`) | entity-attribute binding (comparable to Phase 1) | log-likelihood | every ckpt |
| Binding | **Probe v3** — new sentence structures not used by the contrastive corpus or v2 | binding that generalizes beyond the training templates | log-likelihood | every ckpt |
| Binding | Fixed greedy-generation prompts (marathon/sprint, wardrobe/doorway, capitals, …) | does binding show up in actual output? | saved side-by-side for inspection | every ckpt |
| FIM | Held-out single-line infill exact match on unseen code (PSM prompt, greedy to end of line) | is window-level FIM training working? | exact match | every ckpt |
| Agentic format | Held-out tool-use episodes: prompt up to the assistant turn that should call a tool | well-formed `<tool_call>`, valid JSON, name from the episode's `<tools>`, name = gold, arg keys = gold, `<|im_end|>` closure | deterministic checks | every ckpt |
| Reasoning format | Held-out reasoning episodes | `<think>` opened and `</think>` closed within budget | deterministic | every ckpt |
| Routing | `scripts/analyze_routing.py` — per (layer, expert) load by domain | dead/overloaded experts, specialization | counts | every ckpt |
| Code reasoning | **CRUXEval-O** (predict output of given code + input) | understanding code, not just pattern-matching | string match, model code never executed | milestone |
| Tool calling | **BFCL AST subset** (simple / multiple / parallel), converted to our `<tools>`/`<tool_call>` format | right function, right arguments | AST/argument match, no execution | milestone |
| Code generation | **HumanEval + MBPP pass@1**, base-model style (signature + docstring → body) | working code | **executes model-generated code** → sandbox required | milestone |
| Per-domain loss | `val/*`, `val_group/*` | cheapest early signal per domain | cross-entropy | live in W&B |

## Cadence

- **Every checkpoint (every 1,000 steps), ~15 min on A10:** probe v2 + v3, greedy prompts, FIM exact match, agentic + reasoning format, routing analysis.
- **Milestones — steps 5,000, 10,000, 19,073 (end of Stage 2A), 22,888 (final):** everything above plus the English suite, CRUXEval-O, BFCL, HumanEval/MBPP.
- **Baselines, once:** Phase 1 `step_19073` on the whole suite; **SmolLM2-360M** (similar size, same base tokenizer, `transformers`-native) on the standard benchmarks for a size-realistic reference.

## Gates (what "Phase 2 worked" means)

1. **English:** 6-task average within ~1 point of Phase 1 final (`benchmark/step_19073.json`).
2. **Code / agentic:** clearly above the Phase 1 baseline on FIM exact match, agentic format, CRUXEval-O, BFCL, HumanEval/MBPP.
3. **Binding:** probe **v3** and greedy generations improve — not only v2, which shares templates with the contrastive training data.

## Open items before trusting numbers

1. **Sandbox for HumanEval/MBPP.** Model-written code will be executed. Plan: container with no network, CPU/memory/time limits, on the throwaway A10 VM. If Docker isn't available there, this needs a decision before enabling.
2. **Contamination.** StarCoderData was decontaminated against HumanEval/MBPP by BigCode; ToolMind/Glaive vs. BFCL overlap is unknown → n-gram overlap check before reporting BFCL.
3. **Probe v3 must not share surface forms** with `scripts/gen_contrastive.py` styles (paired "because" statements, "Put the other way around", "Question: … Answer:", "it was not because", "People who saw it agreed") or with v2's "X … because the Y was" frame.

## Build order

1. Per-checkpoint suite (`scripts/eval_phase2_quick.py`), probe v3, Phase 1 baseline run. **← in progress**
2. CRUXEval-O + BFCL AST (no execution).
3. HumanEval/MBPP once the sandbox is settled.
4. SmolLM2-360M reference run.

Results go under `benchmark/phase2/` (one JSON per checkpoint per eval), summarized in `benchmark/README.md` at milestones.
