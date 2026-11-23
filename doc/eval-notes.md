# How Base-LM Evaluation Actually Works with `lm-evaluation-harness`

When you train a base language model from scratch — no supervised fine-tuning, no chat template, no RLHF — you cannot judge its quality by asking it to "answer with the letter B" or follow a prompt rubric. A raw pretrained model is a next-token probability distribution, not an assistant. Ask it to generate a chat response and you mostly measure formatting luck, not world knowledge or language modeling quality.

That is why standard base-model suites evaluate checkpoints through **log-likelihood scoring**, not free-form generation. In BubblSpace MoE-0.9B-A0.2B, we run six zero-shot benchmarks via EleutherAI's `lm-evaluation-harness`:

```text
hellaswag, piqa, arc_easy, arc_challenge, winogrande, lambada_openai
```

If you have ever wondered *who* downloads these datasets, *how* multiple-choice questions get turned into forward passes without the model ever outputting `"A"` or `"B"`, why WinoGrande is structurally backwards from the others, and where held-out perplexity fits in, this walkthrough traces the exact execution path through [`scripts/run_benchmarks.py`](../scripts/run_benchmarks.py) and [`src/eval/lm_eval_adapter.py`](../src/eval/lm_eval_adapter.py).

---

## The Mental Model: Who Does What?

The cleanest way to understand `lm-evaluation-harness` is as a strict separation of responsibilities:

- **The harness owns the benchmark.** It resolves task names in its registry, downloads and caches the Hugging Face datasets on first run, turns each dataset row into prompt/continuation pairs, calls your model for log-probabilities, computes `acc`, `acc_norm`, `perplexity`, and standard errors, and aggregates the final report. Nothing in this repository fetches benchmark datasets or computes accuracy.
- **Your adapter owns one question:** *"Given this context string, how likely is this continuation string under my model?"*

Because our `Transformer` (`src/model/transformer.py`) is a custom PyTorch `nn.Module` rather than a Hugging Face `transformers`-native architecture, the harness's built-in `HFLM` wrapper cannot load it. Instead, [`src/eval/lm_eval_adapter.py`](../src/eval/lm_eval_adapter.py) subclasses EleutherAI's `lm_eval.api.model.LM` directly as `BubblSpaceLM`. (See [What It Takes to Make a Custom MoE Transformers-Native](transformer-native.md) for what is required to drop this adapter and run `lm_eval --model hf` directly.)

```mermaid
flowchart LR
    CLI["scripts/run_benchmarks.py"] --> Load["Load .pt or .safetensors"]
    Load --> Adapter["BubblSpaceLM(model, tok, cfg)"]
    Adapter --> Harness["lm_eval.simple_evaluate"]
    Harness -->|"Fetch HF datasets + build pairs"| LL["BubblSpaceLM.loglikelihood"]
    LL -->|"batch_size=1 forward"| Model["Transformer.forward"]
    Model -->|"slice logits[..., :49164]"| Score["(sum_logprob, is_greedy)"]
    Score --> Harness
    Harness -->|"argmax / length-norm / exp(-mean)"| Output["Console + JSON summary"]
```

---

## Inside the Runner and Adapter

### 1. Entry Point: `scripts/run_benchmarks.py`

Running a benchmark pass against any checkpoint looks like this:

```bash
python scripts/run_benchmarks.py \
    --config configs/base.yaml \
    --checkpoint /path/to/step_6000.pt \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --output results/step_6000.json
```

The script handles four setup steps before handing control to the harness:

1. **MoE backend resolution:** If `cfg.model.moe.backend == "auto"`, it probes `torch._grouped_mm` once via `probe_grouped_mm()` and selects `"grouped_mm"` on capable CUDA builds or `"fallback"` otherwise.
2. **Checkpoint loading:** `load_weights()` inspects the extension. A `.safetensors` file is loaded straight into `model.load_state_dict(...)`. A training `.pt` checkpoint contains optimizer and scheduler state alongside the weights, so it extracts `ckpt["model"]` (and reads `ckpt.get("step")` for metadata) while ignoring training state.
3. **Wrapping the model:** It sets `model.eval()`, wraps `(model, tok, cfg, device)` in `BubblSpaceLM`, and calls `lm_eval.simple_evaluate(model=lm, tasks=task_list, limit=args.limit)`.
4. **Persisting results:** It prints `results["results"]` and optionally writes a JSON file alongside the checkpoint path, step, task list, and `--limit`.

### 2. Scoring a Single Pair: `BubblSpaceLM._score_continuation`

All six default benchmarks only ever invoke `BubblSpaceLM.loglikelihood(requests)`. Each `Instance` carries `(context, continuation)` in `req.args`. Both strings are encoded with `add_special_tokens=False` and passed to `_score_continuation`:

```python
full_enc = context_enc + continuation_enc
inp = torch.tensor([full_enc[:-1]], dtype=torch.long, device=self.device)
targets = torch.tensor(full_enc[1:], dtype=torch.long, device=self.device)

with torch.autocast(device_type=self.device, dtype=self._autocast_dtype, enabled=(self.device == "cuda")):
    logits, _, _, _ = self.model(inp)
logits = logits[0, :, : self.real_vocab_size].float()

cont_len = len(continuation_enc)
cont_logits = logits[-cont_len:]
cont_targets = targets[-cont_len:]

log_probs = F.log_softmax(cont_logits, dim=-1)
token_logprobs = log_probs.gather(1, cont_targets.unsqueeze(-1)).squeeze(-1)
greedy_tokens = cont_logits.argmax(dim=-1)
is_greedy = bool((greedy_tokens == cont_targets).all().item())

return token_logprobs.sum().item(), is_greedy
```

Four implementation details here are easy to overlook, and each one matters for exactness:

1. **Teacher-forced next-token shift:** Concatenating `context_enc + continuation_enc` into `full_enc` and feeding `full_enc[:-1]` against targets `full_enc[1:]` aligns position `t` with predicting token `t + 1`. Taking the trailing `cont_len` positions scores *only* the continuation tokens, conditioned on the full context plus every preceding continuation token.
2. **Boundary handling (empty context & overflow):** If `context_enc` is empty (for instance, in an unconditional probability check), the adapter prepends `[self.eot_token_id]` so the first continuation token still has a preceding input token. If `len(context_enc) + len(continuation_enc) > max_seq_len` (`2048`), it truncates the context from the left so the immediate prefix before the continuation is preserved.
3. **Slicing out the padded vocabulary tail:** Our tokenizer (`bubblspace-tokenizer`) has **49,164** real tokens, while `model.vocab_size` is padded to **49,280** (a multiple of 128 for Tensor Core alignment). During training (`scripts/train.py`), those 116 unused output rows are sliced out before cross-entropy so they never appear in the softmax denominator. `BubblSpaceLM` applies the exact same slice (`logits[0, :, : self.real_vocab_size].float()`) before `log_softmax` and `argmax`. Skipping that slice would let untrained padding rows leak probability mass—or even win `argmax`.
4. **Why `batch_size = 1`:** Our `Attention` module (`src/model/attention.py`) calls `F.scaled_dot_product_attention(..., is_causal=True)` with no `attention_mask` support. That is ideal for packed, fixed-length `2048`-token training sequences, but batching variable-length eval requests requires padding masks so pad tokens do not corrupt keys, values, or RoPE positions. Running unbatched (`batch_size=1`) guarantees exact math, and a few thousand short forward passes take only a tiny fraction of a 19k-step H100 training run.

What about `generate_until()`? It is implemented in `BubblSpaceLM` (greedy decoding up to `max_gen_toks` with `until` stop strings) purely for interface completeness in case someone passes a generative task later. None of the six default benchmarks call it.

---

## How Each Benchmark Turns Rows into `(context, continuation)` Pairs

Every multiple-choice task in `lm-evaluation-harness` shares one subtle formatting convention: **`target_delimiter`** (default `" "`, a single leading space). The harness prepends that space to each continuation string rather than leaving it on the context. With BPE tokenizers like our SmolLM2-based tokenizer, leading whitespace is part of the word token (`" thermometer"` vs `"thermometer"`). Putting the space on the continuation ensures the model scores a normal word token instead of a glued-on suffix.

Below is what the harness actually sends into `BubblSpaceLM.loglikelihood()` for each of the six tasks in zero-shot mode.

### 1. HellaSwag — One Context, Four Endings

HellaSwag tests commonsense physical/situational continuation. Each validation row carries an `activity_label`, two context fragments (`ctx_a`, `ctx_b`), four candidate `endings`, and a `label` index.

**Raw dataset row:**

```text
activity_label: "Removing ice from a car"
ctx_a:          "A man scrapes the windshield."
ctx_b:          "he"
endings: [
  "then puts the scraper in the trunk.",
  "starts dancing on the roof.",
  "pours coffee on the glass.",
  "throws the scraper into the street.",
]
label: "0"
```

**Pairs sent to `loglikelihood` (4 requests):**

The harness cleans WikiHow bracket artifacts, capitalizes `ctx_b`, and builds one shared query with four continuations:

```text
1. ("Removing ice from a car: A man scrapes the windshield. He", " then puts the scraper in the trunk.")
2. ("Removing ice from a car: A man scrapes the windshield. He", " starts dancing on the roof.")
3. ("Removing ice from a car: A man scrapes the windshield. He", " pours coffee on the glass.")
4. ("Removing ice from a car: A man scrapes the windshield. He", " throws the scraper into the street.")
```

**How the harness scores it:**

- **`acc`**: picks `argmax(lls)` over the four summed continuation log-probabilities and checks whether it matches `label` (`0`).
- **`acc_norm`**: divides each continuation's summed log-probability by the **character length** of that choice string (`lls / completion_len`), then takes `argmax`. Because every extra token adds another negative log-probability term, raw `acc` systematically favors shorter endings; `acc_norm` corrects that length bias and is the headline number usually reported for HellaSwag.

---

### 2. PIQA — One Goal, Two Solutions

PIQA (Physical Interaction QA) tests everyday physical commonsense.

**Raw dataset row:**

```text
goal:  "How do you loosen a stuck jar lid?"
sol1:  "Run hot water over the lid, then twist."
sol2:  "Put the jar in the freezer overnight."
label: 0
```

**Pairs sent to `loglikelihood` (2 requests):**

```text
1. ("Question: How do you loosen a stuck jar lid?\nAnswer:", " Run hot water over the lid, then twist.")
2. ("Question: How do you loosen a stuck jar lid?\nAnswer:", " Put the jar in the freezer overnight.")
```

**How the harness scores it:**

Same mechanism as HellaSwag: `acc` compares raw summed continuation log-probabilities, and `acc_norm` normalizes by choice character length. Notice that option letters like `"A"` or `"B"` never appear anywhere in the prompt—the model directly compares the plausibility of the full answer texts.

---

### 3. ARC-Easy and ARC-Challenge — One Question, N Answer Choices

Both `arc_easy` and `arc_challenge` come from `allenai/ai2_arc` and use the exact same prompt template and scoring rule. The only difference is the split (`ARC-Easy` vs the harder `ARC-Challenge` subset of grade-school science questions).

**Raw dataset row:**

```text
question:      "Which tool measures temperature?"
choices.text:  ["ruler", "thermometer", "scale", "clock"]
choices.label: ["A", "B", "C", "D"]
answerKey:     "B"
```

**Pairs sent to `loglikelihood` (N requests, usually 4, occasionally 3 or 5):**

```text
1. ("Question: Which tool measures temperature?\nAnswer:", " ruler")
2. ("Question: Which tool measures temperature?\nAnswer:", " thermometer")
3. ("Question: Which tool measures temperature?\nAnswer:", " scale")
4. ("Question: Which tool measures temperature?\nAnswer:", " clock")
```

**How the harness scores it:**

The harness maps `answerKey` to an index via `choices.label.index(answerKey)` (here `"B"` -> `1`) and computes `acc` and `acc_norm` over the choice strings. Again, the model never has to know what multiple-choice letter labels mean; it only scores the actual candidate answers.

---

### 4. WinoGrande — The Inverted Case (Two Contexts, One Shared Tail)

WinoGrande tests pronoun/coreference resolution with minimal pairs, and its harness implementation is the opposite of HellaSwag, PIQA, and ARC.

**Raw dataset row:**

```text
sentence: "The trophy does not fit in the suitcase because _ is too large."
option1:  "trophy"
option2:  "suitcase"
answer:   "1"
```

If you scored the option words themselves, a model could cheat simply because `"trophy"` is more common in general English than `"suitcase"`. Instead, the harness uses **`multiple_input` mode**: it splits the sentence at the blank `_`, plugs each option into the **context**, and scores both contexts on the **exact same shared continuation tail**:

**Pairs sent to `loglikelihood` (2 requests):**

```text
1. ("The trophy does not fit in the suitcase because trophy",   " is too large.")
2. ("The trophy does not fit in the suitcase because suitcase", " is too large.")
```

**How the harness scores it:**

It compares:

$$P(\text{" is too large."} \mid \text{"...because trophy"}) \quad \text{vs.} \quad P(\text{" is too large."} \mid \text{"...because suitcase"})$$

`answer: "1"` maps to index `0` (`option1`). Because both pairs score the *identical* continuation string, their lengths are guaranteed to be equal—so length normalization is meaningless and WinoGrande reports only `acc` (no `acc_norm`).

---

### 5. LAMBADA (`lambada_openai`) — One Passage, One Final Word

LAMBADA is not multiple choice at all (`output_type: loglikelihood`). It tests whether the model can track long-range narrative context well enough to predict the final word of a passage.

**Raw dataset row:**

```text
text: "Alice opened the window. Outside she saw a bright garden"
```

**Pair sent to `loglikelihood` (1 request):**

The harness splits on whitespace, puts everything except the final word in `context`, and puts `" " + last_word` in `continuation`:

```text
1. ("Alice opened the window. Outside she saw a bright", " garden")
```

**How the harness scores it:**

Here, both elements of the `(ll, is_greedy)` tuple returned by `BubblSpaceLM.loglikelihood()` are used directly:

- **`acc`**: `int(is_greedy)`. Every single token of the final word must match the model's greedy `argmax` prediction. If `" garden"` is one token, the model's top-1 logit must be that token. If a rarer final word splits into two BPE subwords, *both* subwords must be the top-1 argmax in sequence.
- **`perplexity`**: aggregated across the test set as $\exp\left(-\frac{1}{N}\sum_{i=1}^{N} \text{ll}_i\right)$, where each $\text{ll}_i$ is the summed log-probability of the entire final word (so a multi-token final word still contributes one whole-word log-probability).

---

### Quick Reference Across the Six Tasks

| Task | `output_type` | Requests / Example | What Varies Across Requests | Reported Metrics |
| :--- | :--- | :---: | :--- | :--- |
| `hellaswag` | `multiple_choice` | 4 | Continuation (4 endings) | `acc`, `acc_norm` |
| `piqa` | `multiple_choice` | 2 | Continuation (2 solutions) | `acc`, `acc_norm` |
| `arc_easy` / `arc_challenge` | `multiple_choice` | N (usually 4) | Continuation (answer texts) | `acc`, `acc_norm` |
| `winogrande` | `multiple_choice` (`multiple_input`) | 2 | **Context** (blank filled); tail shared | `acc` |
| `lambada_openai` | `loglikelihood` | 1 | Single `(passage, " " + last_word)` pair | `acc` (via `is_greedy`), `perplexity` |

---

## Held-Out Perplexity: What Exists Today (and What Doesn't)

It is easy to assume a training repository automatically tracks validation perplexity, so it is worth being explicit about what is currently wired up:

1. **No validation loop in Phase 1 training (`scripts/train.py`):** *(Phase 2 adds one: per-pool held-out validation loss every 250 steps; see "Phase 2 Evaluation" below.)* During Phase 1 pretraining, `scripts/train.py` computes and logs cross-entropy (`ce_loss`), MoE load-balance loss (`lb_loss`), and router z-loss (`z_loss`) strictly on the training stream. There is no held-out validation dataloader or standalone perplexity script in `scripts/`.
2. **Rolling perplexity is supported via the harness (`loglikelihood_rolling`):** `BubblSpaceLM` implements `loglikelihood_rolling(requests)`, which is the method `lm-evaluation-harness` calls for perplexity benchmarks such as `wikitext`. It is not in `DEFAULT_TASKS`, so the default command never invokes it, but you can run it immediately by adding `wikitext` to `--tasks`:

```bash
python scripts/run_benchmarks.py \
    --config configs/base.yaml \
    --checkpoint /path/to/step_6000.pt \
    --tasks wikitext
```

### How `loglikelihood_rolling` Works in `BubblSpaceLM`

For rolling perplexity tasks, each `Instance` contains a single document string `(text,)` rather than a `(context, continuation)` pair. `BubblSpaceLM.loglikelihood_rolling()` tokenizes the document and walks it in **non-overlapping chunks of `stride = max_seq_len` (`2048`)**:

- Each chunk runs one forward pass (`inp = chunk[:-1]`, `targets = chunk[1:]`), slices logits to `:real_vocab_size` (`49,164`), and sums token log-probabilities.
- The adapter returns the total document log-probability `total_logprob`.
- The harness divides that log-probability by the document's word count and UTF-8 byte count to report `word_perplexity`, `byte_perplexity`, and `bits_per_byte`.

Because the windows do not overlap, the first tokens of chunk 2 are conditioned only on the start of chunk 2 rather than the tail of chunk 1. That is the standard `lm-evaluation-harness` fast simplification (good enough for consistent checkpoint-to-checkpoint sanity checks and much cheaper than a densely strided sliding window, where every chunk overlaps heavily). Note also that the very first token of each chunk has no preceding token in `chunk[:-1]` (so `len(chunk) - 1` transitions are scored per chunk).

### If You Want a Dedicated Held-Out FineWeb-Edu Perplexity Check

If you later want an in-distribution held-out perplexity metric alongside `wikitext` (for example, a fixed FineWeb-Edu slice excluded from `src.data.prepare`), you have two lightweight options:

- **Standalone script (~40 lines):** Read a held-out `.bin` shard with the same `uint16` memmap reader from `src/data/dataset.py`, run `model(x)` under `torch.no_grad()` and `bf16` autocast over fixed `2048`-token windows (slicing `:real_vocab_size` identically to `scripts/train.py`), and report `mean_ce` and `math.exp(mean_ce)`. Because it uses fixed `2048`-token blocks with no padding, you can batch it (`micro_batch=16`) without changing `Attention`.
- **Custom harness task YAML:** Point a local `lm_eval` task YAML with `output_type: loglikelihood_rolling` at your held-out JSONL/Parquet file and pass `--include_path` to `lm_eval`.

---

## Results Log

### step_7000 (2026-11-20)

Checkpoint: `step_7000.pt` — step 7,000 / 19,073 planned steps (~37% through
the single 10B-token FineWeb-Edu pretraining run), loss ≈2.8 at this point.
Tokens seen so far: 7,000 × 524,288 ≈ **3.67B** (of a 10B-token total planned
for this entire run). Raw results: [`benchmark/step_7000.json`](../benchmark/step_7000.json).

| Task | Metric | Score | Random baseline |
|---|---|---|---|
| PIQA | acc_norm | 64.7% | 50% (2-choice) |
| ARC-Easy | acc_norm | 51.1% | ~25% (multi-choice) |
| HellaSwag | acc_norm | 37.0% | 25% (4-choice) |
| LAMBADA (openai) | acc (exact last-word match) | 29.0% | ~0% (strict exact-match) |
| LAMBADA (openai) | perplexity | 58.1 | — |
| WinoGrande | acc | 52.4% | 50% (2-choice) |
| ARC-Challenge | acc_norm | 28.0% | 25% (4-choice) |

**Read:** tasks that reward broad surface-pattern learning (PIQA, ARC-Easy,
HellaSwag) are clearly above chance already. LAMBADA's strict exact-match
accuracy (29%) is a genuinely positive signal of real language-modeling
quality this early. The two hardest, most reasoning-heavy benchmarks
(ARC-Challenge, WinoGrande) sitting near chance is expected at this model
size (~915M total / ~211M active) and training stage — not a sign of a
problem.

#### Comparison against other models of similar scale

| Task | **Ours (step_7000)** | GPT-2-124M (fully trained) | Pythia-160M (fully trained) | Pythia-410M (fully trained) | SmolLM2-360M (fully trained, curated data) |
|---|---|---|---|---|---|
| PIQA | 64.7% | 62.9% | 73.1% | 74.1% | 77.6% |
| ARC-Easy | 51.1% | 39.5% | 46.3% | 51.2% | ~61.0%* |
| ARC-Challenge | 28.0% | ~19%** | 19.2% | 21.8% | ~61.0%* |
| HellaSwag | 37.0% | 31.1% | — | — | 69.3% |
| WinoGrande | 52.4% | 51.6% | 59.2% | 59.3% | 61.1% |
| LAMBADA acc | 29.0% | 32.6% | 63.3% | 65.7% | — |

\*SmolLM2's reported "ARC" is an E+C average, not split — not perfectly
comparable to our per-split numbers.
\*\*Commonly-cited approximate figure; less firmly sourced than the others
in this table.

#### How many tokens were these comparison models actually trained on?

| Model | Training tokens | Notes |
|---|---|---|
| **Ours (step_7000)** | ~3.67B seen so far (**10B planned total** for the entire run) | Single pass (not multiple epochs) over FineWeb-Edu `sample-10BT` |
| GPT-2-124M | ~8-9B unique tokens (WebText), ~10-53B total across training | WebText itself is ~8-9B tokens; the original paper does not state an exact epoch count, so total tokens-seen estimates vary by source |
| Pythia-160M / 410M | ~300B tokens (exactly 299,892,736,000) | 1 epoch over the Pile: 143,000 steps × 1024 batch × 2048 seq_len — precisely documented in the Pythia paper |
| SmolLM2-360M | ~4T tokens (4,000B) | Multi-stage curated-data curriculum (same tokenizer family as ours) |

**Why this context matters:** our model is already ahead of GPT-2-124M on
most tasks and competitive with the fully-trained Pythia-160M on ARC-Easy/
ARC-Challenge — despite still being mid-training *and* working with a far
smaller total token budget (10B) than every comparison point here (30x
smaller than Pythia's 300B, over 1000x smaller than SmolLM2's 4T). That
makes the LAMBADA gap unsurprising (it specifically rewards precise
long-range calibration, which scales strongly with total tokens seen) and
reframes the strong early showing elsewhere as a genuinely efficient result
for the architecture, not just "it got lucky" — though it also means this
run's *final* checkpoint will still never catch up to Pythia-410M or
SmolLM2-360M on raw data volume alone.

Caveat: this is not a perfectly apples-to-apples comparison — these models
differ in total/active parameter split (ours is MoE: 915M total / 211M
active, vs. fully dense comparisons), training token budgets, and possibly
minor eval-harness version differences across sources.

Sources: [Pythia paper](https://arxiv.org/pdf/2304.01373),
[EleutherAI/pythia-160m-v0](https://huggingface.co/EleutherAI/pythia-160m-v0),
[SmolLM2 paper](https://arxiv.org/pdf/2502.02737).

### step_8000 (2026-11-20)

Checkpoint: `step_8000.pt` — step 8,000 / 19,073 planned steps (~42% through
the run). Tokens seen: 8,000 × 524,288 ≈ **4.19B**. Raw results:
[`benchmark/step_8000.json`](../benchmark/step_8000.json).

| Task | Metric | step_7000 | step_8000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 64.7% | 64.5% | -0.2 (noise) |
| ARC-Easy | acc_norm | 51.1% | 50.7% | -0.4 (noise) |
| HellaSwag | acc_norm | 37.0% | 38.0% | +1.0 |
| LAMBADA (openai) | acc | 29.0% | 31.0% | **+2.0** |
| LAMBADA (openai) | perplexity | 58.1 | **50.3** | **-7.8 (better)** |
| WinoGrande | acc | 52.4% | 51.7% | -0.7 (noise) |
| ARC-Challenge | acc_norm | 28.0% | 27.6% | -0.4 (noise) |

**Read:** most metrics are flat within statistical noise over just 1,000
steps (stderr on most of these tasks is ~1-1.4%, so ±1 point swings aren't
meaningful on their own). LAMBADA is the exception — a real, consistent
improvement. This is the exact metric flagged in the `step_7000` comparison
above as most dependent on total tokens seen (vs. Pythia/SmolLM2's much
larger token budgets), so it being the first to move clearly as training
continues is a good, expected signal rather than a coincidence.

### step_9000 (2026-11-20)

Checkpoint: `step_9000.pt` — step 9,000 / 19,073 planned steps (~47% through
the run). Tokens seen: 9,000 × 524,288 ≈ **4.72B**. Raw results:
[`benchmark/step_9000.json`](../benchmark/step_9000.json).

| Task | Metric | step_8000 | step_9000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 64.5% | **65.9%** | **+1.4** |
| ARC-Easy | acc_norm | 50.7% | 50.8% | +0.1 (noise) |
| HellaSwag | acc_norm | 38.0% | 38.5% | +0.5 |
| LAMBADA (openai) | acc | 31.0% | 30.6% | -0.4 (noise) |
| LAMBADA (openai) | perplexity | 50.3 | **49.3** | -1.0 (better) |
| WinoGrande | acc | 51.7% | 51.5% | -0.2 (noise) |
| ARC-Challenge | acc_norm | 27.6% | 27.2% | -0.4 (noise) |

**Read:** PIQA shows a real move this step (+1.4, clear of its ~1.1%
stderr), and HellaSwag keeps climbing steadily (+0.5, consistent with its
trend since step_7000: 37.0% → 38.0% → 38.5%). LAMBADA perplexity continues
its monotonic improvement (58.1 → 50.3 → 49.3), now 15% better than the
step_7000 baseline — still the cleanest token-count-sensitive signal in
this suite. The reasoning-heavy benchmarks (ARC-Challenge, WinoGrande)
remain flat within noise, as expected at this model size and training
stage.

---

### step_18000 (2026-11-21)

Checkpoint: `step_18000.pt` — step 18,000 / 19,073 planned steps (~94%
through the run). Tokens seen: 18,000 × 524,288 ≈ **9.44B**. Raw results:
[`benchmark/step_18000.json`](../benchmark/step_18000.json). This entry
jumps 9,000 steps ahead of the last logged checkpoint (`step_9000`), so
these deltas reflect ~4.7B additional tokens, not the usual 1,000-step
increment.

| Task | Metric | step_9000 | step_18000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 65.9% | **68.8%** | **+2.9** |
| ARC-Easy | acc_norm | 50.8% | **56.6%** | **+5.8** |
| HellaSwag | acc_norm | 38.5% | **42.6%** | **+4.1** |
| LAMBADA (openai) | acc | 30.6% | **35.3%** | **+4.7** |
| LAMBADA (openai) | perplexity | 49.3 | **31.0** | **-18.3 (better)** |
| WinoGrande | acc | 51.5% | 50.7% | -0.8 (noise) |
| ARC-Challenge | acc_norm | 27.2% | **29.9%** | **+2.7** |

**Read:** every task except WinoGrande moves well clear of its stderr band
(~1-1.4% on most tasks) — the clearest real-learning signal of the run so
far, matching the near-doubling of tokens seen since step_9000. LAMBADA
perplexity has now fallen 37% from the step_7000 baseline (58.1 → 31.0)
with accuracy up 6.3 points (29.0% → 35.3%), confirming it as the most
token-sensitive metric tracked here. ARC-Easy (+5.8) and HellaSwag (+4.1)
post their largest single-entry jumps of the run. WinoGrande stays flat
within noise — expected, as it and ARC-Challenge are the two hardest,
most reasoning-heavy tasks in this suite, and WinoGrande typically needs
much more scale/training before it moves meaningfully (ARC-Challenge did
move here, +2.7).

---

### step_19073 (2026-11-21) — Phase 1 final checkpoint

Checkpoint: `step_19073.pt` — step 19,073 / 19,073 planned steps (100%,
Phase 1 complete), full 10.0B tokens. Raw results:
[`benchmark/step_19073.json`](../benchmark/step_19073.json).

| Task | Metric | step_18000 | step_19073 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 68.8% | 68.2% | -0.6 (noise) |
| ARC-Easy | acc_norm | 56.6% | 56.1% | -0.5 (noise) |
| HellaSwag | acc_norm | 42.6% | 42.9% | +0.3 (noise) |
| LAMBADA (openai) | acc | 35.3% | 34.8% | -0.5 (noise) |
| LAMBADA (openai) | perplexity | 31.0 | 32.0 | +1.0 (noise) |
| WinoGrande | acc | 50.7% | 51.4% | +0.7 (noise) |
| ARC-Challenge | acc_norm | 29.9% | 29.8% | -0.1 (noise) |

**Read:** flat within stderr across all seven metrics over the final 1,073
steps, exactly as expected since the cosine LR schedule had already decayed
to its `min_lr_ratio` floor well before this point -- there just isn't much
room left for the last steps to move anything. This closes out Phase 1:
the model plateaus at PIQA 68%, ARC-Easy 56%, HellaSwag 43%, LAMBADA acc
35%/ppl 32, with WinoGrande and ARC-Challenge essentially at their
model-size-appropriate floor (~50% and ~30%). Phase 2 continual training
is the next lever, not further steps on this run.

---

## Phase 2 Evaluation (continual training: English + code + agentic + contrastive)

Phase 2 continues from `step_19073.pt` for 22,888 more steps (12B tokens). The
six-task suite above can't see what Phase 2 is for, so it now acts as a
**forgetting gate**, and a broader suite was added. The full design is in
[`eval-plan-phase2.md`](eval-plan-phase2.md). Phase 2 step numbers below are
Phase 2's own counter. In W&B, `global_step` = Phase 2 step + 19,073.

### What was added, and how each eval is scored

| Eval | Script | Scoring | Cadence |
|---|---|---|---|
| Binding probe v2 (100 cases, 10 categories) | `scripts/probe_entity_binding.py` | log-likelihood minimal pairs: correct only if the preference flips with the named entity | every 1,000 steps |
| Binding probe v3 (70 cases) | same | same; sentence structures disjoint from v2 and from the contrastive training corpus | every 1,000 steps |
| Binding probe v4 (112 cases, added at step 9,000) | same | same; built by `scripts/gen_contrastive_v2.py` only from sentence frames, entity pairs and names **held out** of all contrastive training data. A coin flip scores ~25% (both contexts must be right) | every 2,000 steps from 11,000 |
| Code fill-in-the-middle (60 held-out lines) | `scripts/eval_phase2_quick.py` | PSM prompt, greedy to end of line, exact match | every 1,000 steps |
| Agentic tool-call format (23 held-out episodes) | same | emits `<tool_call>`, closes it, valid JSON, name = gold, argument keys = gold, ends the turn | every 1,000 steps |
| Reasoning format (8 held-out episodes) | same | opens `<think>`, closes `</think>` within 512 tokens | every 1,000 steps |
| MoE routing per (layer, expert) | `scripts/analyze_routing.py` | load CV, entropy, starved (<0.1×) / overloaded (>3×) experts, domain specialization | every 1,000 steps |
| CRUXEval-O (800) | `scripts/eval_phase2_milestone.py` | predict the output of given code; `ast.literal_eval` comparison; **model code is never executed** | milestones |
| BFCL v3 simple / multiple / parallel (100 each) | same | AST match of function name and arguments in our `<tools>`/`<tool_call>` format; 11 contaminated questions excluded | milestones |
| English 6-task (forgetting gate) | `scripts/run_benchmarks.py` | as above | every 2,000 steps from step 7,000 |
| HumanEval (164) + MBPP (500) pass@1 | `scripts/eval_code_gen.py` → `scripts/sandbox_exec.py` | generation and execution are separate steps. Each program runs in its own container: no network, 1 CPU, 512 MB, 64 processes, read-only filesystem, non-root, all capabilities dropped, 10 s kill. Image pinned by digest. Adversarial self-test (`--selftest`) must pass first | milestones |
| Per-pool validation loss (16 pools) | `scripts/train.py` | held-out tail of each pool, 2M tokens each | every 250 steps, live in W&B |

Three measurement fixes came first, because the numbers aren't trustworthy
without them:

1. **Determinism.** Two identical forward passes of this MoE differed by up
   to ~0.1 in the logits. Non-deterministic CUDA kernels flip near-tied top-8
   router choices, so greedy generations diverged and close probe cases
   flipped between runs. `src/eval/determinism.py` (deterministic algorithms
   plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`) makes repeated runs bit-identical.
   All Phase 2 evals use it. This is why the Phase 1 v2 probe reads **24/100**
   here instead of the earlier 25/100: same checkpoint, deterministic
   re-score.
2. **Contamination.** `scripts/contamination_check.py` runs a 13-gram overlap
   check of BFCL against the raw ToolMind / Hermes / Glaive sources. It found
   11 of 603 questions (simple 3, multiple 1, parallel 7), and those are
   excluded from scoring. Result: `benchmark/phase2/contamination_bfcl.json`.
3. **The exact held-out split.** The quick suite reads the same validation
   tail the training loader holds out (`val_tokens_per_pool` from
   `configs/phase2.yaml`), so FIM and agentic prompts are never training data.

Generation evals use a KV cache (`Transformer.new_kv_cache()`), tested to
match the full forward pass.

### Results through step 19,000 (all of Stage 2A, peak learning rate)

Raw files: [`benchmark/phase2/`](../benchmark/phase2/).

| Eval | Phase 1 | P2 1k | 3k | 5k | 7k | 9k | 11k | 13k | 15k | 17k | **19k** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic: valid call, name = gold | 0% | 17% | 39% | 30% | 48% | 30% | 26% | 35% | 43% | 35% | 35% |
| Agentic: argument keys = gold | 0% | 13% | 39% | 30% | 43% | 30% | 26% | 30% | 43% | 35% | 30% |
| Reasoning: closes `</think>` ≤512 tok | 0% | 0% | 0% | 0% | 0% | 12% | 0% | 0% | 0% | 0% | 0% |
| FIM exact match | 0% | 0% | 0% | 3.3% | 5.0% | 1.7% | 3.3% | 1.7% | 1.7% | 3.3% | 3.3% |
| Probe v2 (/100) | 24 | 25 | 25 | 25 | 28 | 24 | 24 | 22 | 29 | 28 | 30 |
| Probe v3 (/70) | 4 | 6 | 6 | 5 | 6 | 3 | 1 | 4 | 4 | 4 | 4 |
| Probe v4 (/112, chance ≈ 28) | 13 | 21 | — | — | — | 11 | 17 | 13 | 24 | 18 | 17 |
| Routing: mean load CV | 0.623 | 0.385 | 0.318 | 0.293 | 0.285 | 0.274 | 0.264 | 0.260 | 0.252 | 0.252 | 0.246 |

The agentic set has only 23 episodes, so one episode is ~4 points. Read it as a trend, not as exact percentages.

**Milestones:**

| Benchmark | Phase 1 final | P2 step 5,000 | **P2 step 19,000** |
|---|---|---|---|
| CRUXEval-O pass@1 (800) | 0.0% (0% parsable) | 8.5% (94% parsable) | **17.7%** (98% parsable) |
| BFCL simple, AST | 0% | 9% | **25%** |
| BFCL multiple, AST | 0% | 6% | **13%** |
| BFCL parallel, AST | 0% | 0% | 0% |

**Code generation with execution (HumanEval / MBPP):**

| | Phase 1 final | P2 step 9,000 | **P2 step 19,000** |
|---|---|---|---|
| HumanEval pass@1 | 0/164 (0.0%) | 15/164 (9.1%) | **19/164 (11.6%)** |
| MBPP pass@1 (full test) | not run (no code in Phase 1) | 42/500 (8.4%) | **63/500 (12.6%)** |

Notes on how this was measured:
- **Prompt boundary fix.** HumanEval prompts end with a bare `"\n"`, while this tokenizer merges a newline with the following indentation. The model answered the lone trailing newline with `<|endoftext|>`, so every completion came out empty. The trailing newline is stripped before generation.
- **No early-exit cheats.** Every pass, at both checkpoints, was checked for `exit()` / `sys.exit` / `os._exit` before the tests run. None found.
- **Steps 9k → 19k:** HumanEval +6 new solves / −2 lost; MBPP +32 / −11. Syntax errors fell (HumanEval 23 → 17, MBPP 45 → 22). Most failures are code that runs but returns the wrong answer.
- **Format failures on MBPP that SFT should fix:** NameError 36, and `input()` calls 21.
- **Easy problems only so far.** Every HumanEval pass is in the first ~60 problems.
- **Contamination isn't ruled out.** `python-edu-cleaned` hasn't been checked against HumanEval/MBPP. StarCoderData was decontaminated by BigCode.

**English forgetting gate** (every 2,000 steps):

| Task | Metric | Phase 1 | 5k | 7k | 9k | 11k | 13k | 15k | 17k | **19k** |
|---|---|---|---|---|---|---|---|---|---|---|
| HellaSwag | acc_norm | 42.86 | 40.39 | 40.69 | 40.64 | 40.28 | 40.61 | 40.65 | 40.72 | 40.26 |
| PIQA | acc_norm | 68.17 | 67.30 | 67.52 | 67.90 | 67.19 | 67.57 | 67.46 | 66.92 | 66.87 |
| ARC-Easy | acc_norm | 56.10 | 52.78 | 51.18 | 52.57 | 53.20 | 52.90 | 51.64 | 52.15 | **55.09** |
| ARC-Challenge | acc_norm | 29.78 | 27.99 | 27.39 | 27.99 | 27.99 | 29.18 | 27.90 | 29.61 | 29.18 |
| WinoGrande | acc | 51.38 | 51.07 | 51.70 | 52.72 | 52.57 | 51.78 | 50.67 | 53.75 | 52.25 |
| LAMBADA (openai) | acc | 34.80 | 32.85 | 33.03 | 35.30 | 34.54 | 34.02 | 32.08 | 33.96 | 32.21 |
| **Average (6 accuracies)** | | **47.18** | 45.40 | 45.25 | 46.19 | 45.96 | 46.01 | 45.07 | 46.18 | **45.98** |

### Read

- **Code and agentic grew steadily through the plateau.** CRUXEval-O doubled from step 5,000 to 19,000 (8.5% → 17.7%), and BFCL simple nearly tripled (9% → 25%). Both are scored without running model code. Under real execution, HumanEval went 9.1% → 11.6% and MBPP 8.4% → 12.6% from step 9,000 to 19,000 (Phase 1: 0%). The held-out agentic format check moved within 26–48%, with 23 episodes, which is noisy.
- **Reasoning closed `</think>` for the first time at step 9,000** (1 of 8
  episodes within 512 tokens). Until then it opened the block and never
  finished. BFCL shows the same problem: many failures are a ToolMind-style
  `<think>` block that never reaches the call.
- **English dipped, then held at roughly −1 for the rest of Stage 2A** (step 19,000: 45.98; ARC-Easy back to 55.1). Before that, the average fell
  to −1.78 at step 5,000 and was flat at step 7,000, then recovered to −0.99
  at step 9,000. LAMBADA is now above Phase 1, and WinoGrande is the highest
  in either phase (within its stderr). HellaSwag (−2.2) is the one task still
  clearly down. Phase 1 was measured after its learning-rate decay and these
  are at Phase 2's peak, so the anneal should add more. As insurance,
  especially for HellaSwag, Stage 2B keeps FineWeb-Edu at ~5.7% instead of 0%
  (applied by resuming from `step_19000.pt`; an exact continuation).
- **Binding has not moved, and probe v4 explains why.** v2 stays 24–31 and v3
  3–6. Probe v4 uses only phrasings and entities that never appear in any
  contrastive training data. It scores 13/112 for Phase 1 and 11/112 at step
  9,000, both **below** the ~28 a coin flip would get: the model picks the
  same attribute regardless of which entity is named. Contrastive validation
  loss went flat at ~0.59 by step ~500, and the model began copying v1's
  fixed phrases ("Put the other way around:") into free generation. It
  learned the template, not the binding. For the anneal, v1 is replaced by
  `scripts/gen_contrastive_v2.py`:
  - 15 sentence frames instead of 1
  - pronoun and former/latter resolution
  - 9 question formats
  - narrative distractors and multi-step possession tracking
  - no wrapper phrase in more than 6.4% of documents
  - 133.5M unique tokens, so each document is seen about once
- **Routing is healthier than Phase 1.** Load CV 0.62 → 0.27, with no
  overloaded experts since step 3,000. One layer-0 expert is briefly starved
  at some checkpoints, but a different one each time, which is rotation, not
  death.

---

## Next Step: Dropping the Custom Adapter Altogether

`BubblSpaceLM` works cleanly today because it hides our custom `Transformer.forward()` signature and works around the lack of padding masks by running at `batch_size=1`. If you want to understand what it would take to package BubblSpace MoE-0.9B-A0.2B so `AutoModelForCausalLM.from_pretrained(...)` and `lm_eval --model hf` can load it natively, continue to [What It Takes to Make a Custom MoE Transformers-Native](transformer-native.md).
