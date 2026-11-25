# Plan 05: bits-per-byte eval on post-cutoff text (scope, 25 Nov 2026)

Status: **scoped, not built.** Waiting for the user's go.

## 1. Why

Our W&B validation losses are per token, measured with our own tokenizer and on held-out slices of our own training pools. That leaves three gaps:
1. **Not comparable across models.** A per-token loss depends on the tokenizer, so we can't compare ours with SmolLM2, OLMo or Qwen.
2. **Not fully outside the training distribution.** The held-out tails come from the same sources we train on.
3. **Not contamination-proof.** Benchmarks can leak into web data.

**Bits per byte (BPB) on text published after all our training data** closes all three. BPB is total −log₂ p over a document divided by the document's UTF-8 byte count. Because it counts bytes, not tokens, it doesn't depend on the tokenizer, and text that didn't exist at training time can't have been memorised. Compression tracks downstream ability fairly closely (Huang et al. 2024, *Compression Represents Intelligence Linearly*). Elie Bakouch's [Pre-training Index v0](https://eliebak.com/pretraining-index) (Nov 2026) ranks open models this way.

What it gives us:
- **A number comparable with other models** for BubblSpace at each phase.
- **A clean read on 2.5a, its decay and 2.5b** on code and writing the model has never seen. This feeds the decay-mix decision in [plan-03 §2.7](plan-03-phase-2.5-knowledge-context.md).
- **A forgetting signal per category** (code, papers, news, fiction, Wikipedia) that our English 6 tasks don't cover. Fiction in particular is where LAMBADA and HellaSwag have been slipping.

## 2. Data

| Item | Choice |
|---|---|
| Dataset | `Jellyfish042/UncheatableEval-2026-10` on Hugging Face: 7,500 documents published in Oct 2026, one parquet file (231 MB). Columns: `content` (truncated), `untruncated_content`, `category`, `date`, `url`. |
| Link to the index | This is the index's "Uncheatable Eval" half, 50% of its score. The control set there is "77 of 7,500", so the document count matches. Index categories: GitHub code, arXiv & bioRxiv, BBC News, AO3 fiction, Wikipedia & READMEs. The index's other 50% (277 hand-picked sources) isn't published, so **we can match only this half, not its headline number.** |
| Post-cutoff for us? | Yes. Our newest sources are `wikimedia/wikipedia` 20231101.en, StarCoderData (2023), FineWeb-Edu, and 2025 synthetic sets (Nemotron Fact-Seeking, SYNTH, our Qwen3 passages). Synthetic text can carry knowledge only up to its generator's cutoff, which is well before Oct 2026. |
| Security | Parquet (no pickle). HF shows no scan result for the file, so run **ClamAV** before use (project rule). |
| License | The card states none. **Use it for evaluation only:** don't redistribute it, don't train on it, and add its URLs and 13-gram hashes to the decontamination blocklist so later data (Phase 3 SFT scraping) can't pull these documents in. |
| Later releases | The maintainer publishes every ≈3 months (2026-01, -04, -07, -10). Re-run on each new release to keep the test after-cutoff. |

## 3. Method

- **Per-document BPB** = Σ(−log₂ p(tᵢ)) ÷ bytes(UTF-8 of the scored text). The first token is scored after a document-start token (our `<|endoftext|>`; each HF model's BOS or EOS).
- **Text:** the `content` column, the version the UncheatableEval script scores. Check this against the maintainer's GitHub code (`Jellyfish042/uncheatable_eval`) before the first run, so our numbers line up with theirs.
- **Long documents:** a sliding window of 2,048 tokens with stride 1,024. Each scored token gets at least 1,024 tokens of context, and every token is scored exactly once. **Every model uses the same window**, including models with longer context, otherwise context length becomes the thing measured. A second pass at 16,384 (for 2.5b) is reported separately, because longer context lowers BPB on its own.
- **Report:**
  - BPB per category and overall;
  - two ways of averaging: byte-weighted (total bits ÷ total bytes) and mean over documents. The index's page doesn't state which it uses.
  - per model, a "win rate": the share of documents where it beats a reference model.
- **Precision:** bf16 forward with fp32 log-softmax, the same as `binding_levels_*`. Deterministic, so reruns match.
- **Memorisation check:** none needed for this post-cutoff set. The index's r-score (loss ÷ difficulty × skill) is only useful for older text, so it's left out of v1.

## 4. Models

| Group | Checkpoints |
|---|---|
| BubblSpace | Phase 1 final (19,073), Phase 2 final (22,888), 2.5a full-tier checkpoints (8k / 16k / 24k / 32k / 38,147), 2.5b final; later Phase 3 and Phase 4 |
| Small public base models, same harness | SmolLM2-135M / 360M / 1.7B, Qwen2.5-0.5B / 1.5B, OLMo-2-0425-1B, Pythia-410M / 1B |
| The index's models | Not re-run. Their numbers include the unpublished half, so they're context only, not a comparison. |

The small public models are the fair comparison: similar size, same window, same harness. Pythia (≈300B tokens, 2023) and OLMo-2 1B (4T tokens) bracket how much training data matters at our size.

## 5. Build

| Piece | Work |
|---|---|
| `scripts/eval_bpb.py` | ≈150 lines. Reuses the model wrappers already written: `BubblLM` from `scripts/binding_levels_bubbl.py` for our checkpoints, `LM` from `scripts/binding_levels_hf.py` for HF models, including selective download, ClamAV for `.bin` and weight hashes. Adds parquet loading, sliding-window scoring and the per-category report. |
| Checks | Running on `content` vs `untruncated_content`; that the window covers every token exactly once; and a cross-check against UncheatableEval's own script on one small HF model (expect agreement to about 0.001 BPB). |
| Watcher | Add to the **full tier** of `scripts/eval_watch_25a.sh` / `_25b.sh` for future checkpoints. The running watcher script isn't edited mid-run; past checkpoints get a one-off queue under the GPU lock (`/tmp/a10_gpu.lock`). |
| Summary | Add a BPB column to `scripts/summarize_evals.py`, plus `benchmark/bpb/report.md` with the per-category table across models. |

## 6. Cost and time

- **Tokens:** `content` is at most ≈235 MB of the 470 MB dataset, likely less after truncation. That's roughly 25–60M tokens with our tokenizer, to be measured on download.
- **A10 time:**
  - our model, forward pass only: ≈15–40 min per checkpoint at the 2,048 window;
  - 0.1–0.5B public models: similar;
  - 1.5–1.7B models: about 2× that.
  - The whole first round (≈8 of our checkpoints + 9 public models) is roughly 8–12 h of A10 time, run between eval-watcher jobs.
- **Money:** nothing extra. The A10 is already running. No H100 time.
- **Build time:** about half a day, including the cross-check.

## 7. Risks and caveats

- **Distribution mismatch:** the eval is heavy on GitHub code, arXiv, news, fiction and Wikipedia, while our training mix has no fiction or news. Expect our BPB to look worst on AO3 and BBC. That's a real gap, not an artifact, but read per category before reading the overall number.
- **Model size dominates.** A 0.2B-active model will lose to 1.5B models on raw BPB. The useful comparisons are against size-matched models (SmolLM2-360M, Qwen2.5-0.5B, Pythia-410M) and our own checkpoints over time.
- **Truncation:** `content` is cut by the dataset's own rule. Using the same column for every model keeps the comparison fair.
- **Not the index's headline number.** Say "UncheatableEval-2026-10 BPB" in any write-up, not "Pre-training Index score".

## 8. Decisions for the user

1. Go / no-go on building `scripts/eval_bpb.py` and the first round (§4) on the A10.
2. The list of public models: the 9 above, or fewer to finish sooner.
3. Whether to add the decontamination blocklist entry now (recommended, cheap).
