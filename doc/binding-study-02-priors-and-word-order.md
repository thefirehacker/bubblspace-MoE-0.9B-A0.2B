# Binding study 02: the model does bind; priors and a word-order rule hide it

Status: **re-analysis of saved results, 25 Nov 2026. Preliminary.** No new model runs went into §2–§4. The A10 test in §6 is the follow-up.

Related: [binding-study-01-cross-model.md](binding-study-01-cross-model.md) (public checkpoints), [wsp-experiments.md](wsp-experiments.md) (background), [plan-03 §2.7](plan-03-phase-2.5-knowledge-context.md) (2.5a decay mix). Script: `scripts/probe_prior_free.py`. Inputs: `benchmark/phase2/probe_v*_*.json`, `benchmark/phase25a/probe_v*_phase25a_step*.json`.

## 1. Summary

Until today we read the probe and WinoGrande results as "the model doesn't bind; it only copies templates". The raw per-case scores say otherwise:

1. **The model does bind.** When the entity in a sentence changes, its preference moves toward the right ending in 96% of probe v4 pairs (Phase 2 final). Phase 1 final was at 66%, so the contrastive data in Phase 2 created this.
2. **2.5a hasn't damaged it.** The fall on probe v4 from 77 to 38 is a shift in which ending the model prefers in general. Direction is still right in 96% of pairs at every 2.5a checkpoint.
3. **The held-out probe (v5) failed for two specific reasons, not for lack of binding:**
   - **Passive voice is read with a word-order rule.** In 39–40 of 40 pairs, the model confidently gives the reversed answer. Our contrastive data contains no passive sentences, so "first-mentioned name = the doer" was true 100% of the time in training.
   - **Priors hide the rest.** On relative clauses the direction is right in 40/40 pairs, but the pair rule scores 0–1/40 because the model prefers one ending in both contexts.
4. **The binding signal is weaker than the model's general preferences** (for an ending, for a name). The pair rule and written answers need the signal to be larger than those preferences, so both fail. The signal is also getting smaller during 2.5a.

None of these is a model-size argument. Each one points to a change in data or scoring.

## 2. Method

`scripts/probe_entity_binding.py` saves four log-probabilities per minimal pair: context A with endings c1 and c2, and context B with c1 and c2. A should prefer c1; B should prefer c2.

| Metric | Rule | What a general preference does to it |
|---|---|---|
| **Pair rule** (used so far) | A prefers c1 **and** B prefers c2 | A model that likes one ending in general fails the pair, even if the context moves it the right way |
| **Prior-free** (new) | (A_c1 − A_c2) − (B_c1 − B_c2) > 0 | A general preference for one ending or one entity cancels out. Chance is 50%. |

The prior-free metric checks direction only, so it's a weaker bar than the pair rule. Both are reported, with the median size of the effect in nats. The idea isn't new: it's the same correction as calibrating out a prompt's label bias (Zhao et al. 2021) or scoring by pointwise mutual information (Holtzman et al. 2021), applied to minimal pairs.

Failure types under the pair rule:
- **Reversed:** both contexts prefer the wrong ending. This is a consistent wrong rule.
- **Same ending:** both contexts prefer the same ending. This is a prior, or the context is ignored.

## 3. Results

### 3.1 Probe v4 (held-out phrasings, entity pairs and names; structures seen in training)

| Checkpoint | Pair rule | Prior-free | Median effect (nats): ability / comparative / emotional / fit / possession |
|---|---|---|---|
| Phase 1 final | 13/112 | 74/112 (66%) | −0.1 / +0.1 / +0.8 / +2.0 / +0.4 |
| **Phase 2 final** | 77/112 | **108/112 (96%)** | +13.0 / +12.9 / +6.5 / +9.2 / +9.5 |
| 2.5a step 2,000 | 59/112 | 109/112 | +6.7 / +10.6 / +2.2 / +8.3 / +5.6 |
| 2.5a step 4,000 | 42/112 | 110/112 | +7.3 / +11.6 / +2.1 / +4.3 / +3.2 |
| 2.5a step 6,000 | 41/112 | 108/112 | +4.7 / +10.3 / +1.6 / +5.1 / +2.6 |
| **2.5a step 8,000** | 38/112 | **108/112 (96%)** | +2.9 / +12.5 / +1.4 / +4.9 / +2.3 |

- **Phase 2's contrastive data created the sensitivity** (66% → 96%), with large effects, on names and phrasings the model never trained on.
- **2.5a kept the direction but is losing the size of the effect** in every family except comparatives, which contrastive v3 covers heavily. All of the pair-rule failures at step 8,000 are "same ending" (74 of 74), none reversed: the model's general preference for an ending now outweighs a smaller signal.

### 3.2 Probe v5 (structures held out of all contrastive data)

| Checkpoint | Passive: pair / prior-free / effect | Relative clause | "Respectively" |
|---|---|---|---|
| Phase 2 final | 0/40 / **0/40** / −11.5 | 1/40 / **40/40** / +4.4 | 9/40 / 33/40 / +0.6 |
| 2.5a step 8,000 | 0/40 / **0/40** / −12.4 | 0/40 / **40/40** / +3.6 | 8/40 / 30/40 / +0.9 |

- **Passive:** 39 of 40 pairs are *reversed* at Phase 2 final and 40 of 40 at step 8,000, by a wide margin. For "Aarav was forgiven by Pia because Aarav was…", the model prefers "big-hearted" (the forgiver's property). It reads the first-mentioned name as the doer.
- **Relative clause:** the direction is right in every pair. The pair rule fails because the model prefers the same ending (context A's) in both contexts.
- **"Respectively":** the direction is right in 75–83% of pairs, but the effect is small.
- **Without passive, held-out structures are at 73/80 (91%) prior-free** at Phase 2 final. Binding does transfer to new structures. The earlier reading, "v5 at 10/120 means no structural transfer", was wrong.

### 3.3 WinoGrande

Not yet re-analysed: the saved rows hold only right/wrong, not the scores. What the saved numbers allow:

| Checkpoint | Twins given different answers | of those, both right / both wrong |
|---|---|---|
| Phase 2 final | 31% of 284 pairs | 52 / 37 |
| 2.5a step 8,000 | 29% | 48 / 35 |

When the model does tell the twins apart, it's right more often than wrong (58%), which is a hint of signal under a name preference. It isn't significant on its own. Re-running with scores saved (§6) settles it.

## 4. Why binding looked absent

**Cause A: we taught a word-order rule.** `scripts/gen_contrastive_v3.py` asserts that no training document contains passive voice, so that passive stays a clean held-out structure (v2 has no passive frames either). The side effect is that in all contrastive data the first-mentioned name is the doer. That rule costs the model nothing and is always right in training. The held-out passive probe then measures the shortcut we built in. This is shortcut learning from a confound in the data (Geirhos et al. 2020; McCoy et al. 2019), not a failure to bind.

**Cause B: general preferences are larger than the binding signal.**
- The pair rule needs the signal to beat the model's preference for an ending. WinoGrande's ranking needs it to beat the preference for a name. Greedy writing needs it to beat both.
- That would explain "same name for both twins" and "writing lags ranking" in study 01 as the same effect. It's a hypothesis until WinoGrande is re-scored.
- The signal shrinks when the data that built it is reduced: 2.5a has no contrastive v2, and v3's share is 4.5% at a high learning rate.

**Cause C (inference, not measured): binding decides very little of the training loss.** Contrastive data is 1.5–6% of tokens, and only the few answer tokens in each document depend on binding. In natural text almost no next token depends on resolving who did what. Frequency-based preferences reduce far more loss, so they're learned first and stay larger. This matches the "frequency first" phase of Chan et al. 2026.

## 5. What this corrects

| Earlier statement | Corrected |
|---|---|
| "The model doesn't bind; it copies templates." | It binds (96% direction on held-out names and phrasings; 91% on two held-out structures). Template copying in free generation is real, but it isn't the whole story. |
| "Probe v5 (10/120) is the honest number: no structural transfer." | v5's pair score is dominated by passive (a word-order rule, 40 cases) and by ending priors. Without passive, direction is right in 91%. |
| "2.5a hurt binding (v4: 77 → 38)." | Direction is unchanged at 96%. The effect is smaller and the ending priors moved. |
| "Our contrastive data isn't making an impact." | It took prior-free sensitivity from 66% to 96%. Its flaw is the passive ban, plus too little weight to beat the priors. |
| "Below-chance pair scores are a habit, not binding." (study 01) | For our probes, "habit" means a prior on top of a real signal. Whether that also holds for the public models needs their scores re-saved. |

Still true: the model rarely *writes* the right answer in both twins, and WinoGrande pair scores are below chance.

## 6. Next steps

### 6.1 A10 test (approved 25 Nov; no H100 time, 2.5a untouched)

**Question:** does removing the word-order confound fix the passive reversal, and does that carry over to structures the model never trained on?

**Built and started on 25 Nov** (all on the A10, every GPU job under the shared lock):

1. **WinoGrande re-scored with per-row margins saved** (`scripts/binding_levels_hf.py` now stores them and reports `prior_free_pairs`). Queued for Phase 2 final and 2.5a step 8,000, plus OLMo-2 1B (21B tokens and final), Pythia-1B (21B and final), SmolLM2-360M and Qwen2.5-1.5B. Output: `benchmark/binding_levels/scores/`.
2. **Corrected data:** `scripts/gen_contrastive_v4.py`, a test version with two arms from the same generator and seed, 100,000 documents each.
   - **Control:** exactly contrastive v3.
   - **Fix:** v3 with order alternation. Half of the eligible events are written receiver-first: "Y was forgiven by X" for people, and "The lift could not take the sofa" for physical fit. That is 10.5% passive and 9.1% inverse-fit documents. Nothing else changes.
   - **Ending balance needed no change.** The generator already picks at random which of the two entities a document is about, so each ending is the true one half the time. The one confound was mention order.
3. **Probe v6** (`probefinding-winogrande-v6-order-20261125.json`, 120 pairs, training names), on structures neither arm trains on:
   - passive with a held-out frame;
   - object relatives ("Y, whom X forgave, was…" against "X, who forgave Y, was…"): the receiver comes first with no passive verb;
   - inverse fit with a held-out frame.

   Relative clauses and "respectively" (probe v5) stay held out of both arms. The fix arm also skips any document containing a probe-v5 passive context word for word.
4. **Fine-tune:** `scripts/binding_finetune_test.py`, from Phase 2 final, 400 steps of 192 documents (≈2.5M tokens), learning rate 1e-4 decaying to 6e-6, with the project's Muon + AdamW. One document per row, so no document sees another.
5. **Scoring** (`scripts/run_binding_test.sh`), for the base checkpoint and both arms: probes v4 / v5 / v6 with the pair rule and the prior-free metric, written answers on v5 / v6, WinoGrande at three levels with margins, and LAMBADA + PIQA as a damage check. Results: `benchmark/binding_test/`.

**Reading the result:**
- **Passive fixed and held-out structures up, in the fix arm only:** the confound was the problem. The corrected data goes to the user for review, as a candidate for the 2.5a decay and Phase 3.
- **Passive fixed but no transfer:** the model learns each structure separately. The next lever is the number of distinct structures.
- **No change in written answers in either arm:** the signal still doesn't beat the priors. The next levers are loss weight on the answer and a paired-twin reward (Phase 4).

### 6.2 Other fixes

- **Scoring:** report the prior-free metric next to the pair rule in the eval watchers and `scripts/summarize_evals.py`. Keep both; they answer different questions.
- **More loss on binding:** continuations where several sentences depend on the resolution, or a higher loss weight on answer tokens. Test after §6.1.
- **2.5a decay mix:** contrastive v3 is set to 2% for the decay ([plan-03 §2.7](plan-03-phase-2.5-knowledge-context.md)). Given the shrinking effect sizes in §3.1, revisit that once the A10 test is in. Any change needs the user's approval before the decay starts (step 32,424, ≈Fri 27 Nov 07:00 IST).
- **Study 01:** re-run a few public checkpoints with scores saved, to see whether their below-chance pair scores are also a prior over a real signal.

## 7. Caveats

- **Direction only.** The prior-free metric doesn't say the model would answer correctly; the pair rule and the writing check do, and those are still low.
- **Small sets.** 112 and 120 pairs, built from the training vocabulary by our own generator. No outside benchmark has been re-scored this way yet.
- **No public-model baseline.** We don't know what OLMo-2 or Pythia score on the prior-free metric, so this isn't a comparison claim.
- **Cause C is an inference.** Nothing here measures how much of the loss depends on binding.
- **Why passive fails only in the "because" frame is open.** In "Emil was tutored by Hana, who was…", the direction is right in 40/40. The model seems to get the role from the adjacent "by Hana", but falls back on mention order when it has to look a re-mentioned name up.
