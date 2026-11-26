# Binding study 02: the model does bind; priors and a word-order rule hide it

Status: **re-analysis 25 Nov; A10 test results 26 Nov (§6.1). Preliminary.** §2–§4 use saved results only; §3.3 and §6.1 use new runs.

Related: [binding-study-01-cross-model.md](binding-study-01-cross-model.md) (public checkpoints), [wsp-experiments.md](wsp-experiments.md) (background), [plan-03 §2.7](plan-03-phase-2.5-knowledge-context.md) (2.5a decay mix). Script: `scripts/probe_prior_free.py`. Inputs: `benchmark/phase2/probe_v*_*.json`, `benchmark/phase25a/probe_v*_phase25a_step*.json`.

## 1. Summary

Until today we read the probe and WinoGrande results as "the model doesn't bind; it only copies templates". The raw per-case scores say otherwise:

1. **The model does bind.** When the entity in a sentence changes, its preference moves toward the right ending in 96% of probe v4 pairs (Phase 2 final). Phase 1 final was at 66%, so the contrastive data in Phase 2 created this.
2. **2.5a hasn't damaged it.** The fall on probe v4 from 77 to 38 is a shift in which ending the model prefers in general. Direction is still right in 96% of pairs at every 2.5a checkpoint.
3. **The held-out probe (v5) failed for two specific reasons, not for lack of binding:**
   - **Passive voice is read with a word-order rule.** In 39–40 of 40 pairs, the model confidently gives the reversed answer. Our contrastive data contains no passive sentences, so "first-mentioned name = the doer" was true 100% of the time in training.
   - **Priors hide the rest.** On relative clauses the direction is right in 40/40 pairs, but the pair rule scores 0–1/40 because the model prefers one ending in both contexts.
4. **The binding signal is weaker than the model's general preferences** (for an ending, for a name). The pair rule and written answers need the signal to be larger than those preferences, so both fail. The signal is also getting smaller during 2.5a.

5. **Tested on 26 Nov (§6.1):** fixing the word-order confound in the data, with the same steps and settings as a control run, turned passive from 40/40 reversed into 40/40 right, in ranking and in written answers. It also lifted a held-out structure (relative clauses, 1 → 33 of 40 written right). It didn't help every structure ("respectively", object relatives). The control run changed nothing.

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

Re-scored on 25–26 Nov with per-row margins saved. For a twin pair, the two margins (score of the correct option minus the wrong one) are added. A general preference for one name adds to one twin's margin and subtracts from the other's, so the sum is positive only if the differing word moves the preference the right way. Chance is 50%; with 284 pairs, ±3 points is one standard error.

| Model | Tokens | Pair rule (L2) | Prior-free | Median effect (nats) | Written, both twins (L3) |
|---|---|---|---|---|---|
| **BubblSpace, Phase 2 final** | 22B | 18.3% | **57.4%** | +0.12 | 1.8% |
| BubblSpace, 2.5a step 8,000 | 26B | 16.9% | 57.0% | +0.13 | 2.8% |
| OLMo-2 1B | 21B | 16.6% | **49.6%** | −0.01 | 1.1% |
| Pythia-1B | 21B | 14.4% | 59.5% | +0.08 | 1.1% |
| Pythia-1B, final | 300B | 15.1% | 59.2% | +0.16 | 1.8% |
| SmolLM2-360M, final | 4T | 25.7% | 59.2% | +0.30 | 3.9% |
| Qwen2.5-1.5B, final | 18T | 31.7% | 70.8% | +1.11 | 12.0% |
| **OLMo-2 1B, final** | 4T | 34.5% | **71.8%** | +0.75 | **3.2%** |

- **Below-chance pair scores hide a real signal, in public models too.** BubblSpace and Pythia-1B are at 57–60% prior-free while their pair scores are 14–18%.
- **At equal tokens BubblSpace has a signal that OLMo-2 1B doesn't yet:** 57% against 50% at 21–22B.
- **A strong signal doesn't produce correct written answers by itself.** OLMo-2 1B final reaches 72% prior-free but writes the right answer in both twins only 3% of the time, and writes the same name for both twins in 94% of pairs. Its preferences decide what it writes. This is Cause B (§4) measured on a public model.

Results: `benchmark/binding_levels/scores/`.

## 4. Why binding looked absent

**Cause A: we taught a word-order rule.** `scripts/gen_contrastive_v3.py` asserts that no training document contains passive voice, so that passive stays a clean held-out structure (v2 has no passive frames either). The side effect is that in all contrastive data the first-mentioned name is the doer. That rule costs the model nothing and is always right in training. The held-out passive probe then measures the shortcut we built in. This is shortcut learning from a confound in the data (Geirhos et al. 2020; McCoy et al. 2019), not a failure to bind.

**Cause B: general preferences are larger than the binding signal.**
- The pair rule needs the signal to beat the model's preference for an ending. WinoGrande's ranking needs it to beat the preference for a name. Greedy writing needs it to beat both.
- That explains "same name for both twins" and "writing lags ranking" in study 01 as the same effect. §3.3 supports it: OLMo-2 1B final has a strong signal (72% prior-free) but writes the right answer in both twins 3% of the time.
- The signal shrinks when the data that built it is reduced: 2.5a has no contrastive v2, and v3's share is 4.5% at a high learning rate.

**Cause C (inference, not measured): binding decides very little of the training loss.** Contrastive data is 1.5–6% of tokens, and only the few answer tokens in each document depend on binding. In natural text almost no next token depends on resolving who did what. Frequency-based preferences reduce far more loss, so they're learned first and stay larger. This matches the "frequency first" phase of Chan et al. 2026.

## 5. What this corrects

| Earlier statement | Corrected |
|---|---|
| "The model doesn't bind; it copies templates." | It binds (96% direction on held-out names and phrasings; 91% on two held-out structures). Template copying in free generation is real, but it isn't the whole story. |
| "Probe v5 (10/120) is the honest number: no structural transfer." | v5's pair score is dominated by passive (a word-order rule, 40 cases) and by ending priors. Without passive, direction is right in 91%. |
| "2.5a hurt binding (v4: 77 → 38)." | Direction is unchanged at 96%. The effect is smaller and the ending priors moved. |
| "Our contrastive data isn't making an impact." | It took prior-free sensitivity from 66% to 96%. Its flaw is the passive ban, plus too little weight to beat the priors. |
| "Below-chance pair scores are a habit, not binding." (study 01) | It's a prior on top of a real signal, for our model and for Pythia-1B and OLMo-2 1B (§3.3). |

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

**Results (26 Nov).** Probes, pair rule / prior-free:

| Probe and structure | In fix-arm training? | Base (Phase 2 final) | Control (v3) | **Fix (v3 + order alternation)** |
|---|---|---|---|---|
| v5 passive | passive yes; these exact sentences no | 0/40 / 0/40 (39 reversed) | 0/40 / 0/40 (40 reversed) | **40/40 / 40/40** |
| v6 passive, held-out frame | passive yes; frame no | 0/40 / 0/40 | 0/40 / 0/40 | **39/40 / 40/40** |
| v5 relative clause | **no** (held out of both arms) | 1/40 / 40/40 | 0/40 / 40/40 | **35/40 / 40/40** |
| v5 "respectively" | no | 9/40 / 33/40 | 8/40 / 37/40 | 12/40 / 37/40 |
| v6 object relative ("Y, whom X forgave, was") | no | 24/40 / 39/40 | 28/40 / 39/40 | 22/40 / 40/40 |
| v6 inverse fit, held-out frame | inverse fit yes; frame no | 40/40 / 40/40 | 35/40 / 40/40 | 40/40 / 40/40 |
| v4 (held-out phrasings, structures seen) | – | 77/112 / 108/112 | 50/112 / 105/112 | 45/112 / 105/112 |

**Written answers, judged by role.** The exact-match check only accepts the probe's own ending, and the model often writes the right role in other words: "a beginner" for the learner, "experienced" for the teacher. So written answers were also scored by which of the generator's two attribute lists (receiver's or doer's) the first written attribute comes from:

| Structure | Base | Control | **Fix** |
|---|---|---|---|
| v5 passive, both twins right | 0/40 (38 reversed) | 0/40 (39 reversed) | **40/40** |
| v6 passive, held-out frame | 0/40 (22 reversed) | 0/40 (23 reversed) | **40/40** |
| v5 relative clause | 1/40 (38 same role in both) | 0/40 | **33/40** |
| v6 object relative | 5/40 | 5/40 | 11/40 (22 unlisted words) |

(Exact-match flip on v5 / v6: base 0 / 41, control 0 / 37, fix 8 / 43 of 120.)

**Damage check:** both arms broke general language modelling the same way. LAMBADA fell from 34.9% to 5.2% (perplexity from 33 to 12,000–19,000), PIQA from 67.8 to 60–62, and the model stopped answering WinoGrande in the expected format (85% → 26–34%). That's expected from 2.5M tokens of nothing but 34-token synthetic documents at a learning rate of 1e-4. The damage is equal in both arms, so the control-vs-fix comparison stands. WinoGrande numbers after the fine-tune aren't meaningful, and this recipe is a test, not a training method.

**What it shows:**
1. **The word-order confound was the cause of the passive failure.** The same number of steps on contrastive v3 changes nothing (still 40/40 reversed). Adding receiver-first sentences flips passive to 40/40, both in ranking and in **written answers**.
2. **It carries over beyond the exact training form:** to passive sentences in frames never trained (39/40 ranked, 40/40 written), and to **relative clauses, a structure held out of both arms** (0 → 35/40 ranked, 1 → 33/40 written). One caveat: the relative-clause probe's second context is itself passive ("Emil was tutored by Hana, who was…"), so part of this gain may come from reading passive correctly.
3. **It does not carry over to everything.** "Respectively" barely moves (9 → 12/40), and object relatives don't improve. The model learned "passive means the receiver comes first", not a general rule for mapping any syntax to roles.
4. **Writing follows ranking once the signal is large.** Where the fix made the effect large (+13 nats), written answers were right in both twins in 40/40. Where the effect stayed small (object relatives, +2 nats), writing didn't follow. That supports Cause B: correct answers need a signal bigger than the preferences, not a bigger model.

**Reading the result:**
- **Passive fixed and held-out structures up, in the fix arm only:** the confound was the problem. The corrected data goes to the user for review, as a candidate for the 2.5a decay and Phase 3.
- **Passive fixed but no transfer:** the model learns each structure separately. The next lever is the number of distinct structures.
- **No change in written answers in either arm:** the signal still doesn't beat the priors. The next levers are loss weight on the answer and a paired-twin reward (Phase 4).

### 6.2 Other fixes

- **Scoring:** report the prior-free metric next to the pair rule in the eval watchers and `scripts/summarize_evals.py`. Keep both; they answer different questions.
- **More loss on binding:** continuations where several sentences depend on the resolution, or a higher loss weight on answer tokens. Test after §6.1.
- **2.5a decay mix (proposal, awaiting the user's approval):** replace contrastive v3 in the decay stage with the fix-arm data (v3 + order alternation), at the same 2% or slightly more. In real training it's a small share mixed with everything else, so the damage seen in the fine-tune doesn't apply. It needs a full-size corpus (tokenized shards), the user's data review, and one more stop-and-resume before step 32,424 (≈Fri 27 Nov 06:15–07:00 IST).
- **Next data generation (v5 of the generator):** order alternation for every structure, not just passive and fit (object relatives, "respectively" with both orders, clefts), since point 3 shows each structure has to be covered.
- **Probe scoring:** add the role-based written-answer check (above) to `scripts/eval_binding_generation.py`. Exact match understates writing.
- **Study 01:** re-run a few public checkpoints with scores saved, to see whether their below-chance pair scores are also a prior over a real signal.

## 7. Caveats

- **Direction only.** The prior-free metric doesn't say the model would answer correctly; the pair rule and the writing check do, and those are still low.
- **Small sets.** 112 and 120 pairs, built from the training vocabulary by our own generator. No outside benchmark has been re-scored this way yet.
- **No public-model baseline.** We don't know what OLMo-2 or Pythia score on the prior-free metric, so this isn't a comparison claim.
- **Cause C is an inference.** Nothing here measures how much of the loss depends on binding.
- **Why passive fails only in the "because" frame is open.** In "Emil was tutored by Hana, who was…", the direction is right in 40/40. The model seems to get the role from the adjacent "by Hana", but falls back on mention order when it has to look a re-mentioned name up.
