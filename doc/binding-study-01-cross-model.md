# Binding study 01: ranking vs writing across public pretraining checkpoints

Status: **main study and controls complete (24–25 Nov 2026).** Preliminary: read §6 before quoting any number.

Background and motivation: [wsp-experiments.md §6.1, §7.6 and §9](wsp-experiments.md). Scripts: `scripts/binding_levels_hf.py` (scoring), `scripts/binding_study.py` (job lists), `scripts/binding_study_report.py` (table), `scripts/binding_controls_hf.py` (controls). Full table: [`benchmark/binding_levels/report.md`](../benchmark/binding_levels/report.md), raw JSON in `benchmark/binding_levels/`.

## 1. What was measured

The same cases are scored at three levels:
- **L1, single row:** each sentence alone; is the right ending ranked higher? This is how WinoGrande is officially scored. Chance is 50%.
- **L2, pair:** both twins must be right. This is Elazar et al. 2021's group scoring and our flip rule. Chance is 25%; a fixed habit scores below that.
- **L3, writing:** the model must *write* the right answer, greedily from the full vocabulary, in both twins.
  - For WinoGrande, the model is asked in a 5-shot prompt which name fills the blank.
  - For our probes, it continues the sentence.

Suites: probes v4 (112 pairs) and v5 (120 pairs, structures held out of our training data), and WinoGrande-XL validation (284 twin pairs rebuilt from adjacent rows).

**Models, 85 checkpoints:**

| Model | Checkpoints |
|---|---|
| Pythia 160M / 410M / 1B / 1.4B / 2.8B-deduped / 6.9B / 12B | 7 each, at 2 / 8 / 21 / 42 / 105 / 210 / 300B tokens |
| OLMo-2 1B and 7B | Stage 1 from 1B to ≈4T tokens; the end of stage 1; 4 points along anneal ingredient 1; the end of ingredients 2 and 3; `main` |
| Final models only | OLMo-2 13B, Qwen2.5 0.5B / 1.5B / 7B, SmolLM2 360M / 1.7B |

Machines: the A10 ran all models up to 1.4B, plus OLMo-2 1B and 7B, Qwen2.5 and SmolLM2. The H100 ran Pythia 2.8B-deduped, 6.9B and 12B, and OLMo-2 13B.

## 2. Main findings

### 2.1 Writing lags ranking everywhere, and appears only once ranking is well above chance

WinoGrande pairs. L2 = both twins ranked right; L3 = both twins written right.

| Model | Checkpoint | L2 | L3 |
|---|---|---|---|
| Pythia-160M … 12B | any, up to 300B tokens | 9–36% | **≤ 8%** |
| OLMo-2 1B | 42B / 420B / 4T / final | 19 / 26 / 36 / 35% | 1 / 4 / 4 / 3% |
| OLMo-2 7B | 42B / 105B / 1T / 2T / 3.9T | 26 / 32 / 42 / 45 / 50% | 2 / 2 / **11 / 14 / 16%** |
| OLMo-2 7B | after the anneal (`main`) | 51% | **25%** |
| Qwen2.5 0.5B / 1.5B / 7B | final | 26 / 32 / 48% | 4 / 12 / **35%** |
| SmolLM2 360M / 1.7B | final | 26 / 39% | 4 / 7% |
| OLMo-2 13B | final | 59% | **35%** |

- **L3 stays near zero until L2 reaches roughly 40%, then rises slowly.** In no model does it reach L2. Even the strongest models write the right name in both twins only 35% of the time, against about 50–60% ranked right.
- **Pythia never gets there.** No Pythia model up to 12B passes 8% at L3 within 300B tokens.

### 2.2 The OLMo-2 anneal improves writing, not ranking, and only at 7B

| OLMo-2 7B | L2 | L3 |
|---|---|---|
| End of stage 1 (3.9T tokens) | 50% | 16% |
| Anneal, first 1,000 steps (≈5B tokens) | 52% | **23%** |
| End of the three anneal runs (ingredients 1 / 2 / 3) | 51 / 50 / 49% | **20 / 27 / 23%** |
| `main` (average of the three) | 51% | **25%** |

- **At 7B:** ranking doesn't move across the anneal, but writing gains 4–11 points, most of it in the first 5B anneal tokens.
- **At 1B there's no gain:** stage-1 end L2 36% / L3 4%, `main` 35% / 3%.

This is the clearest support so far for the "goldilocks" idea in [wsp-experiments §9](wsp-experiments.md). The anneal turns an existing ranking ability into writing, but only where the ranking ability is already strong.

### 2.3 The habit: same name for both twins, first-mentioned early, second-mentioned later

When a model writes a name on WinoGrande, the "first-mentioned share" is how often that name is the first one mentioned in the sentence:

| Model @ tokens | First-mentioned share | Same name for both twins | Ranking prefers first-mentioned |
|---|---|---|---|
| Pythia-12B @ 8B | **89%** | 88% | 48% |
| OLMo-2 7B @ 9B | **79%** | 57% | 44% |
| Pythia-12B @ 300B | 23% | 90% | 49% |
| OLMo-2 7B @ 42B | 20% | **94%** | 50% |
| OLMo-2 7B @ 3.9T | 41% | 81% | 51% |
| OLMo-2 7B final | 30% | 71% | 48% |
| Qwen2.5-7B final | 31% | 61% | 51% |
| OLMo-2 13B final | 44% | **59%** | 48% |

- **Early in training:** the written name is mostly the **first-mentioned** one.
- **Later:** it's mostly the **second-mentioned** one, which is the one closer to the blank.
- **For most of training,** models write the same name for both twins in 80–96% of pairs. Only the strongest final models get below 70%.
- **The ranking never shows a position preference** (≈50% throughout). The position habit is a property of writing, not ranking.

This matches the frequency → position phases of Chan et al. 2026 (entity comparison, OLMo-2), here for in-context binding and at the writing level. The order-swap control (§5) will separate name preference from position.

### 2.4 Below-chance pairs early in training

Every model starts with L2 below 25%, the signature of a fixed preference. When L2 first reaches chance:
- **OLMo-2 7B:** ≈42B tokens.
- **Pythia 6.9B / 12B:** ≈105B.
- **OLMo-2 1B:** ≈420B.
- **Pythia ≤ 1.4B:** never within 300B tokens.

So larger models leave the habit sooner, measured in tokens.

## 3. A problem with our own probes at L3

> **Update, 25 Nov:** a re-analysis of the per-case probe scores ([binding-study-02](binding-study-02-priors-and-word-order.md)) changes the reading of BubblSpace's probe numbers below. Sensitivity to the entity is right in 96% of v4 pairs once the model's general preference for an ending is removed, and v5's low score comes mostly from passive voice being read by word order. "Familiarity, not binding" was too strong.

On probes v4 / v5, **L3 is 0–2% for every public model**, including Qwen2.5-7B and OLMo-2 13B, which reach 35% L3 on WinoGrande. Our probe's writing level is too strict. It requires the model to produce our exact endings (" mistaken.", " well informed."), which even strong models rarely write word for word.

Consequences:
1. **Probe L3 isn't a valid binding measure as built.** WinoGrande L3 is the usable writing measure for now.
2. **BubblSpace's 7/112 at probe L3 (Phase 2 final) is probably familiarity, not binding.** It's higher than every public model's, because our model trained on contrastive v2, which comes from the same generator family and phrasing. The same caveat applies to its 77–82/112 at probe L2. **v5 (held-out structures, 10/120) is the fair number for our model.**
3. **Fix to build:** score probe L3 on whether the model writes the *distinguishing word* anywhere in its continuation, or on a semantic judgement, rather than an exact token match.

## 4. Where BubblSpace stands

Measured on the same scale on 25 Nov with `scripts/binding_levels_bubbl.py` (same prompts and scoring as the public models):

| BubblSpace checkpoint | Tokens | L1 (all rows) | L2 | L3 | Same name for both twins |
|---|---|---|---|---|---|
| Phase 2 final (step 22,888) | 22B | 54.3% | 18.3% | 1.8% | 78% |
| 2.5a step 8,000 | 26B | 52.9% | 16.9% | 2.8% | 92% |

At equal tokens (≈21B), OLMo-2 1B is at L2 17% / L3 1% and Pythia 410M–1.4B at 14–15% / 1%, so Phase 2 final is slightly ahead, within noise against OLMo-2 1B. Results: `benchmark/binding_levels/bubblspace/`. Comparable public checkpoints at 42B tokens (BubblSpace has ≈42B after 2.5a):

| Model @ 42B tokens | L2 | L3 |
|---|---|---|
| OLMo-2 7B | 26% | 2% |
| OLMo-2 1B | 19% | 1% |
| Pythia 1B–12B | 12–19% | 0–4% |

## 5. Controls

For 12 checkpoints (Pythia-1B, Pythia-6.9B, OLMo-2 1B and OLMo-2 7B, each early / middle / final):
- **Prompt shuffle:** three different 5-shot prompts. Does the same-name habit survive?
- **Order swap:** the names swap places in every sentence. Does the model keep the name (frequency) or the position?

The first test (Pythia-160M @ 42B) kept the same name 74% of the time after the swap, pointing to name preference at that size.

**Results (all 12 done, 25 Nov).** "Keeps the name" = after the swap the model writes the same name as before (a name preference). "Keeps the position" = it writes whichever name now sits in the same place.

| Checkpoint | Keeps the name | Keeps the position | Same name for both twins (3 prompts) |
|---|---|---|---|
| OLMo-2 7B @ 9B / 42B / final | 72 / 52 / 57% | 28 / 48 / 43% | 45–57 / 92–95 / 71–75% |
| OLMo-2 1B @ 21B / 42B / final | 46 / 53 / 69% | 54 / 47 / 31% | 66–73 / 77–87 / 93–94% |
| Pythia-1B @ 8B / 42B / 300B | 73 / 75 / 72% | 27 / 25 / 28% | 74–88 / 73–94 / 93–94% |
| Pythia-6.9B @ 8B / 42B / 300B | 82 / 62 / 72% | 18 / 38 / 28% | 74–94 / 84–90 / 88–93% |

- **The same-name habit survives all three prompts**, so it isn't caused by one prompt's examples.
- **Name preference dominates early** (72–82% at 8–9B tokens) and in the small models throughout. A position phase shows up only in OLMo-2 around 21–42B tokens (≈50/50).
- **No clean frequency → position → correct order.** OLMo-2 1B goes from position back to name preference, and Pythia stays on names. This partly corroborates Chan et al. 2026, with differences: a different task (in-context binding), and the habit shows in writing, not ranking.
- Raw files: `benchmark/binding_levels/controls/`.

## 6. Caveats

- **Prompt dependence:** the main table uses one 5-shot prompt. The controls (§5) re-ran 12 checkpoints with three prompts, and the same-name habit held in all of them.
- **Hardware:** OLMo-2 7B ran on the A10 and the 12B/13B models on the H100. bf16 kernels differ by 1–2 cases between the machines.
- **Corrupt Hub checkpoints:** EleutherAI `pythia-2.8b` stores identical weights under different revisions (same SHA-256), so 2.8B-deduped was used instead. Every result now records weight hashes, and the report flags duplicates.
- **Sample size:** 284 WinoGrande twin pairs and 112/120 probe pairs. Differences of a few points are within noise.
- **Novelty:** not yet a citable novelty claim; see [wsp-experiments §7.6](wsp-experiments.md). The closest published work is Chan et al. 2026 (phases over training, ranking only) and Elazar et al. 2021 (pair scoring).
