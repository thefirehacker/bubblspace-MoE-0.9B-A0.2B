# Weight-space programming (WSP) on BubblSpace MoE-0.9B-A0.2B

Two attempts to fix a reasoning weakness by editing the weights directly, with no training run, on the Phase 1 final checkpoint (`step_19073`). Written as source material for a post. Everything below is from the actual runs (21 Nov 2026). Scripts: [`scripts/wsp_experiment.py`](../scripts/wsp_experiment.py) and [`scripts/wsp_closed_form.py`](../scripts/wsp_closed_form.py).

**Credit:** the second attempt uses the closed-form editing approach from [**areu01or00/qwen0.8B-wsp**](https://github.com/areu01or00/qwen0.8B-wsp), which applies ROME/MEMIT-style regularized least squares to Qwen 0.8B's MLP `down_proj` to inject facts. We adapted the same math to a different target (an attention output projection) and a different goal (strengthening a general skill rather than inserting one fact).

## TL;DR

| | Probe score (out of 100) | Target categories (0/30 at baseline) | Collateral damage | Free generation |
|---|---|---|---|---|
| Baseline (Phase 1 final) | 25 | 0/30 | — | wrong |
| **Attempt 1:** naive head amplification | 20 (**−5**) | 0/30 | spatial_size 40 → 10%, social 40 → 20% | wrong |
| **Attempt 2:** closed-form, activation-derived edit | **31 (+6)** | **10/30** | social 40 → 10%, small drops elsewhere | **still wrong** |

The closed-form edit moved a scored benchmark for the first time, but the model's actual generated text didn't get better. **What it gave us is a diagnostic: it showed where in the network a weak binding signal lives. It isn't a fix.**

## 1. The problem we tried to fix

The Phase 1 model (10B FineWeb-Edu tokens, ≈915M total / ≈211M active parameters) was fluent but bound attributes to the wrong entity. WinoGrande sat at chance (51%) for the whole run. To measure this directly, we built a probe: minimal pairs where only the named entity changes, scored by log-likelihood the same way lm-evaluation-harness scores WinoGrande.

```
The marathon took longer to finish than the sprint because the marathon was …  → "much longer in distance."
The marathon took longer to finish than the sprint because the sprint was …    → "over in seconds."
```

A case counts as correct only if the preferred completion **flips** when the entity flips. The 100-case probe (v2, 10 categories) scored **25/100**, with three categories at a flat **0%**:
- comparative value
- emotional attribution
- possession/obligation

A first version of the probe had a grammar confound ("had" vs "was" endings gave the answer away). It was caught and fixed before any of this; v1 → v2 moved agent–patient from 100% → 40% and possession from 60% → 0%.

You set the rule: no "small model capacity" explanations. If binding is broken here, scale won't fix it. Then: "for learning purpose we can now try weight space programming … can you do wsp and see if our problem can be fixed?", on a **copy** of the checkpoint (saved with a `wsp_` prefix), never on the original.

## 2. Attempt 1: amplify the heads that react to the entity swap

**Hypothesis:** a binding signal exists somewhere but is too weak to beat the model's word-frequency shortcuts at the output. Unlike a steering vector, which pushes every input the same way, amplifying a head whose output already **differs** between the two versions of a minimal pair strengthens a signal that's already input-specific.

**Method** (`wsp_experiment.py`):
1. For the 30 target-category cases, run both contexts and hook each layer's `attn.o_proj` *input*, which holds each head's output before it's mixed into the residual stream, at the last token.
2. Score each (layer, head) by `‖head_A − head_B‖ / mean(‖head_A‖, ‖head_B‖)`. That's how much the head's output changes when the entity changes, relative to its normal size, so heads that are just noisy don't win.
3. Take the **top 16 heads** (they landed across **8 layers**; the most sensitive was layer 10, head 10, at 0.619 relative change). Multiply their input columns in `o_proj.weight` by a **fixed gain of 2.0**.

**Result:** 25 → **20/100**.

| Category | Baseline | Attempt 1 |
|---|---|---|
| comparative_value (target) | 0% | 0% |
| emotional_attribution (target) | 0% | 0% |
| possession_obligation (target) | 0% | 0% |
| spatial_size | 40% | **10%** |
| social_attribution | 40% | **20%** |
| others | unchanged | unchanged |

**Why it failed:**
1. **No effect on the target.** "Reacts to the swap" isn't the same as "is the bottleneck for the right answer". Turning these heads up gave the target categories no extra leverage.
2. **Collateral damage, with nothing to prevent it.** Amplifying a head amplifies everything it does, including on cases it wasn't failing.
3. **The gain (2.0) was a guess,** not derived from anything.

**Comparison with qwen0.8B-wsp at this point:** same philosophy (edit weights directly, guided by activations), different mechanics.

| | qwen0.8B-wsp | Our attempt 1 |
|---|---|---|
| What's edited | `mlp.down_proj`, 5 of 488 tensors (layers 17–21) | `attn.o_proj` columns for 16 heads (8 layers) |
| Theory | MLPs as key-value memory (Geva et al.; ROME/MEMIT): facts live in MLP associative memory | attention circuits (IOI-style): binding is about which heads route information between entity tokens |
| How the edit is computed | closed-form regularized least squares with a **protection term** for unrelated inputs, norm-capped | a heuristic score, then a flat multiplier, no protection |
| Goal | insert one specific fact | strengthen a general skill across arbitrary minimal pairs |

Your push: "you can do better than naive scaling. You can look at activations and then make changes instead of deciding pre-task. Isn't WSP supposed to be that?" That led to attempt 2.

## 3. Attempt 2: a closed-form edit derived from activations (ROME/MEMIT-style)

The idea: let the model's own activations and gradients decide the **direction and size** of the edit, and solve for the smallest weight change that does it while explicitly protecting everything else. It's the same math as qwen0.8B-wsp, applied to one attention layer's `o_proj` instead of an MLP.

**Method** (`wsp_closed_form.py`, edit at **layer 6**):

1. **Where the output should go (positive keys and targets).** For each of the 30 target cases, under both contexts (60 keys total):
   - Run a *differentiable* version of the harness scorer and capture `o_proj`'s output `z` at the last context token, the position whose state predicts the answer.
   - Compute a **margin loss** that prefers the correct completion over the wrong one, backpropagate, and read the gradient at `z`.
   - Target value: `v* = z − η·∇`, one local step with **η = 0.3·‖z‖**, so the step size comes from the activation's own scale rather than a hand-picked constant.
   - Column of **R** = `v* − z`. Column of **Kp** = the `o_proj` input at that position.
2. **What must not change (protected keys).** The same capture, input only, for the **other 70 probe cases × 2 contexts = 140 keys**, all with target change 0. They form **Kn**.
3. **Solve the edit in closed form:**

   `ΔW = R · Kpᵀ · (λ · Kn·Knᵀ + Kp·Kpᵀ + μI)⁻¹`

   with **λ = 1.0** (how hard to protect) and **μ = 0.01 × mean diagonal of the Gram matrix** (ridge regularizer, scaled to the data).
4. Add ΔW to layer 6's `o_proj.weight`. The weight norm went **32.22 → 34.41** (+6.8%), with **‖ΔW‖ = 12.02**. Saved as a new checkpoint.

**Result:** 25 → **31/100**.

| Category | Baseline | Attempt 1 | **Attempt 2** |
|---|---|---|---|
| emotional_attribution (target) | 0% | 0% | **50%** |
| possession_obligation (target) | 0% | 0% | **30%** |
| comparative_value (target) | 0% | 0% | **20%** |
| spatial_width | 30% | 30% | 50% |
| spatial_height | 70% | 70% | 60% |
| spatial_size | 40% | 10% | 40% |
| agent_patient_causation | 40% | 40% | 40% |
| material_property | 20% | 20% | 10% |
| ability_attribution | 10% | 10% | 0% |
| social_attribution | 40% | 20% | **10%** |

All three dead categories came alive (**0/30 → 10/30**), and deriving the edit from gradients on real activations clearly beat guessing a multiplier. The protection term only partly worked: λ = 1.0 treated all 70 other cases as one blob, enough to spare spatial width/size and agent–patient, but not social attribution.

## 4. The test that mattered: free generation

A probe score only asks whether the model ranks one pre-written answer above another pre-written answer. You tested what the model **writes**, with the chat REPL on the edited checkpoint:

- **Wardrobe/doorway.** The probe calls this case *fixed* by the edit. But at temperature 0.8 the generation was "…because the doorway was usually painted white. We also know that women were expected to be in the wardrobe…". Greedy decoding gave "…the doorway was too narrow", which is right, but then repeated that sentence in a loop.
- **Marathon/sprint.** The probe says *correct in both directions* after the edit. Free generation: "…because the sprint was … longer than the marathon", which is wrong and contradicts the premise.

The model's real top choice from the full vocabulary was neither of our two candidates but a third, wrong continuation. The edit had changed which of two fixed answers scores higher, not what the model believes.

Your verdict: **"it's not a win if final output is bad."** The honest conclusion:
1. **What the edit really is:** a one-shot, closed-form stand-in for a single gradient step on 30 examples. It's too narrow to change behaviour beyond a forced choice, and not something to build later training on. The clean `step_19073` stayed the Phase 2 starting point.
2. **What it did show:** there is a weak, input-specific binding signal in the middle layers (heads in layers 5–11, editable at layer 6), weak enough that a small nudge flips a two-way comparison without changing what the model generates. That's a mechanism-level confirmation of the probe's "weak signal" finding.
3. **What it taught about evaluation:** every binding claim since needs **both** a scored probe and free generation. That became a standing rule for Phase 2.

## 5. What we did instead, and what happened

The fix moved into training data. Phase 2 added a **synthetic contrastive corpus** of minimal-pair documents targeting the zero categories ([storyboard 04](storyboard/04-Phase2-Trainig.md)).
- **Corpus v1 (1 sentence frame):** got memorized as a template. Its loss bottomed out at ≈0.59 within 500 steps, the model began copying its phrases into answers, and no probe moved.
- **Corpus v2 (15 frames, pronoun resolution, varied questions):** replaced it for the final anneal. It's being measured with **probe v4**, built only from phrasings and entities that never appear in training, so templates can't help.
- **Result so far:** probe v4 has stayed below chance (11–24 of 112, where guessing scores ≈28) through step 19,000.

So neither weight surgery nor contrastive pretraining text has fixed binding yet at this scale. It's an open problem.

## 6. Ideas we didn't run

- **Sweep λ** (2, 5, 10) to map the tradeoff between target gains and collateral damage, instead of one setting.
- **Edit toward generation, not a two-way margin.** Use a loss over the full vocabulary, with targets taken from correct *generations*, so the edit has to change what the model writes.
- **Edit several layers at once, MEMIT-style,** spreading a smaller change across layers 5–11 where the signal lives, rather than one large change at layer 6.
- **Evaluate on probe v4 and in free generation,** not on the probe whose cases the edit was solved from (attempt 2 solved on 30 probe cases and was scored on the same probe, which flatters it).
- **Try it on the Phase 2 model.** A model with a stronger underlying signal might respond very differently to the same edit.

## 7. Literature review

**Objective this reading serves:** don't assume model capacity is the problem. The model is poor at relations, so find ways to teach a *small* model relational structure, then carry what works to larger models and longer training runs.

The papers below are grouped by the question they answer for this project. Titles, authors and years are given so each one can be searched directly. Verify the details when pulling the papers.

### 7.1 The editing methods WSP builds on

| Paper | What it gives us |
|---|---|
| Meng et al. 2022, *Locating and Editing Factual Associations in GPT* (ROME) | The rank-one closed-form edit behind qwen0.8B-wsp, and causal tracing to find where a fact is stored. |
| Meng et al. 2023, *Mass-Editing Memory in a Transformer* (MEMIT) | ROME spread across several layers and thousands of edits. The regularised least-squares solve with a protection term (our attempt 2) comes from here. |
| Fang et al. 2024/25, *AlphaEdit: Null-Space Constrained Knowledge Editing* | Projects the edit into the null space of protected keys, so unrelated inputs can't change. Aimed at exactly our collateral damage (social attribution 40 → 10%). |
| Mitchell et al. 2022, *Fast Model Editing at Scale* (MEND) | A learned editor network instead of a closed-form solve. Useful contrast. |
| Yao et al. 2023, *Editing Large Language Models: Problems, Methods, and Opportunities* | Survey of the field and its standard evaluations. |

### 7.2 Why the edit moved the probe but not generation

| Paper | What it gives us |
|---|---|
| Hase et al. 2023, *Does Localization Inform Editing?* | Where causal tracing locates a fact doesn't predict which layer is best to edit. Questions our "most swap-sensitive heads / layer 6" choice. |
| Cohen et al. 2023, *Evaluating the Ripple Effects of Knowledge Editing* | Edits pass the targeted test but fail on related or rephrased cases. The same pattern as our 31/100 probe score with unchanged free generation. |
| Ilharco et al. 2023, *Editing Models with Task Arithmetic* | Weight-space edits that add or remove *skills* rather than single facts. Closest to our goal of strengthening a general skill. |
| Turner et al. 2023, *Activation Addition*; Zou et al. 2023, *Representation Engineering* | Steering activations at inference time. A cheap test of whether a direction for a relation exists before editing weights. |
| Schaeffer et al. 2023, *Are Emergent Abilities of Large Language Models a Mirage?* | The choice of metric (log-prob ranking vs generated text) can make an ability look present, absent or sudden. Why we now require both a scored probe and free generation. |

### 7.3 How transformers represent binding

| Paper | What it gives us |
|---|---|
| **Feng & Steinhardt 2024, *How do Language Models Bind Entities in Context?*** | **Start here.** Finds "binding ID" vectors that attach attributes to entities, shared across tasks. The mechanism our wardrobe/stairwell probe tests. |
| Feng et al. 2024, *Monitoring Latent World States in Language Models with Propositional Probes* | Reads (entity, attribute) bindings directly from activations. A candidate probe that needs neither log-prob ranking nor generation. |
| Hernandez et al. 2024, *Linearity of Relation Decoding in Transformer Language Models* | Many relations are well approximated by one linear map on the subject's representation. If a relation is roughly linear, a least-squares edit is the natural tool, which grounds the closed-form approach. |
| Kim & Schuster 2023, *Entity Tracking in Language Models* | Small models track entity state poorly; larger and code-trained models do better. Supports the code share in Phase 2.5a. |
| **Prakash et al. 2024, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking*** | Fine-tuning improved entity tracking by strengthening the circuit that already existed, not by building a new one. Basis for re-running WSP *after* Phase 2.5a: editing needs a formed circuit to strengthen. |
| Wang et al. 2022, *Interpretability in the Wild* (IOI circuit) | A complete circuit for a who-did-what-to-whom task in GPT-2 small. The method to copy for mapping our binding heads. |
| Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*; Geva et al. 2023, *Dissecting Recall of Factual Associations* | Facts are stored in MLPs, while attention moves and extracts them. Context for editing attention `o_proj` (our choice) versus MLP `down_proj` (qwen0.8B-wsp). |
| Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*; Olsson et al. 2022, *In-context Learning and Induction Heads* | Background on how attention heads move information between positions. |

### 7.4 Not capacity: data and format decide

| Paper | What it gives us |
|---|---|
| **Allen-Zhu & Li, *Physics of Language Models*, Part 3.1: Knowledge Storage and Extraction** | **Essential.** The same small model can or can't use a fact depending on whether pretraining contained varied rephrasings of it. Capacity is equal; data format decides. The theory behind our contrastive corpus with many sentence frames. |
| Allen-Zhu & Li, Part 3.2: Knowledge Manipulation | Comparison, classification and inverse lookup fail without the right training signal. Comparison is one of our zero categories. |
| Allen-Zhu & Li, Part 3.3: Knowledge Capacity Scaling Laws | About 2 bits of knowledge per parameter, a way to check whether a 0.9B model is really capacity-limited for a task. |
| Wang et al. 2024, *Grokked Transformers are Implicit Reasoners* | Small transformers learn composition and comparison with enough training. Comparison generalises out of distribution, composition doesn't, explained via circuits. Directly about relation types. |
| Berglund et al. 2023, *The Reversal Curse* | Training on "A is B" doesn't teach "B is A". Why our relational data states relations in both directions. |
| Eldan & Li 2023, *TinyStories*; Gunasekar et al. 2023, *Textbooks Are All You Need* (phi-1) | Very small models gain abilities from targeted data. Precedent for designing the data instead of scaling the model. |

### 7.5 Designing data that teaches relations

| Paper | What it gives us |
|---|---|
| Kaushik et al. 2020, *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data* | Minimal-pair counterfactuals stop shortcut learning. The principle behind contrastive v2/v3. |
| Gardner et al. 2020, *Evaluating Models' Local Decision Boundaries via Contrast Sets* | The evaluation side of the same idea: our flip-based probe scoring. |
| Sakaguchi et al. 2020, *WinoGrande* | Adversarial filtering to remove word-association giveaways. Relevant to the grammar confound we fixed between probe v1 and v2. |
| Yang et al. 2024, *Synthetic Continued Pretraining* (EntiGraph) | Generates text about relations between entities from a small corpus for continued pretraining. The closest published analogue to our Qwen3 relational passages. |
| Maini et al. 2024, *Rephrasing the Web* (WRAP) | Rewriting web text into styles such as Q&A helps small models learn. Connects to Physics 3.1. |

### 7.6 Small model to large runs

| Paper | What it gives us |
|---|---|
| Yang et al. 2022, *Tensor Programs V* (μP) | Hyperparameters tuned on a small model transfer to a large one. The standard tool for "test small, run big". |
| Magnusson et al. 2025, *DataDecide* | Whether data choices made with small runs hold at scale, and which metrics predict well. |
| Xie et al. 2023, *DoReMi*; Liu et al. 2024, *RegMix* | Choosing data mixtures on small proxy models, then using them at scale. Our 2.5a mixture shares are this problem. |

### 7.7 How this reading maps to our results and next steps

| Our observation | Literature that explains it | What it suggests |
|---|---|---|
| Attempt 2 fixed probe cases but not generation | Cohen et al. (ripple effects); Hase et al. (localization); Schaeffer et al. (metric choice) | Edit toward full-vocabulary targets from correct generations; score on held-out probe v5 plus free generation. |
| Collateral damage on social attribution | MEMIT, AlphaEdit | Replace the single λ blob with a null-space projection, or spread the edit across layers 5–11. |
| Weak binding signal in layers 5–11 | Feng & Steinhardt (binding IDs); Prakash et al. (fine-tuning amplifies existing circuits) | Look for binding-ID directions; repeat WSP on the 2.5a checkpoint, where training should have strengthened the circuit. |
| Contrastive v1 memorised as a template | Physics of LMs 3.1; Kaushik et al. | Many frames and held-out structures (v3, probe v5) instead of one template. |
| "Not capacity" stance | Physics of LMs 3.1–3.3; Grokked Transformers; TinyStories | Treat failures as data and signal problems first; use the capacity estimate (≈2 bits/param) as a sanity check. |
| Carrying results to larger models | μP; DataDecide; RegMix | Validate data interventions at 0.9B with metrics that are known to predict at scale. |

**Suggested reading order (about two weeks):**
1. Feng & Steinhardt, then Physics of LMs 3.1 and 3.2. The binding mechanism exists, and data decides whether it forms.
2. ROME, then MEMIT, then Hase et al. What our edit did and why it was narrow.
3. Prakash et al. and Hernandez et al. Why WSP after Phase 2.5a should work better.
4. Grokked Transformers, then Counterfactually-Augmented Data. Data design per relation type.
5. DataDecide and μP. Carrying results to larger runs.

## 8. Key numbers

| Item | Value |
|---|---|
| Model | BubblSpace MoE-0.9B-A0.2B, Phase 1 final (`step_19073`) |
| Probe | 100 minimal pairs, 10 categories, log-likelihood scoring; baseline 25/100 |
| Attempt 1 | top-16 heads (8 layers), `o_proj` columns × 2.0 → 20/100 |
| Attempt 2 | layer 6 `o_proj`; 60 positive keys, 140 protected keys; λ = 1.0, η = 0.3·‖z‖, μ = 0.01·mean diag(Gram); ‖W‖ 32.22 → 34.41, ‖ΔW‖ = 12.02 → 31/100 |
| Target categories | 0/30 → 0/30 (attempt 1) → 10/30 (attempt 2) |
| Free generation | not improved in either attempt |
| Checkpoints (A10) | `wsp_checkpoints/wsp_step19073_20261121T191046.pt`, `wsp_checkpoints/wsp_closedform_step19073_20261121T191951.pt` |

## 9. Goldilocks zone for reasoning experiments

WSP is a tool we can use to check and improve reasoning in a language model, not only a one-off edit on a finished checkpoint. The idea is to run it during the early and middle part of training, on a model that is still being pretrained, instead of on a heavily trained instruction-following model. Too early and there is no circuit to push. Too late, after instruction tuning, and the behaviour is already baked in and hard to read. We want the checkpoint that is just right: not underbaked, not overbaked.

What makes a result useful is a specific pattern. If WSP moves the probes and the evals but the actual generated text stays wrong, that is not a fix. It is a signal. We run the same experiment on the next checkpoint, one that has been trained on reasoning and relational data. If training strengthened the circuit, the same edit should show up in the output, not only in the score.

### What the initial runs showed

The first runs gave us a signal, and that signal is what the new data was built from.

WinoGrande sat at chance (about 51%) for the whole of Phase 1. That said binding was weak. It did not say which relation was broken, or whether the model was using a shortcut. A WinoGrande row is one sentence, two fill-ins, one answer. The official score can move because one option is simply the more likely word.

So we built our own set. Each case is a minimal pair: the same sentence, the named entity swapped, two different endings. A case counts only if the preferred ending flips with the entity. The first version had a grammar confound ("had" versus "was" gave the answer away). After that was removed, probe v2 scored 25/100, and three categories sat at zero: comparative value, emotional attribution, and possession/obligation.

WSP on the Phase 1 final checkpoint (`step_19073`) then split the score from the output. Attempt 2 moved the probe from 25 to 31, and those three categories from 0/30 to 10/30. Free generation stayed wrong. That is the pattern above. The probe was sharp enough to steer the data. Contrastive v2 was written at those zero categories. It carried to new words and frames (probe v4, 17 to 82/112) and did not carry to new sentence structures (probe v3, 4 to 6/70). Phase 2.5a is the next dataset for that gap: relational passages in natural prose, not another copy of the same template.

### How we plan to scale the experiment

**A. Our own training run.** Repeat the probes and the same WSP setup on later checkpoints, especially after the relational and reasoning data in Phase 2.5a. Same edit, same probe, plus free generation. If the score moves and the text still does not, we go to the next checkpoint rather than tuning the edit harder on a model that has not learned the relation yet.

**B. Unannealed checkpoints from other open models.** Run the same evals and the same WSP experiment on public checkpoints taken before the learning-rate anneal, while the model is still in that goldilocks part of training. Instruction-tuned releases are the wrong target. We want the base checkpoint where a binding circuit can exist and has not yet been covered over by chat tuning.
