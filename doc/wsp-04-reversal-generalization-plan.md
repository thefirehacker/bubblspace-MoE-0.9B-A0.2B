# WSP for generalization: can a small edit teach the model to look both ways?

Status: **plan, 26 Nov 2026.** Nothing below has been run yet. Companion to [wsp-experiments.md](wsp-experiments.md) and [binding-study-02](binding-study-02-priors-and-word-order.md).

## 1. Objective

A **small weight edit** (start at < 0.1 % of weights, ceiling 0.5 %, hard limit 1 %) that makes the model answer facts in the reverse direction **for facts the edit never saw**.

- Forward: "New Delhi is the capital of" → India.
- Reverse: "The capital of India is" → New Delhi.

**Win condition (either one):**
1. **The model generalizes.** Reverse accuracy rises on held-out facts, and ideally on held-out relations too, with no damage to forward facts, LAMBADA or PIQA.
2. **We learn something fundamental** about where and how this model stores and retrieves relational knowledge, measured, not assumed.

**Not a win:** editing individual facts. Every edit is scored only on facts and relations it was not solved from.

## 2. The working assumption

If the model can generalize, the machinery is somewhere inside it. A reverse query needs three steps (Geva et al. 2023, *Dissecting Recall of Factual Associations*):

```
"The capital of  India  is"  →  ?
                  │               │
   (1) STORE:     │               │  the representation of "India" must contain "capital = New Delhi"
       early–mid MLPs/experts at the subject token write the subject's attributes
                  │               │
   (2) ROUTE:     the relation ("capital of") must reach the last token
                                  │
   (3) EXTRACT:   attention heads at the last token read the matching attribute from "India"
                                  │
   (4) OUTPUT:    late layers must not overwrite it with a habit ("a city…", copying "India")
```

Steps 2–4 are **shared by every fact of a relation**. They are not per-fact memories. An edit to them generalizes by construction. Step 1 is per-fact storage.

So the research question is precise: **at which step does the reverse query fail?** That decides what kind of edit can work:

| Hypothesis | Where the reverse fails | Small edit that would generalize |
|---|---|---|
| **H1 storage** | "India" does not carry "New Delhi" | No direct fix. Test H5 first. |
| **H2 extraction** | Stored, but the last-token heads don't pull it out for this relation | Low-rank edit to the extraction heads (attention value/output, or query/key) at the layers found by patching |
| **H3 override** | Extracted in mid layers, then overwritten late by a habit or copy bias | Small edit to the late layers or a steering direction that suppresses the habit |
| **H4 routing (MoE-specific)** | Stored in experts that the router does not pick in reverse-query contexts | Router-only edit (≤ 0.11 % of weights) |
| **H5 latent inverse** | Not stored as a reverse lookup, but recoverable from the forward memory (transpose or inverse of the forward relation map) | Closed-form edit built from the model's own forward operator. Needs no reverse examples at all. |

Evidence that steps 2–4 are relation-level, which is why the small-edit goal is plausible:
- **Hernandez et al. 2024**, *Linearity of Relation Decoding in Transformer LMs* (ICLR): for many relations the subject → object decoding is well approximated by **one affine map per relation**, shared across facts.
- **Geva et al. 2023:** extraction is done by attention heads at the last token, and the same heads work across subjects.
- **Wang & Sun 2025**, *Is the Reversal Curse a Binding Problem?*: argues the reversal curse comes from inconsistent and entangled concept representations. This links directly to our binding work.
- Our own REPL evidence points to H3 for at least some facts. "Question: What is the capital of India? Answer:" gave "India." (copy), and "The capital of India is" gave a generic phrase (habit). India ↔ New Delhi appears both ways all over web text, so storage (H1) is unlikely for that fact.

Citations are from memory. Check the details before writing results.

## 3. Phase 0: locate the failure (measurement only, no edits)

**Fact set.**
- About 600 fact pairs, each written forward and reverse, from **one-to-one relations only**, so the reverse has a single answer:
  - country ↔ capital
  - US state ↔ capital
  - element ↔ symbol
  - element ↔ atomic number
  - country ↔ currency (unique currencies only)
  - author ↔ only famous book
  - country ↔ demonym
  - team ↔ city (unique)
- Aliases are accepted (Delhi / New Delhi).
- Each relation is split 50/50 into **solve facts** and **held-out facts**. Two whole relations are held out entirely.
- Hand-curated from known facts, no downloads.

**Checkpoints.** Phase 2 final (`step_22888`), 2.5a step 20000, then the 2.5a final when it lands. This also gives the first look at whether knowledge storage changes across training.

| Test | What it measures | Hypothesis it decides |
|---|---|---|
| **0a Behaviour** | Greedy written answer (as in your REPL), and the rank of the correct answer among all candidates of that relation (e.g., all capitals). Every fact gets a class: both ways / forward only / reverse only / neither | The size of the problem, per relation |
| **0b In-context control** | The same reverse query with the forward sentence placed earlier in the prompt plus a distractor sentence | Whether the model can do reversal at all when the fact is visible |
| **0c Is it stored?** | Linear probe at the subject position ("India"), layer by layer: fit a map from the hidden state to the object ("New Delhi") on solve facts, score on held-out facts. Control: the same probe fitted on shuffled pairs (Hewitt & Liang 2019) | H1 vs H2/H3 |
| **0d Where does the answer die?** | Logit lens at the last token, layer by layer: the rank of the correct answer and of the habit/copy token | H3: the correct answer rises mid-network, then gets overwritten |
| **0e Patching** | Patch attention output and MoE output per layer, from a run where the answer is right (the in-context version) into the failing reverse run. Also patch the subject position vs the last position | Where extraction happens (H2), separately for attention and experts |
| **0f Routing** | Expert choices at the subject token in forward vs reverse prompts. Causal tracing of the forward fact to find which experts hold it | H4 |
| **0g Latent inverse (zero parameters)** | (i) Fit the forward relation map W_r (subject → object) from the model's own states; test whether its pseudo-inverse maps "India" to "New Delhi" on held-out facts. (ii) Transpose read-out through the experts that hold the forward fact (our embeddings are tied, so the output direction for "India" is its input embedding) | H5. If it works, the reverse knowledge is already in the weights and only the computation is one-way |

**Output of Phase 0:** a per-relation, per-fact map of which step fails, at which layer, and whether it is attention, experts or routing. That map is win condition 2 on its own. It's a measured answer to "where would generalization live in this model".

**Cost:** about 2–3 h on the A10, queued behind the GPU lock. No H100 time, training untouched.

## 4. Phase 1: the smallest edit that generalizes (only after Phase 0)

The Phase 0 map picks the candidates. Edits are tried from the smallest up, and the first one that generalizes wins:

| Edit | Size (share of 915.6M) | Used when Phase 0 shows |
|---|---|---|
| **E0 steering vector:** one direction added at one layer ("reverse-retrieval mode") | 1,024 params (0.0001 %) | H3, or a mode difference between forward and reverse |
| **E1 inverse operator, closed form:** install the pseudo-inverse of the model's own forward relation map at one layer (WSP, with no reverse training examples) | rank ≤ 16 on one matrix (< 0.01 %) | H5 |
| **E2 router edit** | ≤ 1.05M (0.11 %) | H4 |
| **E3 low-rank attention edit** (value/output) at the patched layers | rank 4–16 × 2–4 layers (0.01–0.1 %) | H2, writing side |
| **E4 low-rank query/key edit** on the extraction heads | similar | H2, where-to-look side |
| **E5 low-rank late-layer edit** | 0.1–0.5 % | H3 |

**How each edit is found:** closed form where one exists (E1), otherwise a small gradient fit restricted to that parameter subset, solved only on **solve facts**.

**Protection:** a KL term on about 2,000 general-text passages and on forward queries, so forward facts and language modelling are held fixed.

**Scoring, always on what the edit never saw:**
1. Reverse accuracy on **held-out facts** of the solve relations (G1).
2. Reverse accuracy on the **two held-out relations** (G2). This is the strongest evidence of an ability.
3. Forward accuracy (must not drop), LAMBADA, PIQA, the probe v4/v5/v6 binding scores.
4. **Memorization control:** the same edit fitted on shuffled answers. Its held-out gain must be ≈ 0, so any real gain is generalization, not edit capacity.

## 5. Phase 2: the clean ability test with new facts

Natural facts are already in the pretraining data, often both ways. The clean test is facts the model has seen **only one way**:

1. On a copy of the checkpoint, add about 200 fictitious facts written only forward ("Daphne Barrington directed *A Journey Through Time*", in the style of Berglund 2023), by a tiny fine-tune or a ROME insertion.
2. Confirm that forward works and reverse fails. That's the reversal curse reproduced in our model.
3. Apply the Phase 1 edit, which was fitted on *other* facts. **If the new one-way facts become reversible, the edit added an ability.** That's the strongest result possible here.
4. If they don't, then Phase 0 + 1 tell us exactly why (the information never reached the reverse lookup). The training-side fix for future phases is then precise: which tokens to predict (Kitouni et al. 2024, *The Factorization Curse*), reverse training (Golovneva et al. 2024), or recall-then-answer in SFT.

## 6. What each outcome teaches us

| Outcome | What we learn |
|---|---|
| E0–E1 works on held-out facts and relations | Reverse knowledge is already in the weights. The reversal curse in this model is a computation problem, fixable with a tiny closed-form edit. A strong, new result. |
| E2 works | MoE routing causes the reversal failure. Specific to MoE models and new. |
| E3–E5 work on held-out facts, but not on Phase 2's new facts | Extraction is the general ability and can be strengthened, but storage stays one-way. This tells training exactly what to change. |
| Nothing works, and 0c shows the information isn't stored | The reverse lookup is truly absent from the weights. Phase 0's layer-by-layer map is still the contribution, and the training fix becomes targeted. |

## 7. Order and cost

1. Finish the running passive sweep (`scripts/wsp_direction.sh`). It's the old-method baseline.
2. Build Phase 0 (fact set + `scripts/reversal_locate.py`): local, no cost.
3. Run Phase 0 on the A10: about 2–3 h (≈ $4 at $1.29/h).
4. Review the map with the user, then pick the Phase 1 edits.
5. Phase 1: about 4–6 h on the A10. Phase 2: about 3 h.

Nothing here touches the H100 or the 2.5a/2.5b run. Edited checkpoints are copies and are deleted after scoring.
