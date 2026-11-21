# Entity-attribute binding probe results — step_19073 (Phase 1 final checkpoint)

WinoGrande-style minimal-pair probes, freshly written (not from the WinoGrande
dataset), scored via log-likelihood (`BubblSpaceLM._score_continuation`) --
the same mechanism `lm-eval-harness` uses for the real WinoGrande benchmark.
Each case: two prefixes differing only in which entity is named, two
candidate completions; correct binding means the preferred completion flips
the right way when the named entity flips. Case definitions live one level
up in `../probefinding-winogrande-<timestamp>.json`.

## Files

- `probe_results_original_11case_step19073.json` -- first probe: the car/alley
  example plus 10 fresh spatial-fit cases (small/big, narrow/wide, tall/low).
  **6/11 correct.** Narrow/wide axis was 0/4 -- the model always preferred
  "too narrow" regardless of which entity was named.
- `probe_results_v1_confounded_step19073.json` -- expanded to 100 cases across
  10 categories (spatial + 7 non-spatial relational domains). **35/100
  correct.** Contains a known grammar-agreement confound (see below).
- `probe_results_v2_clean_step19073.json` -- same 100 cases with the confound
  fixed in `agent_patient_causation` and `possession_obligation`. **This is
  the trustworthy result: 25/100 correct.**

## The confound (why v1 != v2)

In v1, `agent_patient_causation` (all 10 cases) and 8/10
`possession_obligation` cases had ctxA ending in "had" paired with a
past-participle completion, and ctxB ending in "was" paired with an
adjective completion. The model could pick the grammatically-fluent
completion from verb-tense agreement alone, without resolving which entity
the trait actually describes -- inflating those two categories. Fixed in v2
by making both prefixes end in "had" and both completions grammatically
compatible with "had ___" in every rewritten case.

Effect of the fix:

| Category | v1 (confounded) | v2 (clean) |
|---|---|---|
| agent_patient_causation | 100% | 40% |
| possession_obligation | 60% | 0% |

## v2 (clean) results by category, ranked

| Category | Score |
|---|---|
| spatial_height | 7/10 (70%) |
| agent_patient_causation | 4/10 (40%) |
| social_attribution | 4/10 (40%) |
| spatial_size | 4/10 (40%) |
| spatial_width | 3/10 (30%) |
| material_property | 2/10 (20%) |
| ability_attribution | 1/10 (10%) |
| comparative_value | 0/10 (0%) |
| emotional_attribution | 0/10 (0%) |
| possession_obligation | 0/10 (0%) |
| **Total** | **25/100 (25%)** |

**Read:** this is not a uniform "weak everywhere" signal. Spatial/physical
clearance reasoning (`spatial_height`) holds up as a real, comparatively
strong capability. Three categories sit at a flat, confound-free zero
(`comparative_value`, `emotional_attribution`, `possession_obligation`) --
no directional movement toward the correct entity at all, not just an
insufficient-magnitude signal. That is a sharper, more actionable target for
Phase 2's contrastive/counterfactual data pool than "WinoGrande is at
chance": prioritize comparative, emotional, and possession/obligation
attribution, not more spatial-fit examples (which the model already
partially handles).
