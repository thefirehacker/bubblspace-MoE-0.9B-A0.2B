# Hand test 01: relational prompts, BubblSpace vs OLMo-2 1B across its training

Status: **26 Nov 2026, preliminary** (18 prompts, one greedy run each). Raw outputs: [`benchmark/handtest/relational18_raw_greedy_20261126.txt`](../benchmark/handtest/relational18_raw_greedy_20261126.txt).

## 1. Question

Scored benchmarks rank pre-written answers. This test looks at what each model **writes**, on short relational prompts with one clear answer. The question: how does BubblSpace compare with OLMo-2 1B at matched tokens, and how many more tokens does OLMo-2 1B need to pass it?

## 2. Setup

| | |
|---|---|
| Decoding | raw greedy: `/greedy`, repetition penalty 1.0, no n-gram blocking, 40 new tokens. Each prompt is independent (no history) |
| Scoring | the first answer written, by hand: ✓ if it names the right entity or value |
| Prompts | 18 statement-form prompts (no "?", which triggers 2.5a's quiz habit, §5): cause/effect, two-step chain, passive roles, fit, role reversal, family relations, time order, time arithmetic, counting, swapping, exclusion, "respectively", analogy |
| Scripts | [`scripts/chat_repl.py`](../scripts/chat_repl.py) (ours) and [`scripts/chat_repl_hf.py`](../scripts/chat_repl_hf.py) (any HF model; same commands and decoding), run non-interactively on the A10 |

| Model | Checkpoint | Total tokens | Params (active) |
|---|---|---|---|
| BubblSpace Phase 2 final | `phase2_step_22888` | 22B | 916M (211M) |
| BubblSpace 2.5a | `phase25a_step_20000` (before decay) | 32.5B | 916M (211M) |
| OLMo-2-0425-1B | `stage1-step10000-tokens21B` | 21B | 1.48B (all) |
| OLMo-2-0425-1B | `stage1-step20000-tokens42B` | 42B | 1.48B |
| OLMo-2-0425-1B | `stage1-step80000-tokens168B` | 168B | 1.48B |
| OLMo-2-0425-1B | `stage1-step1907359-tokens4001B` (end of pretraining) | 4.0T | 1.48B |
| OLMo-2-0425-1B | `main` (final base, after its 50B-token stage-2 anneal; no SFT) | ≈4.05T | 1.48B |

All OLMo checkpoints are base models from [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) (safetensors).

## 3. Results

| # | Prompt (shortened) | Right | BS P2 22B | BS 2.5a 32.5B | OLMo 21B | OLMo 42B | OLMo 168B | OLMo 4T | OLMo final |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ice melted… the effect was the | ice melting | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | Ravi > Meera > Sam… the shortest is | Sam | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 3 | Naledi was beaten by Kofi… winner | Kofi | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| 4 | trophy / suitcase too… | small | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 5 | Maria was helped by Joon… needed help | Maria | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| 6 | letter written by Ahmed… writer | Ahmed | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 7 | Priya was taught by Hassan… teacher | Hassan | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | piano / door too… | narrow / small | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 9 | water / cup too… | small | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | Mr. Lee is Daniel's teacher… Daniel is Mr. Lee's | student | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| 11 | Asha is Rohan's grandmother… Rohan is Asha's | grandson | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| 12 | breakfast after a run… first thing | the run | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 13 | started at 7, two hours later… ended at | 9 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| 14 | three apples, gave one, ate one… | one | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 15 | balls swapped… box A has the | blue ball | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 16 | Kiran does not like tea… the one is | Kiran | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 17 | doctor and pilot, respectively… pilot is | Elena | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 18 | hot : cold :: tall : | short | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| | **Score** | | **6/18** | **6/18** | 3/18 | 4/18 | 8/18 | **10/18** | 9/18 |
| | **Answer, then stop** (ended within 40 tokens) | | **8/18** | **6/18** | 0/18 | 0/18 | 0/18 | 0/18 | 0/18 |

**Bare facts, four prompt forms** (same settings):

| Prompt | BS P2 | BS 2.5a | OLMo 168B | OLMo 4T | OLMo final |
|---|---|---|---|---|---|
| What is the capital city of India? | ✓ | ✗ quiz, "C. Kolkata" | ✗ lists more questions | ✓ "Answer: New Delhi" | ✗ (empty) |
| Q: What is the capital of India? A: | ✓ | ✓ "Delhi." | ✓ | ✓ | ✓ but quiz format ("New Delhi B: Mumbai C:…") |
| The capital of India is | ✓ | ✗ describes India | ✓ | ✓ | ✓ |
| what is the capital of India (no "?") | ✓ | ✓ | ✓ | ✗ lists more questions | ✓ |
| **Score** | **4/4** | 2/4 | 3/4 | 3/4 | 3/4 |

## 4. Findings

1. **At matched tokens, BubblSpace leads.** Both our checkpoints score 6/18 at 22B and 32.5B tokens. OLMo-2 1B scores 3/18 at 21B and 4/18 at 42B.
2. **OLMo-2 1B passes us at about 5× the tokens.** It scores 8/18 at 168B and 10/18 at 4T (9/18 after its anneal). From 168B to 4T, 24× more tokens bought 2 more items.
3. **One item only BubblSpace solves.** The two-step chain (#2, "shortest of Ravi > Meera > Sam") is right in both our checkpoints and wrong in all five OLMo checkpoints, up to the final model.
4. **BubblSpace answers and stops; OLMo-2 1B never stops.** Ours ended after a short answer on 6–8 of 18 prompts. OLMo-2 1B ended on none, at any checkpoint, and mostly looped.
5. **Five items fail in every model at every scale, including OLMo-2 1B at 4T:** time order (#12), counting (#14), exclusion (#16), "respectively" (#17), and the agent of a "taught by" passive (#7). Swapping (#15) is right only at OLMo 21B, which looks like luck. These need keeping and updating a state, or mapping roles through syntax. 4T tokens of general web text didn't teach that to a 1B model in a single pass. They are candidates for targeted data, as the v4 order-alternation data was for passive ([binding-study-02 §6.1](binding-study-02-priors-and-word-order.md): passive 0 → 40/40 written in a test fine-tune).
6. **Passive with "beaten by" (#3) is our main gap.** OLMo gets it from 42B on; ours writes "not Kofi" (the doer-first habit from the word-order confound). The 2.5a decay now includes the v4 data aimed at this; rerun #3, #5, #6, #7 on the decay checkpoints.

## 5. Format habits seen on the way (2.5a), with their sources

| Habit | Trigger | Source pool | Share of that pool |
|---|---|---|---|
| Four-option quiz ("A. … D. … Answer: C. …") | a question ending in "?" | `fact_seeking` (Nemotron Fact-Seeking) | 100% of documents |
| Hedging assistant voice ("I'm unsure about the specific…") | a request ("give me…") | `synth_en` (PleIAs SYNTH) | ≈19.5% of documents |
| Relation-explanation essay ("a clear agent-patient relationship… a common confusion arises…") | some role prompts | `relational_llm` | style match |
| Contrastive template ("Which was too narrow, the door or the …? The door.") | fit prompts | contrastive v2/v3 | template |

SYNTH also gives 2.5a a useful habit: it answers bare queries directly. OLMo-2 1B base continues a bare question as a list of more questions unless the prompt has a "Q: … A:" scaffold (stage-1 checkpoints).

## 6. Caveats

- 18 prompts, scored by hand, one greedy run each. Near-ties flip: in a manual REPL run of the same prompts, 2.5a got #5 right ("Maria."), but the scripted run didn't. Treat single-item differences as noise. The totals and the "never stops" pattern are the robust parts.
- The prompts are statement completions, which suit base models. A bare-question test would favour 2.5a, and a "Q: … A:" test would favour OLMo.
- Next step: turn these into an automated behaviour eval (≈40 prompts per category, scored on what the model writes), run on every checkpoint, ours and OLMo's.
