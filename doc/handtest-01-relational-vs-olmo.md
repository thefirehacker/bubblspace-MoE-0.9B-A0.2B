# Hand test 01: relational and calculation prompts, BubblSpace vs OLMo-2 1B across its training

Status: **26 Nov 2026, preliminary** (18 relational + 16 calculation prompts, one greedy run each). **28 Nov:** rerun on the post-trained models, SFT run 02b and RL v1 step 500 (§7). Prompt lists: [`benchmark/handtest/prompts/`](../benchmark/handtest/prompts/). Raw outputs in [`benchmark/handtest/`](../benchmark/handtest/):
- [`relational18_raw_greedy_20261126.txt`](../benchmark/handtest/relational18_raw_greedy_20261126.txt): relational prompts, all base checkpoints
- [`calculation16_raw_greedy_20261126.txt`](../benchmark/handtest/calculation16_raw_greedy_20261126.txt) and [`calculation16_olmo_main_base_20261126.txt`](../benchmark/handtest/calculation16_olmo_main_base_20261126.txt): calculation prompts
- [`instruct_all_raw_and_chat_20261126.txt`](../benchmark/handtest/instruct_all_raw_and_chat_20261126.txt) and [`instruct_chat_200tok_20261126.txt`](../benchmark/handtest/instruct_chat_200tok_20261126.txt): OLMo-2 1B Instruct, every set, raw and with its chat template

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

| OLMo-2-0425-1B-Instruct | `main` of [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct): base `main` + SFT + DPO + RL | ≈4.05T + post-training | 1.48B |

The OLMo base checkpoints come from [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) (safetensors). The Instruct model was run two ways: raw (the prompt as plain text, like the base models) and with its chat template (`scripts/chat_repl_hf.py --chat`, 200 new tokens, because it reasons step by step).

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

OLMo-2 1B **Instruct** on the same 18: **12/18 raw, 14/18 with its chat template**. With the template it still writes "Joon" for #5 ("Maria was helped by Joon… the one who needed help was"), declines #7 as "incomplete", says breakfast came first (#12), and doesn't reach an answer on #2 within 200 tokens.

**Calculation, 16 prompts** (time addition, counting, ages, prices, days of the week, bare arithmetic; same settings):

| # | Prompt (shortened) | Right | BS 2.5a 32.5B | OLMo 42B | OLMo 168B | OLMo 4T | OLMo final base | Instruct raw | Instruct chat |
|---|---|---|---|---|---|---|---|---|---|
| 1 | film 7 + 2 hours | 9 | ✗ 12:00 | ✗ 10:30 | ✗ 10:30 | ✓ | ✗ 9:30 | ✓ | ✓ |
| 2 | meeting 3 + 4 hours | 7 | ✗ 3:00 | ✗ 6 | ✗ 5:30 | ✓ | ✗ 7.30 | ✓ | ✓ |
| 3 | train 9 + 3 hours | 12 | ✗ 10 | ✗ 10 | ✗ 10:30 | ✗ 12:30 | ✓ | ✓ | ✓ |
| 4 | three apples, gave one, ate one | 1 | ✗ three | ✗ three | ✗ two | ✗ two | ✗ 2 | ✗ 2 | ✓ (writes 3−1=2, 2−1=1) |
| 5 | 5 pencils + 2 | 7 | ✓ | ✗ 6 | ✗ 6 | ✓ | ✓ | ✓ | ✓ |
| 6 | 8 birds − 3 | 5 | ✓ | ✗ 8 | ✗ 7 | ✓ | ✓ | ✓ | ✓ |
| 7 | 10 coins − 4 | 6 | ✗ 4 | ✗ 10 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | age 10, sister 3 older | 13 | ✗ | ✗ 4 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 9 | three pens at $2 | 6 | ✗ 3 | ✗ 3 | ✗ 3 | ✓ | ✓ | ✓ | ✓ |
| 10 | two weeks | 14 | ✓ | ✗ 28 | ✗ 28 | ✓ | ✓ | ✓ | ✓ |
| 11 | Monday + 2 days | Wednesday | ✗ | ✗ | ✗ | ✗ Tuesday | ✗ | ✓ | ✓ |
| 12 | today Friday → tomorrow | Saturday | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| 13 | yesterday Tuesday → today | Wednesday | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 14 | twelve minus five | 7 | ✗ 17 | ✗ 12.5 | ✗ twelve | ✗ ten | ✓ | ✓ | ✓ |
| 15 | four plus three | 7 | ✗ | ✗ | ✗ four | ✗ 10 | ✗ six | ✗ 8 | ✓ |
| 16 | half of ten | 5 | ✗ | ✗ | ✗ | ✗ | ✗ 10 | ✓ | ✓ |
| | **Score** | | **5/16** | 1/16 | 2/16 | 10/16 | 10/16 | 14/16 | **16/16** |

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
6. **Calculation comes mostly from pretraining, and post-training makes it reliable.** OLMo-2 1B base goes 1 → 2 → 10 of 16 from 42B to 168B to 4T tokens, with no SFT. Its final base model also scores 10. SFT + DPO + RL take it to 14 (raw) and 16 (chat template), mainly by writing out the steps: two-step problems (#4) and bare arithmetic (#15) are solved only when the model writes the intermediate result. BubblSpace at 32.5B tokens scores 5/16, above OLMo-2 1B at 42B and 168B. Whether SFT can lift a 5/16 base the same way can't be read from OLMo, which was never fine-tuned at 168B.
7. **Post-training doesn't fix binding at 1B.** OLMo-2 1B Instruct still answers "Joon" for "Maria was helped by Joon, so the one who needed help was". It gets none of the three twin pairs right raw, and declines them as "incomplete" with the chat template. It also writes "Chair" for the cap/room prompt and "RMJ LLC" for the JMB renaming.
8. **Passive with "beaten by" (#3) is our main gap.** OLMo gets it from 42B on; ours writes "not Kofi" (the doer-first habit from the word-order confound). The 2.5a decay now includes the v4 data aimed at this; rerun #3, #5, #6, #7 on the decay checkpoints.

## 5. Format habits seen on the way (2.5a), with their sources

| Habit | Trigger | Source pool | Share of that pool |
|---|---|---|---|
| Four-option quiz ("A. … D. … Answer: C. …") | a question ending in "?" | `fact_seeking` (Nemotron Fact-Seeking) | 100% of documents |
| Hedging assistant voice ("I'm unsure about the specific…") | a request ("give me…") | `synth_en` (PleIAs SYNTH) | ≈19.5% of documents |
| Relation-explanation essay ("a clear agent-patient relationship… a common confusion arises…") | some role prompts | `relational_llm` | style match |
| Contrastive template ("Which was too narrow, the door or the …? The door.") | fit prompts | contrastive v2/v3 | template |
| ConceptNet template ("Four plus three is a kind of number. Is number a kind of four plus three? No; …") | "X is" prompts | `kg_text` | template |

OLMo's anneal left the same kind of fingerprint: its final base model answers calculation prompts as classroom dialogue ("Student A: Got it…").

SYNTH also gives 2.5a a useful habit: it answers bare queries directly. OLMo-2 1B base continues a bare question as a list of more questions unless the prompt has a "Q: … A:" scaffold (stage-1 checkpoints).

## 6. Caveats

- 18 prompts, scored by hand, one greedy run each. Near-ties flip: in a manual REPL run of the same prompts, 2.5a got #5 right ("Maria."), but the scripted run didn't. Treat single-item differences as noise. The totals and the "never stops" pattern are the robust parts.
- The prompts are statement completions, which suit base models. A bare-question test would favour 2.5a, and a "Q: … A:" test would favour OLMo.
- Next step: turn these into an automated behaviour eval (≈40 prompts per category, scored on what the model writes), run on every checkpoint, ours and OLMo's.

## 7. After SFT and RL (28 Nov): SFT run 02b and RL v1 step 500

Same 34 prompts and settings (greedy, repetition penalty 1.0, no n-gram block), run two ways like OLMo-2 1B Instruct:
raw text with 40 new tokens, and in our chat template with 200. Per-item tables and outputs:
[`benchmark/phase4/handtest_longctx/`](../benchmark/phase4/handtest_longctx/summary.md).

| Model | Total tokens | Relational raw | Relational chat | Calculation raw | Calculation chat |
|---|---|---|---|---|---|
| BubblSpace 2.5a step 20,000 (base) | 32.5B | 6/18 | — | 5/16 | — |
| BubblSpace SFT run 02b | ≈40B | 10/18 | 5/18 | 6/16 | 8/16 |
| **BubblSpace RL v1 step 500** | ≈40B | **10/18** | **8/18** | **9/16** | **12/16** |
| OLMo-2 1B final base | ≈4.05T | 9/18 | — | 10/16 | — |
| OLMo-2 1B Instruct (SFT + DPO + RL) | ≈4.05T | 12/18 | 14/18 | 14/16 | 16/16 |

1. **Raw text: 10/18 relational after SFT, above OLMo-2 1B's final base (9/18) at about 1/100 of its tokens.** RL kept
   the same 10 items. Among them is the "beaten by" passive (#3), our main gap in §4.8, now right. RL raised raw
   calculation from 6 to 9/16 by writing the equation ("8 − 3 = 5", "2 * 7 = 14").
2. **Chat: RL v1 takes relational from 5 to 8/18 and calculation from 8 to 12/16.** The "helped by" and "taught by"
   passives (#5 Maria, #7 Hassan) are right for the first time, as is "Rohan is Asha's grandson" and the apples (#14).
   Run 02b answered 11 of the 34 chat prompts with long step-by-step text, several never reaching an answer or
   miscounting ("three pens cost 8", "4 + 3 … = 2"). RL v1 answers 33 of 34 in one short line.
3. **New errors in chat:** RL v1 says "Naledi" for "beaten by Kofi, so the winner was" (raw: Kofi), puts the film's end
   at "14:20", and calls the suitcase "too heavy" and the cup "too full".
4. **Wrong in every model and mode:** days of the week (Monday + 2 days → Tuesday; "yesterday was Tuesday" → Tuesday,
   which the 2.5a base got right), time order (#12), swapping (#15), "respectively" (#17), and "written by Ahmed" (#6).
   These state-tracking items were not in the RL task mix; §4.5's case for targeted data stands.
5. **Still behind OLMo-2 1B Instruct** in chat (8 vs 14/18, 12 vs 16/16), which had about 100× more pretraining tokens
   and a full post-training pipeline.

Caveat: our PyTorch greedy generator isn't bit-identical between calls. Run 02b gave two different answers to the same
apples prompt in the two lists.
