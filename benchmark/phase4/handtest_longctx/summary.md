# Hand tests and long context: SFT run 02b vs RL v1 step 500 (A10, 28 Nov)

Script: `scripts/handtest_longctx_a10.sh`. Weights: the HF exports of `phase3-run02b-step1500-tokens40B` (the RL start) and
`phase4-rlv1-step500-tokens40B` (the model on HF `main`).

## Hand tests

The 18 relational and 16 calculation prompts of [doc/handtest-01](../../../doc/handtest-01-relational-vs-olmo.md)
(prompt lists in `benchmark/handtest/prompts/`). Greedy, repetition penalty 1.0, no n-gram block, one run each,
scored by hand on the answer given. **Raw** = plain text continuation, 40 new tokens (as the 26 Nov base-model runs).
**Chat** = our chat template, 200 new tokens (as OLMo-2 1B Instruct with its template). Outputs: `handtest_raw.txt`,
`handtest_chat.txt`.

| Model | Total tokens | Relational raw | Relational chat | Calculation raw | Calculation chat |
|---|---|---|---|---|---|
| BubblSpace 2.5a step 20,000 (base, 26 Nov) | 32.5B | 6/18 | — | 5/16 | — |
| BubblSpace SFT run 02b | ≈40B | 10/18 | 5/18 | 6/16 | 8/16 |
| **BubblSpace RL v1 step 500** | ≈40B | **10/18** | **8/18** | **9/16** | **12/16** |
| OLMo-2 1B final base (26 Nov) | ≈4.05T | 9/18 | — | 10/16 | — |
| OLMo-2 1B Instruct, SFT + DPO + RL (26 Nov) | ≈4.05T | 12/18 | 14/18 | 14/16 | 16/16 |

**Relational, per item** (✗ with the answer given where it's informative):

| # | Prompt (shortened) | Right | SFT 02b raw | RL v1 raw | SFT 02b chat | RL v1 chat |
|---|---|---|---|---|---|---|
| 1 | ice melted… the effect was the | ice melting | ✓ | ✓ | ✓ | ✓ |
| 2 | Ravi > Meera > Sam… the shortest is | Sam | ✓ | ✓ | ✗ no answer in 200 tokens | ✓ |
| 3 | Naledi was beaten by Kofi… winner | Kofi | ✓ | ✓ | ✓ | ✗ Naledi |
| 4 | trophy / suitcase too… | small | ✓ | ✓ | ✗ essay on "too" | ✗ heavy |
| 5 | Maria was helped by Joon… needed help | Maria | ✗ Joon | ✗ Joon | ✗ Joon | ✓ |
| 6 | letter written by Ahmed… writer | Ahmed | ✗ | ✗ | ✗ Sofia | ✗ Sofia |
| 7 | Priya was taught by Hassan… teacher | Hassan | ✗ experienced | ✗ experienced | ✗ a teacher | ✓ |
| 8 | piano / door too… | narrow | ✓ | ✓ | ✓ | ✓ |
| 9 | water / cup too… | small | ✓ | ✓ | ✗ essay | ✗ full |
| 10 | Mr. Lee is Daniel's teacher… Daniel is Mr. Lee's | student | ✓ | ✓ | ✗ "a typo" | ✗ repeats the prompt |
| 11 | Asha is Rohan's grandmother… Rohan is Asha's | grandson | ✓ | ✓ | ✗ repeats the prompt | ✓ |
| 12 | breakfast after a run… first thing | the run | ✗ | ✗ | ✗ | ✗ |
| 13 | started at 7, two hours later… ended at | 9 | ✗ 10:30 | ✗ 10:30 | ✓ | ✗ 14:20 |
| 14 | three apples, gave one, ate one… | one | ✗ 3 | ✗ 3 | ✗ no answer | ✓ |
| 15 | balls swapped… box A has the | blue ball | ✗ red | ✗ red | ✗ red | ✗ red |
| 16 | Kiran does not like tea… the one is | Kiran | ✓ | ✓ | ✓ | ✓ |
| 17 | doctor and pilot, respectively… pilot is | Elena | ✗ | ✗ | ✗ loops | ✗ |
| 18 | hot : cold :: tall : | short | ✓ | ✓ | ✗ essay | ✗ "the sun is to the earth" |
| | **Score** | | **10/18** | **10/18** | 5/18 | **8/18** |

**Calculation, per item:**

| # | Prompt (shortened) | Right | SFT 02b raw | RL v1 raw | SFT 02b chat | RL v1 chat |
|---|---|---|---|---|---|---|
| 1 | film 7 + 2 hours | 9 | ✗ 10:30 | ✗ 10:30 | ✓ | ✗ 14:20 |
| 2 | meeting 3 + 4 hours | 7 | ✗ 6 | ✗ 6 | ✓ | ✓ |
| 3 | train 9 + 3 hours | 12 | ✗ 11 | ✗ 11 | ✗ 10:00 | ✓ |
| 4 | three apples, gave one, ate one | 1 | ✓ | ✗ 3 | ✗ no answer | ✓ |
| 5 | 5 pencils + 2 | 7 | ✓ | ✓ | ✓ | ✓ |
| 6 | 8 birds − 3 | 5 | ✗ 11 | ✓ | ✓ | ✓ |
| 7 | 10 coins − 4 | 6 | ✓ | ✓ | ✓ | ✓ |
| 8 | age 10, sister 3 older | 13 | ✓ | ✓ | ✓ | ✓ |
| 9 | three pens at $2 | 6 | ✓ | ✓ | ✗ 8 (adds 2 + 6) | ✓ |
| 10 | two weeks | 14 | ✗ 28 | ✓ | ✓ | ✓ |
| 11 | Monday + 2 days | Wednesday | ✗ | ✗ | ✗ Tuesday | ✗ Tuesday |
| 12 | today Friday → tomorrow | Saturday | ✓ | ✓ | ✗ Monday | ✗ Monday |
| 13 | yesterday Tuesday → today | Wednesday | ✗ Tuesday | ✗ Tuesday | ✗ Tuesday | ✗ Tuesday |
| 14 | twelve minus five | 7 | ✗ twenty-five | ✓ | ✓ | ✓ |
| 15 | four plus three | 7 | ✗ | ✓ | ✗ 2 | ✓ |
| 16 | half of ten | 5 | ✗ | ✗ | ✗ | ✓ |
| | **Score** | | 6/16 | **9/16** | 8/16 | **12/16** |

Our PyTorch greedy generator isn't bit-identical between calls: run 02b gave "3 apples" (relational list) and "1 apple"
(calculation list) to the same prompt; RL v1 answered the same both times. Each list is scored on its own answer.

## Long context

`scripts/eval_long_context.py`, the Phase 2.5b test: a 5-digit passkey hidden at 5 depths (40 trials per length), and
2-hop questions (person → town → bakery, 3 distractor sets; chance 25% among the 4 listed bakeries; 24 trials per
length), on held-out text. **Raw** = the 2.5b format (plain text, comparable with the base model). **Chat** (new
`--chat` flag) = the document and question in a user turn, the assistant turn starting with the answer stem
("The pass key is" / "The bakery is called"), same scoring.

| Model | Format | Passkey 2k | 4k | 8k | 16k | 2-hop 2k | 4k | 8k | 16k | Loss at 1–2k / 8–16k tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.5b final (base) | raw | 88% | 70% | 73% | 48% | 25% | 29% | 29% | 17% | 2.49 / 2.24 |
| SFT run 02b | raw | 85% | 73% | 80% | 40% | 13% | 29% | 25% | 0% | 2.52 / 2.27 |
| **RL v1 step 500** | raw | 85% | 73% | 83% | 50% | 29% | 13% | 8% | 8% | 2.53 / 2.28 |
| SFT run 02b | chat | 75% | 48% | 38% | 30% | 13% | 25% | 17% | 0% | — |
| **RL v1 step 500** | chat | 75% | 55% | 38% | 33% | 33% | 25% | 21% | 0% | — |

1-hop control (raw): 2.5b 58 / 29 / 21 / 13%, run 02b 42 / 33 / 38 / 25%, RL v1 46 / 21 / 38 / 46%. Chat: run 02b
13 / 29 / 25 / 17%, RL v1 42 / 42 / 33 / 25%. With 40 and 24 trials, ±8–10 points is noise.
