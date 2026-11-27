# How much knowledge can BubblSpace hold, and how far should we train?

Status: **discussion note, 27 Nov 2026.** Written after asking two linked questions: (1) is there a rough ceiling on how many facts a 0.9B MoE like ours can store, knowing OLMo-2 trains models this size out to 4T tokens; (2) if we want more knowledge and skill, how many more tokens do our own benchmarks justify. The agreed training order that follows from this is in [plan-06](plan-06-phase-4-rl.md). Numbers come from the [README](../README.md), [handtest-01](handtest-01-relational-vs-olmo.md), and [binding-study-02](binding-study-02-priors-and-word-order.md).

## 1. The question

BubblSpace MoE-0.9B-A0.2B has about 915 million parameters and about 211 million active per token. After Phase 2.5 and SFT we sit near 40 billion tokens. OLMo-2 1B is a dense model of similar size, trained to about 4 trillion tokens. That gap invites a natural worry: maybe we have barely begun to fill the model, and the right move is to keep pouring data until we are closer to OLMo's scale.

Two different things get mixed up in that worry. One is **capacity**: how much information the weights can hold if every bit of them is used. The other is **how full they already are**, and which skills still move when you add more tokens. Capacity is a property of the architecture. Fullness and returns are properties of the training curve. This note separates them, then uses our evals and OLMo's checkpoints to pick a token target that is not "as far as OLMo went."

## 2. A ceiling from the weights

A common working estimate in the literature is that a trained network stores on the order of **two bits of information per parameter**. That is not a theorem about our MoE; it is a scale to think with. Applied to ≈915 million parameters it gives roughly two billion bits, about 200 megabytes of effective storage if every weight is doing useful work.

Almost all of that room lives in the experts. The embedding and the attention together are only about 110 million parameters; the 64 experts are about 805 million. A short closed-book fact — one subject, one relation, one object — is often treated as about a hundred bits once the model stores it with the redundancy language models need. Two billion bits divided by that order of magnitude is on the order of **ten to twenty million** such facts.

That ceiling is large enough to matter. It is a useful slice of Wikipedia-style knowledge, the kind of thing a small assistant can answer from memory. It is still small compared with the web, and it is smaller than what a 7 billion parameter model can hold, because that model has about eight times as many weights.

Capacity to **store** and capacity to **retrieve** are not the same number. On every token our router picks only 8 of the 64 experts, so only about 101 million of those 805 million expert parameters are read. A fact that sits in an expert the router rarely chooses for that question is in the checkpoint and still does not come out. So the practical guess for facts you can actually get back is a **few million**, not the full ten-to-twenty million.

None of this says the model is full. How full it is depends on how many tokens have written into those weights. For dense models near a billion parameters, the classic loss-optimal balance of parameters and tokens is around twenty billion tokens. We are past that, at about forty billion, so the weights are no longer empty. They are also nowhere near the storage ceiling. That matches what we see in closed-book evals after the knowledge phase: TriviaQA at 15.8% and Natural Questions at 4.5%. Those are the scores of a model that has started to hold facts, not of a model that has packed the experts.

OLMo-2 1B is denser (about 1.48 billion parameters, every weight read on every token) and was trained to about four trillion tokens. Knowledge and calculation scores at this size keep rising across that whole range. The same order of weights would still be absorbing facts at a few hundred billion tokens, and still at a trillion. They would not suddenly know what a seven-billion-parameter model knows. Scale of data fills a fixed box; it does not enlarge the box.

So the short reading of capacity is: the checkpoint can hold on the order of ten million short facts; the router can surface a few million of them; forty billion tokens has written in only a small fraction of either number. Room remains. The next question is whether that room is worth filling with generic tokens, and how far.

## 3. What our own curve is still doing

If more tokens always helped every skill equally, the answer would be "train as long as OLMo." Our phase table says otherwise.

From Phase 1 to the end of Phase 2 to the end of 2.5a, closed-book TriviaQA went 8.5% → 11.6% → 15.8%. That is a real climb, and the last step came from a knowledge-dense mix (full Wikipedia, fact-seeking, SYNTH). Natural Questions moved only 2.9% → 3.9% → 4.5%. The English six-task average has been essentially flat since Phase 1 (about 47). HumanEval and MBPP rose through Phase 2 and then only a little in 2.5a. The eighteen relational hand prompts scored 6/18 at both 22B and 32.5B tokens. The sixteen calculation prompts sat at 5/16 at 32.5B.

So at our current scale, **facts are still being written in when the data is factual**, and several other skills have already stopped moving under the mixes we used. More tokens of the same kind will not automatically lift binding, stopping behaviour, or state-tracking. That is the first reason not to copy OLMo's 4T schedule blindly: our curve is already selective about what more data buys.

## 4. What OLMo-2 1B bought with 4T tokens

The fair comparison is the same hand prompts, scored the same way, on OLMo-2 1B checkpoints from early training through the final base model ([handtest-01](handtest-01-relational-vs-olmo.md)).

At matched tokens we lead. Both of our checkpoints score 6/18 on the relational set; OLMo scores 3/18 at 21B and 4/18 at 42B. OLMo then passes us: 8/18 at 168B and 10/18 at 4T (9/18 after its anneal). The important shape is that **most of that relational gain arrives by about 168B**. From 168B to 4T is roughly twenty-four times more data for two more items on that set.

Calculation is different. OLMo goes 1/16 → 2/16 → 10/16 from 42B to 168B to 4T, with no SFT. The big jump is at trillion scale. Post-training then takes that base to 14/16 raw and 16/16 with the chat template, mostly by writing out intermediate steps. Our 5/16 at 32.5B is already ahead of OLMo at 42B and 168B, so we are not behind on that curve yet; we also have not seen whether SFT alone can lift a 5/16 base the way it lifts a 10/16 base.

WinoGrande twin pairs tell a similar long story: OLMo-2 1B moves from about 19% at 42B to 26% near 420B to the mid-30s at 4T. The climb continues, but it is slow.

And five items fail in every model we checked, including OLMo-2 1B at 4T: time order, counting the apples, exclusion, "respectively", and the agent of a "taught by" passive. Swapping the balls is right only once, early, which looks like luck. Those failures need keeping and updating a state, or mapping roles through syntax. Four trillion tokens of general web text did not teach that to a 1B model in a single pass. A related failure — passive roles read with a word-order rule — went from 0 to 40/40 written answers with about 2.5 million tokens of order-alternated contrastive data in a controlled fine-tune ([binding-study-02](binding-study-02-priors-and-word-order.md) §6.1). Scale and targeted data are different levers.

## 5. The token number that follows

Put the capacity estimate and the two curves together.

If the goal is **more knowledge and the skills that still move at this size**, the curves support going to about **200 billion tokens total**. That is roughly 160 billion more than today, about four times what we have already trained, and the neighbourhood where OLMo-2 1B's relational hand-test gains largely arrived. TriviaQA still rising in our 2.5a run is consistent with that: factual data at this size still has somewhere to write.

Going further, toward **about 420B**, still buys a slow climb on twin-pair ranking in OLMo's numbers. It is a weaker default. Going all the way to **4T** is what OLMo did for calculation on a denser 1.5B model. It is not evidence that BubblSpace needs four trillion tokens to fill its knowledge capacity, and it still leaves the state-tracking failures untouched.

So: about 200B is a reasoned later target for a knowledge stage. Four trillion is not the plan by default. Targeted corpora remain the right tool for count, swap, exclusion, "respectively", and similar items, whether or not we ever approach OLMo's token count.

## 6. Why that target is not "train to 200B next"

Room in the weights and a sensible token horizon do not decide the next week of work. Reinforcement learning improves how the model uses what it already knows; a reward of zero says "wrong," not what the right answer was. New facts have to come from data. Behaviour and format problems often come from SFT. Skill problems where the model sometimes gets it right need more RL, not more Wikipedia.

That is why the agreed order in [plan-06](plan-06-phase-4-rl.md) is:

1. Finish SFT, compare run 01 and run 02b, upload the winner.
2. Run **RL v1** on that model (about a day or two of prep, then about six hours of training). RL v1 is also the first full test of the RL pipeline, so a small round before a large one is deliberate.
3. Benchmark that model fully, and for each test question sample **eight answers**, not one.
4. Only then choose the next round: more or bigger RL, an SFT mix fix, a docs or search tool, or a knowledge stage from the 2.5b base followed by a redo of SFT and a bigger RL.

The eight-sample check is what turns "the model is weak at X" into a decision:

- If the first try is often right and the eight tries stay right, that area is solved.
- If the first try is rarely right but one of the eight often is, the model can do the task and does not pick the good answer reliably. That is a skill and choice problem. More or bigger RL is aimed exactly at that.
- If almost none of the eight is ever right, the model does not know it. That is a knowledge problem (TriviaQA-style facts, APIs it never saw). Then a knowledge pretraining stage toward something like the 200B horizon, or a docs tool, is the lever — not another RL run on empty signal.
- If the content is right and the format or behaviour is wrong (refusals, quiz templates, calling a tool when none is needed), the SFT mix is the lever.

A large knowledge stage starts only when that map says knowledge. State-tracking failures stay candidates for targeted data even if we later train much further. The capacity estimate says there is room. The benchmarks say about 200B is a sensible place to aim if we fill that room. The pass@1 / pass@8 map says when to start filling it.
