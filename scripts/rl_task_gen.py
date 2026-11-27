"""Generated Phase 4 RL tasks: the generator picks the numbers/names first, computes the answer, then writes the text,
so every answer key is correct by construction (doc/rl-primer-phase4.md §5.1). Rewards: src/rl/rewards.py.

  time_problems    clock-time word problems in many phrasings: late / early arrival, start + duration, end - duration,
                   "ago", "from now", minutes and 12-hour wraparound; plus "how many minutes until" (number answers).
                   The words movie / film / cinema never appear (they are in the user's hand tests).
  binding          who-did-what: give/receive, passive agent/patient, comparison chains, help/need, with the receiver or
                   the agent named first equally often; names exclude the hand-test and probe names
  tool_tasks       live-data questions with a matching tool among distractors -> the exact gold call; plus no-call
                   tasks (greetings / fixed facts with tools present)
  live_data        live-data questions with no tools -> "I don't have live data" (refusal-balance kind no_live)

Each task: {"area", "kind", "messages": [[role, text], ...], + the kind's answer key}.
"""
import json
import random

from phase3_gen_own_sets import (CAPITALS, FEMALE, HANDTEST_NAMES, MALE, RT_CITIES, RT_TOOLS, _rt_case, _rt_tools,
                                 load_probe_names)

EVENTS = ["concert", "class", "train", "bus", "meeting", "football match", "lecture", "flight", "play", "workshop",
          "dinner party", "wedding", "exam", "cricket match", "parade", "seminar", "piano recital", "ferry"]


def _fmt(h: int, m: int, rng) -> str:
    """12-hour clock text: 2:30 / 2 o'clock (on the hour, sometimes)."""
    if m == 0 and rng.random() < 0.35:
        return f"{h} o'clock"
    return f"{h}:{m:02d}"


def _to_clock(total_min: int) -> tuple[int, int]:
    total_min %= 12 * 60
    h, m = divmod(total_min, 60)
    return (h or 12), m


def _dur(mins: int, rng) -> str:
    h, m = divmod(mins, 60)
    if h and m:
        return rng.choice([f"{h} hour{'s' if h > 1 else ''} and {m} minutes", f"{h * 60 + m} minutes"])
    if h:
        return rng.choice([f"{h} hour{'s' if h > 1 else ''}", f"{h * 60} minutes"]) if h <= 2 else f"{h} hours"
    return f"{m} minutes"


def time_problem(rng) -> dict:
    h0, m0 = rng.randint(1, 12), rng.choice(range(0, 60, 5))
    t0 = (h0 % 12) * 60 + m0
    d = rng.choice([5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 105, 120, 150, 180, 210, 240, 300])
    ev, who = rng.choice(EVENTS), rng.choice(["I", "Sam's sister", "my friend", "the team", "she", "he"])
    T, D = _fmt(h0, m0, rng), _dur(d, rng)
    k = rng.randrange(7)
    if k == 0:    # late arrival: start = arrival - late
        q = rng.choice([f"I reached the {ev} at {T} and I was {D} late. What time did the {ev} start?",
                        f"I got to the {ev} {D} after it began. I arrived at {T}. When did it begin?",
                        f"Arriving at {T}, I was {D} late for the {ev}. What was the start time?"])
        ans = _to_clock(t0 - d)
    elif k == 1:  # early arrival: start = arrival + early
        q = rng.choice([f"I arrived at the {ev} at {T}, {D} early. When does it start?",
                        f"We got to the {ev} {D} before it started. We arrived at {T}. What time did it start?"])
        ans = _to_clock(t0 + d)
    elif k == 2:  # start + duration = end
        q = rng.choice([f"The {ev} started at {T} and lasted {D}. What time did it end?",
                        f"If the {ev} begins at {T} and runs for {D}, when does it finish?",
                        f"A {ev} that starts at {T} takes {D}. At what time is it over?"])
        ans = _to_clock(t0 + d)
    elif k == 3:  # end - duration = start
        q = rng.choice([f"The {ev} ended at {T}. It lasted {D}. When did it start?",
                        f"The {ev} finished at {T} after running for {D}. What time did it begin?"])
        ans = _to_clock(t0 - d)
    elif k == 4:  # ago
        q = rng.choice([f"It is {T} now. {who.capitalize()} left for the {ev} {D} ago. What time was that?",
                        f"The clock shows {T}. What time was it {D} ago?"])
        ans = _to_clock(t0 - d)
    elif k == 5:  # from now
        q = rng.choice([f"It is {T} now. What time will it be in {D}?",
                        f"The {ev} starts {D} from now, and it is {T}. When does it start?"])
        ans = _to_clock(t0 + d)
    else:         # minutes until (number answer)
        h1, m1 = _to_clock(t0 + d)
        q = rng.choice([f"It is {T} now and the {ev} starts at {_fmt(h1, m1, rng)}. How many minutes until it starts?",
                        f"The {ev} begins at {_fmt(h1, m1, rng)}. If it is {T} now, how many minutes do I have to wait?"])
        return {"area": "time", "kind": "number", "messages": [["user", q]], "answer": str(d)}
    if rng.random() < 0.15:  # an irrelevant number now and then, so "add every number" doesn't pay
        q = q.replace(f"the {ev}", f"the {ev} (tickets were {rng.randint(3, 40)} dollars)", 1)
    return {"area": "time", "kind": "time", "messages": [["user", q]], "answer": f"{ans[0]}:{ans[1]:02d}"}


def _names(repo_root="."):
    banned = HANDTEST_NAMES | load_probe_names(repo_root)
    return [n for n in MALE + FEMALE if n not in banned]


OBJECTS = ["red cap", "blue notebook", "silver key", "green umbrella", "old map", "small parcel", "wooden box", "yellow kite"]
PASSIVE = [("praised", "praised", "was praised"), ("thanked", "thanked", "was thanked"), ("helped", "helped", "was helped"),
           ("called", "called", "was called"), ("taught", "taught", "was taught"), ("interviewed", "interviewed", "was interviewed")]


def binding(rng, names) -> dict:
    a, b, c = rng.sample(names, 3)
    k = rng.randrange(5)
    if k == 0:    # give: who has the object now?
        o1, o2 = rng.sample(OBJECTS, 2)
        s = rng.choice([f"{a} gave {b} the {o1}, and {b} gave {a} the {o2}.", f"{b} received the {o1} from {a}, and {a} got the {o2} from {b}."])
        q, ans = rng.choice([(f"Who has the {o1} now?", b), (f"Who has the {o2} now?", a)])
    elif k == 1:  # passive: agent or patient
        verb, _, was = rng.choice(PASSIVE)
        s = rng.choice([f"{a} {was} by {b}.", f"{b} {verb} {a}."])
        q, ans = rng.choice([(f"Who {verb} someone?", b), (f"Who {was}?", a)])
    elif k == 2:  # comparison chain
        adj, adj_r = rng.choice([("taller", "shortest"), ("older", "youngest"), ("faster", "slowest"), ("heavier", "lightest")])
        top = {"shortest": "tallest", "youngest": "oldest", "slowest": "fastest", "lightest": "heaviest"}[adj_r]
        s = rng.choice([f"{a} is {adj} than {b}. {b} is {adj} than {c}.", f"{b} is {adj} than {c}, and {a} is {adj} than {b}."])
        q, ans = rng.choice([(f"Who is the {adj_r}?", c), (f"Who is the {top}?", a)])
    elif k == 3:  # help / need
        s = rng.choice([f"{b} needed help with the move, so {a} helped {b}.", f"{a} helped {b}, who needed help with the move."])
        q, ans = rng.choice([("Who needed help?", b), ("Who gave the help?", a)])
    else:         # sent / received
        thing = rng.choice(["a letter", "an email", "a gift", "a postcard"])
        s = rng.choice([f"{a} sent {b} {thing}.", f"{b} got {thing} from {a}.", f"{thing.capitalize()} was sent to {b} by {a}."])
        q, ans = rng.choice([(f"Who received {thing}?", b), (f"Who sent {thing}?", a)])
    return {"area": "binding", "kind": "name", "messages": [["user", f"{s} {q}"]], "answer": ans, "names": [a, b, c]}


def tool_task(rng) -> dict:
    if rng.random() < 0.7:  # a matching tool exists: exact call expected
        q, tool, args, _, _ = _rt_case(rng)
        optional = [k for k in RT_TOOLS[tool]["properties"] if k not in args]
        sys_t = "<tools>\n" + json.dumps(_rt_tools(rng, tool)) + "\n</tools>"
        return {"area": "tool", "kind": "tool", "messages": [["system", sys_t], ["user", q]],
                "calls": [{"name": tool, "arguments": args, "optional": optional}]}
    sys_t = "<tools>\n" + json.dumps(_rt_tools(rng)) + "\n</tools>"  # tools present, none needed
    c, k = rng.choice(CAPITALS)
    q = rng.choice(["Hi! How are you?", "Thanks for the help!", f"What is the capital of {c}?", "Good evening!",
                    f"What is {rng.randint(2, 20)} plus {rng.randint(2, 20)}?"])
    return {"area": "tool", "kind": "tool", "messages": [["system", sys_t], ["user", q]], "calls": []}


def live_data(rng) -> dict:
    q, _, _, _, _ = _rt_case(rng)
    return {"area": "refusal", "kind": "refusal", "messages": [["user", q]], "expect": "no_live"}


def generate(n_time, n_binding, n_tool, n_live, seed=20261130, repo_root="."):
    rng = random.Random(seed)
    names = _names(repo_root)
    out = [time_problem(rng) for _ in range(n_time)] + [binding(rng, names) for _ in range(n_binding)]
    out += [tool_task(rng) for _ in range(n_tool)] + [live_data(rng) for _ in range(n_live)]
    return out
