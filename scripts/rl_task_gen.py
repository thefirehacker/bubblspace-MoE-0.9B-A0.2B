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
  if_answer        (27 Nov) an instruction constraint on a question with a known answer (time problem, arithmetic,
                   capital): reward needs the constraint AND the right answer
  code_task        (27 Nov) simple Python functions with tests computed by a reference implementation (MBPP-style),
                   to grow the code pool

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


# requests no listed tool can serve (BFCL "irrelevance"-style): answer directly, don't call (27 Nov: run 02b scored 59%
# on BFCL's no-tool-needed questions vs 75% for run 01)
NO_TOOL_REQUESTS = [
    "Write a short poem about the sea.", "Explain what photosynthesis is in two sentences.", "Give me three tips for better sleep.",
    "Tell me a fun fact about octopuses.", "What does 'serendipity' mean?", "Summarize the story of Cinderella in one sentence.",
    "Suggest a name for a black cat.", "Write a haiku about autumn.", "How do I boil an egg?", "What is a prime number?",
    "Explain recursion to a ten-year-old.", "Give me a synonym for 'happy'.", "Write a one-line birthday wish for my friend.",
    "What are the three states of matter?", "How many legs does a spider have?", "Why is the sky blue?",
    "Recommend a good habit for learning a language.", "What is the difference between a lake and a sea?",
    "Write a Python one-liner that prints hello.", "Who wrote Romeo and Juliet?", "What is the boiling point of water in Celsius?",
    "Can you help me plan a simple vegetarian dinner?", "List three primary colours.", "Tell me a short joke about computers.",
    "Translate 'thank you' into French.", "What is 15% of 200?", "Is a tomato a fruit or a vegetable?", "What is the largest ocean?",
]


def tool_task(rng) -> dict:
    """Tool choice among 1-5 distractors (exact call), parallel calls (two cities), or no tool fits (answer directly)."""
    k = rng.random()
    if k < 0.45:  # one matching tool among distractors: exact call expected
        q, tool, args, _, _ = _rt_case(rng)
        optional = [o for o in RT_TOOLS[tool]["properties"] if o not in args]
        names = rng.sample([n for n in RT_TOOLS if n != tool], rng.randint(1, 5)) + [tool]
        rng.shuffle(names)
        sys_t = "<tools>\n" + json.dumps(_tool_defs(names)) + "\n</tools>"
        return {"area": "tool", "kind": "tool", "messages": [["system", sys_t], ["user", q]],
                "calls": [{"name": tool, "arguments": args, "optional": optional}]}
    if k < 0.60:  # parallel: the same tool for two cities -> two calls
        c1, c2 = rng.sample(list(RT_CITIES), 2)
        tool, key, q = rng.choice([
            ("get_weather", "city", rng.choice([f"What's the weather in {c1} and in {c2}?", f"Weather for {c1} and {c2}, please."])),
            ("get_current_time", "timezone", rng.choice([f"What time is it in {c1} and in {c2}?", f"Current time in {c1} and {c2}?"]))])
        val = (lambda c: c) if key == "city" else (lambda c: RT_CITIES[c])
        names = rng.sample([n for n in RT_TOOLS if n != tool], rng.randint(1, 4)) + [tool]
        rng.shuffle(names)
        opt = [o for o in RT_TOOLS[tool]["properties"] if o != key]
        sys_t = "<tools>\n" + json.dumps(_tool_defs(names)) + "\n</tools>"
        return {"area": "tool", "kind": "tool", "messages": [["system", sys_t], ["user", q]],
                "calls": [{"name": tool, "arguments": {key: val(c1)}, "optional": opt},
                          {"name": tool, "arguments": {key: val(c2)}, "optional": opt}]}
    # no tool fits: answer directly (the answer is checked when it is known)
    names = rng.sample([n for n in RT_TOOLS if n != "calculator"], rng.randint(1, 5))
    sys_t = "<tools>\n" + json.dumps(_tool_defs(names)) + "\n</tools>"
    c, cap = rng.choice(CAPITALS)
    x, y = rng.randint(2, 20), rng.randint(2, 20)
    options = [("Hi! How are you?", None), ("Thanks for the help!", None), ("Good evening!", None),
               (f"What is the capital of {c}?", cap.split(",")[0]), (f"What is {x} plus {y}?", str(x + y))]
    options += [(rng.choice(NO_TOOL_REQUESTS), None) for _ in range(5)]  # open requests are most of the no-call share
    q, ans = rng.choice(options)
    task = {"area": "tool", "kind": "tool", "messages": [["system", sys_t], ["user", q]], "calls": []}
    if ans is not None:
        task["answer"] = ans  # no call AND the right answer (27 Nov fix)
    return task


def _tool_defs(names):
    return [{"type": "function", "function": {"name": n, "description": RT_TOOLS[n]["description"],
             "parameters": {"type": "dict", "properties": RT_TOOLS[n]["properties"], "required": RT_TOOLS[n]["required"]}}}
            for n in names]


def live_data(rng) -> dict:
    q, _, _, _, _ = _rt_case(rng)
    return {"area": "refusal", "kind": "refusal", "messages": [["user", q]], "expect": "no_live"}


# ---------------------------------------------------------------- constraint + known answer
IF_CONSTRAINTS = [  # (instruction text, constraint, answer kinds it is safe with)
    ("Do not use any commas in your answer.", {"func_name": "validate_no_commas"}, ("number", "time", "text")),
    ("Wrap your entire answer in double quotation marks.", {"func_name": "validate_quotation"}, ("number", "time", "text")),
    ('End your answer with the exact phrase "Hope this helps!"', {"func_name": "validate_end", "end_phrase": "Hope this helps!"},
     ("number", "time", "text")),
    ('Include the word "result" in your answer.', {"func_name": "verify_keywords", "keyword_list": ["result"]}, ("number", "time", "text")),
    ("Give your answer a title wrapped in double angular brackets, such as <<my title>>.", {"func_name": "validate_title"},
     ("number", "time", "text")),
    ('Answer in JSON format with a single key "answer".', {"func_name": "validate_json_format"}, ("number", "time", "text")),
    ("Your entire answer must be in lowercase letters.", {"func_name": "validate_lowercase"}, ("text",)),
    ("At the end of your answer, add a postscript starting with P.S.", {"func_name": "verify_postscript", "postscript_marker": "P.S."},
     ("text",)),
]


def if_answer(rng) -> dict:
    k = rng.random()
    if k < 0.4:
        base = time_problem(rng)
        q, ans, kind = base["messages"][0][1], base["answer"], ("time" if base["kind"] == "time" else "number")
    elif k < 0.7:
        a, b = rng.randint(3, 60), rng.randint(3, 40)
        q, ans = rng.choice([(f"What is {a} times {b}?", a * b), (f"What is {a} plus {b}?", a + b),
                             (f"What is {a + b} minus {b}?", a), (f"What is {a * b} divided by {b}?", a)])
        ans, kind = str(ans), "number"
    else:
        c, cap = rng.choice([x for x in CAPITALS if "," not in x[1] and x[0] not in ("India", "France")])
        q, ans, kind = f"What is the capital of {c}?", cap, "text"
    text, cons, kinds = rng.choice([x for x in IF_CONSTRAINTS if kind in x[2]])
    prompt = f"{q} {text}" if rng.random() < 0.5 else f"{text} {q}"
    return {"area": "ifeval_answer", "kind": "ifeval_answer", "messages": [["user", prompt]], "constraint": cons,
            "answer": ans, "answer_kind": kind}


# ---------------------------------------------------------------- simple code tasks, tests from a reference implementation
def _words_list(rng, n):
    pool = ["apple", "river", "stone", "cloud", "tiger", "level", "radar", "music", "green", "orbit", "noon", "piano", "table"]
    return [rng.choice(pool) for _ in range(n)]


def _ints(rng, n, lo=-20, hi=50):
    return [rng.randint(lo, hi) for _ in range(n)]


CODE_TEMPLATES = [  # (function name, signature, description, reference, input maker)
    ("reverse_string", "s", "returns the string s reversed", lambda s: s[::-1], lambda r: (r.choice(_words_list(r, 1)) + r.choice(["", "s", "ing"]),)),
    ("count_vowels", "s", "returns the number of vowels (a, e, i, o, u) in the string s", lambda s: sum(c in "aeiou" for c in s),
     lambda r: (" ".join(_words_list(r, r.randint(1, 4))),)),
    ("is_palindrome", "s", "returns True if the string s reads the same backwards, otherwise False", lambda s: s == s[::-1],
     lambda r: (r.choice(_words_list(r, 1)),)),
    ("sum_list", "nums", "returns the sum of the numbers in the list nums", sum, lambda r: (_ints(r, r.randint(0, 7)),)),
    ("max_of_list", "nums", "returns the largest number in the non-empty list nums", max, lambda r: (_ints(r, r.randint(1, 7)),)),
    ("second_largest", "nums", "returns the second largest distinct value in the list nums",
     lambda n: sorted(set(n))[-2], lambda r: (list(dict.fromkeys(_ints(r, 6)))[:6] + [999, 1000],)),
    ("count_occurrences", "items, x", "returns how many times x appears in the list items", lambda items, x: items.count(x),
     lambda r: (lambda l: (l, r.choice(l)))(_ints(r, 8, 0, 5))),
    ("remove_duplicates", "items", "returns a new list with duplicates removed, keeping the first occurrence order",
     lambda items: list(dict.fromkeys(items)), lambda r: (_ints(r, 8, 0, 6),)),
    ("factorial", "n", "returns n! (the factorial of the non-negative integer n)", lambda n: __import__("math").factorial(n),
     lambda r: (r.randint(0, 10),)),
    ("fibonacci", "n", "returns the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1",
     lambda n: (lambda f: f(f, n))(lambda f, k: k if k < 2 else f(f, k - 1) + f(f, k - 2)), lambda r: (r.randint(0, 15),)),
    ("sum_digits", "n", "returns the sum of the decimal digits of the non-negative integer n", lambda n: sum(map(int, str(n))),
     lambda r: (r.randint(0, 99999),)),
    ("is_prime", "n", "returns True if n is a prime number, otherwise False",
     lambda n: n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)), lambda r: (r.randint(0, 60),)),
    ("gcd", "a, b", "returns the greatest common divisor of the positive integers a and b", lambda a, b: __import__("math").gcd(a, b),
     lambda r: (r.randint(1, 60), r.randint(1, 60))),
    ("word_count", "s", "returns the number of words in the string s (words are separated by spaces)", lambda s: len(s.split()),
     lambda r: (" ".join(_words_list(r, r.randint(0, 6))),)),
    ("capitalize_words", "s", "returns s with the first letter of every word in upper case", lambda s: " ".join(w.capitalize() for w in s.split(" ")),
     lambda r: (" ".join(_words_list(r, r.randint(1, 4))),)),
    ("flatten", "lists", "returns a single list with all elements of the list of lists, in order", lambda ls: [x for l in ls for x in l],
     lambda r: ([_ints(r, r.randint(0, 3), 0, 9) for _ in range(r.randint(1, 4))],)),
    ("even_numbers", "nums", "returns a list of the even numbers in nums, in their original order", lambda n: [x for x in n if x % 2 == 0],
     lambda r: (_ints(r, 7),)),
    ("square_list", "nums", "returns a list with the square of every number in nums", lambda n: [x * x for x in n], lambda r: (_ints(r, 5, -9, 9),)),
    ("merge_sorted", "a, b", "returns one sorted list containing all elements of the sorted lists a and b",
     lambda a, b: sorted(a + b), lambda r: (sorted(_ints(r, 4)), sorted(_ints(r, 3)))),
    ("char_frequency", "s", "returns a dict mapping each character of s to the number of times it appears",
     lambda s: {c: s.count(c) for c in dict.fromkeys(s)}, lambda r: (r.choice(_words_list(r, 1)),)),
    ("common_elements", "a, b", "returns a sorted list of the distinct values that appear in both lists a and b",
     lambda a, b: sorted(set(a) & set(b)), lambda r: (_ints(r, 6, 0, 9), _ints(r, 6, 0, 9))),
    ("rotate_left", "items, k", "returns the list items rotated to the left by k positions (k may exceed the length)",
     lambda items, k: items[k % len(items):] + items[:k % len(items)] if items else [], lambda r: (_ints(r, 5, 0, 9), r.randint(0, 7))),
    ("binary_to_decimal", "s", "converts the binary string s (like '101') to its integer value", lambda s: int(s, 2),
     lambda r: (bin(r.randint(0, 255))[2:],)),
    ("decimal_to_binary", "n", "returns the binary representation of the non-negative integer n as a string without a prefix",
     lambda n: bin(n)[2:], lambda r: (r.randint(0, 255),)),
    ("count_long_words", "s, n", "returns how many words in the string s are longer than n characters",
     lambda s, n: sum(len(w) > n for w in s.split()), lambda r: (" ".join(_words_list(r, r.randint(1, 6))), r.randint(3, 5))),
]


def code_task(rng) -> dict:
    name, sig, desc, ref, make = rng.choice(CODE_TEMPLATES)
    cases, seen = [], set()
    while len(cases) < 4:
        args = make(rng)
        key = repr(args)
        if key in seen:
            if len(seen) > 30:
                break
            continue
        seen.add(key)
        cases.append(f"assert {name}({', '.join(repr(x) for x in args)}) == {ref(*args)!r}")
    prompt = (f"Write a Python function {name}({sig}) that {desc}.\nYour code should pass this test:\n{cases[0]}\n"
              "Write the Python function in a ```python code block.")
    return {"area": "code", "kind": "code", "source": f"generated/{name}", "messages": [["user", prompt]],
            "tests": "\n".join(cases), "setup": ""}


def generate(n_time, n_binding, n_tool, n_live, seed=20261130, repo_root=".", n_if_answer=0, n_code=0):
    rng = random.Random(seed)
    names = _names(repo_root)
    out = [time_problem(rng) for _ in range(n_time)] + [binding(rng, names) for _ in range(n_binding)]
    out += [tool_task(rng) for _ in range(n_tool)] + [live_data(rng) for _ in range(n_live)]
    out += [if_answer(rng) for _ in range(n_if_answer)] + [code_task(rng) for _ in range(n_code)]
    return out
