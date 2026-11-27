"""Phase 3 SFT data: our own small sets (doc/plan-04-phase-3-sft.md §3.1), written as JSONL of chat turns.

  B  own_facts     Nemotron Fact-Seeking quizzes -> direct answers. ~80%: question WITHOUT options -> plain answer;
                   ~20%: options given in the prompt -> "C. answer". Teaches: quiz format only when asked (2.5a habit).
  C  own_math      arithmetic word problems with short written steps: time, two-step counting, ages, prices, days,
                   bare arithmetic in words. (OLMo-2 1B Instruct solved these by writing the steps.)
  D  own_binding   who-did-what Q&A: passives, "respectively", exclusion, swaps, before/after, chains, family
                   reversals, fit/inverse fit.
  I  own_identity  short BubblSpace identity conversations.

Names used by the hand tests (doc/handtest-01) and the binding probes (benchmark/data/probefinding-*) are excluded,
and no generated prompt equals a hand-test prompt, so those evals stay clean.

    python scripts/phase3_gen_own_sets.py --facts_parquet <part_000000.parquet> --out ~/phase3_raw/own
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

HANDTEST_NAMES = {"Maria", "Joon", "Ahmed", "Sofia", "Priya", "Hassan", "Kofi", "Naledi", "Ravi", "Meera", "Sam", "Lena",
                  "Omar", "Ali", "Zara", "Tara", "Ben", "Kiran", "Jun", "Elena", "Nina", "Asha", "Rohan", "Daniel", "Lee",
                  "Aarav", "Pia", "Anders", "Sara", "Dmitri", "Basil", "Jack", "Morris", "Bob", "Ronald", "Tom", "Riya"}
MALE = ["Arjun", "Mateo", "Kenji", "Lucas", "Tariq", "Felix", "Nikhil", "Oscar", "Hugo", "Ibrahim", "Leo", "Marcus", "Theo",
        "Diego", "Kwame", "Aiden", "Rahul", "Victor", "Emil", "Yusuf", "Caleb", "Andre", "Kai", "Samir", "Bruno", "Ethan",
        "Gabriel", "Hiro", "Isaac", "Joel", "Karan", "Liam", "Milan", "Noah", "Pavel", "Quentin", "Rafael", "Stefan",
        "Abel", "Adrian", "Akira", "Alonso", "Amit", "Anton", "Arash", "Aryan", "Bashir", "Bilal", "Boris", "Cedric", "Chidi",
        "Cyrus", "Dario", "Dev", "Dorian", "Elias", "Emeka", "Enzo", "Farid", "Florian", "Gautam", "Gideon", "Goran", "Hamza",
        "Harun", "Henrik", "Ilya", "Imran", "Ivo", "Jabari", "Jalen", "Jasper", "Jonas", "Kaito", "Kamal", "Kelvin", "Kunal",
        "Lars", "Lorenzo", "Luka", "Malik", "Manu", "Matteo", "Mihail", "Mohan", "Nils", "Obi", "Omid", "Orlando", "Pedro",
        "Pranav", "Raj", "Reza", "Rohit", "Rustam", "Santiago", "Sergei", "Shiv", "Silas", "Soren", "Tadeo", "Taro", "Tomas",
        "Ulrich", "Vikram", "Wade", "Xavi", "Yohan", "Zaid", "Zoran"]
FEMALE = ["Ananya", "Chloe", "Fatima", "Grace", "Hana", "Isla", "Julia", "Keiko", "Leila", "Mira", "Nadia", "Olivia", "Rosa",
          "Sana", "Talia", "Uma", "Vera", "Wen", "Yara", "Zoe", "Amara", "Beatriz", "Clara", "Divya", "Esther", "Freya",
          "Gita", "Helena", "Ingrid", "Jasmine", "Kavya", "Lucia", "Maya", "Nora", "Paula", "Rhea", "Sienna", "Tanvi",
          "Adaeze", "Agnes", "Aiko", "Alina", "Anika", "Aria", "Astrid", "Ayesha", "Bianca", "Camila", "Carmen", "Dalia", "Delphine",
          "Dina", "Eden", "Elif", "Emi", "Farah", "Fiona", "Gabriela", "Greta", "Hadia", "Harini", "Ilse", "Imani", "Irene",
          "Ishita", "Jana", "Jun-hee", "Kalinda", "Kamila", "Kenya", "Lale", "Lara", "Lin", "Lorena", "Mabel", "Mahsa", "Malia",
          "Marisol", "Meiling", "Naomi", "Neha", "Nkechi", "Odette", "Pari", "Petra", "Priyanka", "Quinn", "Rania", "Renata",
          "Ruth", "Saanvi", "Selin", "Shreya", "Soraya", "Svetlana", "Tamar", "Thandi", "Ursula", "Valentina", "Wanjiru",
          "Ximena", "Yasmin", "Yuki", "Zainab", "Zelda"]
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_probe_names(repo_root):
    names = set()
    for f in glob.glob(str(Path(repo_root) / "benchmark/data/probefinding-winograndestyle/*.json")):
        try:
            for c in json.load(open(f)).get("cases", []):
                names |= set(re.findall(r"\b[A-Z][a-z]+\b", c.get("ctxA", "") + " " + c.get("ctxB", "")))
        except Exception:
            pass
    return names


# ---------------------------------------------------------------- B: facts
QUIZ = re.compile(r"^(?P<q>.+?)\nA\. (?P<A>.+?)\nB\. (?P<B>.+?)\nC\. (?P<C>.+?)\nD\. (?P<D>.+?)\nAnswer: (?P<L>[ABCD])\. ?(?P<ans>.+?)\s*$", re.S)


def gen_facts(parquet, n, rng):
    out = []
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
        for r in batch.to_pylist():
            m = QUIZ.match((r["text"] or "").strip())
            if not m or "\n" in m["q"] or len(m["q"]) > 400:
                continue
            ans = m["ans"].strip().rstrip(".")
            if rng.random() < 0.8:
                out.append([["user", m["q"].strip()], ["assistant", ans[0].upper() + ans[1:] + "."]])
            else:
                opts = "\n".join(f"{k}. {m[k].strip()}" for k in "ABCD")
                out.append([["user", f"{m['q'].strip()}\n{opts}"], ["assistant", f"{m['L']}. {ans}"]])
            if len(out) >= n:
                return out
    return out


# ---------------------------------------------------------------- B2 (run 02): facts asked as commands, in formats
# Run 01 refused harmless commands ("give me capital of India and France in a json" -> "I can't provide real-time data").
# These rows ask for facts as commands and in requested formats (JSON, CSV, list, one line) and answer plainly.
# India and France are left out of the capital table: they are in the hand tests.
CAPITALS = [("Japan", "Tokyo"), ("China", "Beijing"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("Portugal", "Lisbon"), ("United Kingdom", "London"), ("Ireland", "Dublin"), ("Russia", "Moscow"),
            ("Canada", "Ottawa"), ("United States", "Washington, D.C."), ("Mexico", "Mexico City"), ("Brazil", "Brasília"),
            ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"), ("Colombia", "Bogotá"),
            ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("South Africa", "Pretoria"),
            ("Ethiopia", "Addis Ababa"), ("Morocco", "Rabat"), ("Ghana", "Accra"), ("Australia", "Canberra"),
            ("New Zealand", "Wellington"), ("Indonesia", "Jakarta"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
            ("Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"), ("South Korea", "Seoul"), ("Nepal", "Kathmandu"),
            ("Bangladesh", "Dhaka"), ("Sri Lanka", "Sri Jayawardenepura Kotte"), ("Pakistan", "Islamabad"),
            ("Afghanistan", "Kabul"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"),
            ("Turkey", "Ankara"), ("Greece", "Athens"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
            ("Switzerland", "Bern"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
            ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Hungary", "Budapest"),
            ("Czech Republic", "Prague"), ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Israel", "Jerusalem"),
            ("Jordan", "Amman"), ("Qatar", "Doha"), ("United Arab Emirates", "Abu Dhabi"), ("Cuba", "Havana"),
            ("Venezuela", "Caracas"), ("Iceland", "Reykjavík"), ("Mongolia", "Ulaanbaatar"), ("Singapore", "Singapore")]
CMD_ONE = ["Give me the answer: {q}", "Tell me: {q}", "Answer this: {q}", "Quick question: {q}", "{q} Just the answer, please.",
           "I need the answer to this: {q}", "Please answer: {q}", "Answer in one line: {q}"]


def _cap_rows(rng):
    pairs = rng.sample(CAPITALS, rng.randint(1, 4))
    names = [c for c, _ in pairs]
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    fmt = rng.choice(["json", "csv", "list", "plain", "table", "lines"])
    verb = rng.choice(["give me", "Give me", "tell me", "List", "Write", "I need"])
    if fmt == "json":
        q = rng.choice([f"{verb} the capital of {lst} in a json", f"{verb} the capitals of {lst} as JSON",
                        f"capital of {lst} in json format"])
        a = "```json\n" + json.dumps({c: k for c, k in pairs}, ensure_ascii=False, indent=2) + "\n```"
    elif fmt == "csv":
        q = rng.choice([f"{verb} only comma separated values of country and capital for {lst}",
                        f"{verb} the capitals of {lst} as CSV", f"CSV of country,capital for {lst}"])
        a = "country,capital\n" + "\n".join(f'{c},"{k}"' if "," in k else f"{c},{k}" for c, k in pairs)
    elif fmt == "list":
        q = rng.choice([f"{verb} a list of the capitals of {lst}", f"list the capital cities of {lst}"])
        a = "\n".join(f"- {c}: {k}" for c, k in pairs)
    elif fmt == "table":
        q = f"{verb} a table of {lst} and their capitals"
        a = "| Country | Capital |\n|---|---|\n" + "\n".join(f"| {c} | {k} |" for c, k in pairs)
    elif fmt == "lines":
        q = f"{verb} the capital of {lst}, one per line, nothing else"
        a = "\n".join(k for _, k in pairs)
    else:
        q = rng.choice([f"{verb} the capital of {lst}", f"capital of {lst}", f"what are the capitals of {lst}?"])
        a = " ".join(f"The capital of {c} is {k}." for c, k in pairs)
    return [["user", q], ["assistant", a]]


def gen_facts_cmd(parquet, n, rng, skip):
    """Fact-Seeking questions not used by gen_facts (skip the first `skip` matches), asked as commands or in batches."""
    qa, seen = [], 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096, columns=["text"]):
        for r in batch.to_pylist():
            m = QUIZ.match((r["text"] or "").strip())
            if not m or "\n" in m["q"] or len(m["q"]) > 300:
                continue
            seen += 1
            if seen <= skip:
                continue
            ans = m["ans"].strip().rstrip(".")
            qa.append((m["q"].strip(), ans[0].upper() + ans[1:]))
        if len(qa) >= n * 2:
            break
    out, i = [], 0
    while len(out) < n and i < len(qa):
        kind = rng.random()
        if kind < 0.25:
            out.append(_cap_rows(rng))
            continue
        if kind < 0.65:
            q, a = qa[i]; i += 1
            out.append([["user", rng.choice(CMD_ONE).format(q=q)], ["assistant", a + "."]])
            continue
        k = rng.randint(2, 3)
        batch_qa = qa[i:i + k]; i += k
        if len(batch_qa) < k:
            break
        numbered = "\n".join(f"{j + 1}. {q}" for j, (q, _) in enumerate(batch_qa))
        fmt = rng.choice(["json", "list", "csv"])
        if fmt == "json":
            u = rng.choice(["Answer these as JSON:", "Give me the answers in a json:", "Return the answers as a JSON object:"])
            a = "```json\n" + json.dumps({str(j + 1): x for j, (_, x) in enumerate(batch_qa)}, ensure_ascii=False, indent=2) + "\n```"
        elif fmt == "list":
            u = rng.choice(["Answer each of these:", "Give me short answers:", "Answer these questions briefly:"])
            a = "\n".join(f"{j + 1}. {x}" for j, (_, x) in enumerate(batch_qa))
        else:
            u = rng.choice(["Give me only the answers, comma separated:", "Answers only, separated by commas:"])
            a = ", ".join(x for _, x in batch_qa)
        out.append([["user", f"{u}\n{numbered}"], ["assistant", a]])
    return out


# ---------------------------------------------------------------- R (run 02b): live data, with and without tools
# Run 02 (no CoCoNot) made up live facts ("The current time in London is 10:00 PM"). These rows teach the honest answer:
#   no tools          -> "I don't have access to live data ... here is where to check"
#   matching tool     -> a tool call with the right arguments (distractor tools listed too)
#   tools, not needed -> answer directly (greetings, fixed facts): run 01 called get_weather for "Hi! How are you?"
# London, Mumbai and the exact eval phrasings (scripts/eval_chat.py REALTIME) are left out.
RT_CITIES = {"Tokyo": "Asia/Tokyo", "Paris": "Europe/Paris", "New York": "America/New_York", "Sydney": "Australia/Sydney",
             "Berlin": "Europe/Berlin", "Dubai": "Asia/Dubai", "Singapore": "Asia/Singapore", "Toronto": "America/Toronto",
             "Chennai": "Asia/Kolkata", "Bengaluru": "Asia/Kolkata", "Cairo": "Africa/Cairo", "Seoul": "Asia/Seoul",
             "Madrid": "Europe/Madrid", "Nairobi": "Africa/Nairobi", "Mexico City": "America/Mexico_City",
             "Los Angeles": "America/Los_Angeles", "Jakarta": "Asia/Jakarta", "Istanbul": "Europe/Istanbul",
             "Lagos": "Africa/Lagos", "Rome": "Europe/Rome", "Chicago": "America/Chicago", "Kolkata": "Asia/Kolkata"}
RT_TOOLS = {
    "get_current_time": {"description": "Get the current date and time in a timezone.",
                         "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Tokyo"}},
                         "required": ["timezone"]},
    "get_weather": {"description": "Get the current weather for a city.",
                    "properties": {"city": {"type": "string", "description": "City name"},
                                   "unit": {"type": "string", "description": "celsius or fahrenheit"}}, "required": ["city"]},
    "get_news": {"description": "Get the latest news headlines.",
                 "properties": {"topic": {"type": "string", "description": "e.g. technology, sports, world"},
                                "country": {"type": "string", "description": "Country name (optional)"}}, "required": ["topic"]},
    "get_stock_price": {"description": "Get the latest price of a stock.",
                        "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}},
                        "required": ["symbol"]},
    "get_exchange_rate": {"description": "Get the current exchange rate between two currencies.",
                          "properties": {"base": {"type": "string", "description": "3-letter code, e.g. USD"},
                                         "target": {"type": "string", "description": "3-letter code, e.g. EUR"}},
                          "required": ["base", "target"]},
    "get_crypto_price": {"description": "Get the current price of a cryptocurrency in US dollars.",
                         "properties": {"coin": {"type": "string", "description": "e.g. bitcoin, ethereum"}}, "required": ["coin"]},
    "calculator": {"description": "Evaluate an arithmetic expression.",
                   "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    "translate_text": {"description": "Translate text into another language.",
                       "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
                       "required": ["text", "target_language"]},
}
STOCKS = [("Apple", "AAPL"), ("Microsoft", "MSFT"), ("Tesla", "TSLA"), ("Amazon", "AMZN"), ("Nvidia", "NVDA"), ("Infosys", "INFY")]
CURRS = [("US dollars", "USD"), ("euros", "EUR"), ("Japanese yen", "JPY"), ("British pounds", "GBP"), ("Indian rupees", "INR"),
         ("Australian dollars", "AUD"), ("Canadian dollars", "CAD")]
COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
TOPICS = ["technology", "sports", "world", "business", "science", "health"]


def _rt_case(rng):
    """One live-data question: (user text, tool name, arguments, what it is, where to check)."""
    k = rng.choice(["time", "weather", "news", "stock", "fx", "crypto"])
    if k == "time":
        c = rng.choice(list(RT_CITIES))
        q = rng.choice([f"What time is it in {c}?", f"Tell me the time in {c} right now.", f"Current time in {c}?",
                        f"What's the local time in {c} at the moment?"])
        return q, "get_current_time", {"timezone": RT_CITIES[c]}, f"the current time in {c}", "a world clock or your phone's clock app"
    if k == "weather":
        c = rng.choice(list(RT_CITIES))
        q = rng.choice([f"What's the weather in {c} today?", f"Is it raining in {c} right now?", f"How hot is it in {c} now?",
                        f"Weather in {c}?", f"Should I take an umbrella in {c} today?"])
        return q, "get_weather", {"city": c}, f"the current weather in {c}", "a weather app or your local weather service"
    if k == "news":
        tp = rng.choice(TOPICS)
        q = rng.choice([f"What are the latest {tp} headlines?", f"Any {tp} news today?", f"Tell me the top {tp} stories right now."])
        return q, "get_news", {"topic": tp}, f"today's {tp} news", "a news website or app"
    if k == "stock":
        n, s = rng.choice(STOCKS)
        q = rng.choice([f"What's {n}'s stock price right now?", f"How much is {s} trading at today?", f"Current {n} share price?"])
        return q, "get_stock_price", {"symbol": s}, f"the current {n} share price", "a finance website or your broker's app"
    if k == "fx":
        (n1, c1), (n2, c2) = rng.sample(CURRS, 2)
        q = rng.choice([f"What's the exchange rate from {n1} to {n2} today?", f"How many {n2} is one {c1} right now?",
                        f"Current {c1} to {c2} rate?"])
        return q, "get_exchange_rate", {"base": c1, "target": c2}, f"today's {c1} to {c2} exchange rate", "a currency converter or your bank"
    coin = rng.choice(COINS)
    q = rng.choice([f"What's the price of {coin} right now?", f"How much is one {coin} worth today?", f"Current {coin} price?"])
    return q, "get_crypto_price", {"coin": coin}, f"the current price of {coin}", "a crypto exchange or price tracker"


def _rt_tools(rng, needed=None):
    names = rng.sample([n for n in RT_TOOLS if n != needed], rng.randint(1, 3)) + ([needed] if needed else [])
    rng.shuffle(names)
    return [{"type": "function", "function": {"name": n, "description": RT_TOOLS[n]["description"],
             "parameters": {"type": "dict", "properties": RT_TOOLS[n]["properties"], "required": RT_TOOLS[n]["required"]}}}
            for n in names]


def gen_realtime(n, rng):
    no_live = ["I don't have access to live data, so I can't tell you {what}. You can check {where}.",
               "I can't look up {what} because I don't have live information. {Where} will have it.",
               "I don't have real-time access, so I don't know {what}. Please check {where}."]
    greet = [("Hi! How are you?", "I'm doing well, thank you! How can I help you today?"),
             ("Hello", "Hello! What can I do for you?"), ("Good morning!", "Good morning! How can I help?"),
             ("Thanks a lot!", "You're welcome! Let me know if you need anything else."),
             ("Who are you?", "I'm BubblSpace MoE-0.9B-A0.2B, a small open language model trained from scratch by BubblSpace.")]
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:      # no tools: say there's no live access
            q, _, _, what, where = _rt_case(rng)
            a = rng.choice(no_live).format(what=what, where=where, Where=where[0].upper() + where[1:])
            out.append([["user", q], ["assistant", a]])
        elif kind < 0.70:    # matching tool available: call it
            q, tool, args, _, _ = _rt_case(rng)
            sys_t = "<tools>\n" + json.dumps(_rt_tools(rng, tool)) + "\n</tools>"
            call = "<tool_call>\n" + json.dumps({"name": tool, "arguments": args}) + "\n</tool_call>"
            out.append([["system", sys_t], ["user", q], ["assistant", call]])
        else:                # tools available but not needed: answer directly
            sys_t = "<tools>\n" + json.dumps(_rt_tools(rng)) + "\n</tools>"
            if rng.random() < 0.4:
                q, a = rng.choice(greet)
            elif rng.random() < 0.6:
                c, k = rng.choice(CAPITALS)
                q, a = rng.choice([(f"What is the capital of {c}?", f"The capital of {c} is {k}."),
                                   (f"capital of {c}", f"The capital of {c} is {k}.")])
            else:
                x, y = rng.randint(2, 30), rng.randint(2, 30)
                q, a = f"What is {x} plus {y}?", f"{x} + {y} = {x + y}."
            out.append([["system", sys_t], ["user", q], ["assistant", a]])
    return out


# ---------------------------------------------------------------- C: arithmetic with steps
def clock(h):
    return (h - 1) % 12 + 1


def gen_math(n, rng, names):
    events = ["concert", "class", "football match", "workshop", "lecture", "party", "flight", "meeting", "game", "show"]
    objs = ["marbles", "stickers", "pencils", "cookies", "stamps", "coins", "books", "shells", "cards", "oranges"]
    items = [("notebook", 3), ("pen", 2), ("ticket", 5), ("sandwich", 4), ("toy car", 6), ("bottle of juice", 2)]
    out = []
    while len(out) < n:
        k = rng.randrange(8)
        a = rng.choice(names)
        if k == 0:  # time addition, 12-hour clock
            h, d, ev = rng.randint(1, 11), rng.randint(1, 6), rng.choice(events)
            e = clock(h + d)
            q = rng.choice([f"The {ev} began at {h} o'clock and lasted {d} hours. What time did it end?",
                            f"If a {ev} starts at {h} and goes on for {d} hours, when does it finish?"])
            s = f"{h} + {d} = {h + d}" + (f", which is {e} on a 12-hour clock" if h + d > 12 else "")
            out.append([["user", q], ["assistant", f"{s}. It ends at {e} o'clock."]])
        elif k == 1:  # two-step counting
            x = rng.randint(5, 20); g = rng.randint(1, x // 2); b = rng.randint(1, x - g - 1)
            o, other = rng.choice(objs), rng.choice([m for m in names if m != a])
            act, actp = rng.choice([("lost", "losing"), ("used", "using"), ("sold", "selling"), ("ate", "eating")])
            if o in ("marbles", "stamps", "coins", "cards", "shells", "books") and act == "ate":
                act, actp = "lost", "losing"
            q = f"{a} had {x} {o}, gave {g} to {other} and then {act} {b}. How many {o} does {a} have now?"
            ans = f"{a} started with {x}. After giving {g} away: {x} − {g} = {x - g}. After {actp} {b}: {x - g} − {b} = {x - g - b}. {a} has {x - g - b} {o} now."
            out.append([["user", q], ["assistant", ans]])
        elif k == 2:  # one-step add/subtract
            x, y, o = rng.randint(2, 30), rng.randint(1, 15), rng.choice(objs)
            if rng.random() < 0.5:
                out.append([["user", f"{a} has {x} {o} and finds {y} more. How many {o} does {a} have?"],
                            ["assistant", f"{x} + {y} = {x + y}. {a} has {x + y} {o}."]])
            else:
                y = rng.randint(1, x - 1)
                left = x - y
                noun = o if left != 1 else {"cookies": "cookie", "pencils": "pencil", "marbles": "marble", "stickers": "sticker",
                                             "stamps": "stamp", "coins": "coin", "books": "book", "shells": "shell",
                                             "cards": "card", "oranges": "orange"}[o]
                out.append([["user", f"There are {x} {o} in a box and {y} are taken out. How many are left?"],
                            ["assistant", f"{x} − {y} = {left}. There {'is' if left == 1 else 'are'} {left} {noun} left."]])
        elif k == 3:  # ages
            age, dlt, b = rng.randint(4, 40), rng.randint(1, 9), rng.choice([m for m in names if m != a])
            if rng.random() < 0.5:
                out.append([["user", f"{a} is {age} years old. {b} is {dlt} years older. How old is {b}?"],
                            ["assistant", f"{age} + {dlt} = {age + dlt}. {b} is {age + dlt} years old."]])
            else:
                age = max(age, dlt + 1)
                out.append([["user", f"{a} is {age} years old and {b} is {dlt} years younger. How old is {b}?"],
                            ["assistant", f"{age} − {dlt} = {age - dlt}. {b} is {age - dlt} years old."]])
        elif k == 4:  # prices
            (it, p), c = rng.choice(items), rng.randint(2, 9)
            plural = it + ("es" if it.endswith(("ch", "sh")) else "s") if not it.startswith("bottle") else "bottles of juice"
            out.append([["user", f"One {it} costs {p} dollars. How much do {c} {plural} cost?"],
                        ["assistant", f"{c} × {p} = {c * p}. They cost {c * p} dollars."]])
        elif k == 5:  # days of the week
            d0, off = rng.randrange(7), rng.randint(1, 6)
            if rng.random() < 0.5:
                d1 = DAYS[(d0 + off) % 7]
                out.append([["user", f"If today is {DAYS[d0]}, what day will it be in {off} day{'s' if off > 1 else ''}?"],
                            ["assistant", f"Counting {off} day{'s' if off > 1 else ''} forward from {DAYS[d0]} gives {d1}."]])
            else:
                d1 = DAYS[(d0 - off) % 7]
                out.append([["user", f"Today is {DAYS[d0]}. What day was it {off} day{'s' if off > 1 else ''} ago?"],
                            ["assistant", f"Counting {off} day{'s' if off > 1 else ''} back from {DAYS[d0]} gives {d1}."]])
        elif k == 6:  # bare arithmetic in words
            x, y = rng.randint(1, 20), rng.randint(1, 20)
            op = rng.choice(["plus", "minus", "times"])
            if op == "minus" and y > x:
                x, y = y, x
            r = {"plus": x + y, "minus": x - y, "times": x * y}[op]
            sym = {"plus": "+", "minus": "−", "times": "×"}[op]
            out.append([["user", f"What is {NUM_WORDS[x]} {op} {NUM_WORDS[y]}?"], ["assistant", f"{x} {sym} {y} = {r}."]])
        else:  # half / double
            x = rng.randint(1, 25) * 2
            if rng.random() < 0.5:
                out.append([["user", f"What is half of {x}?"], ["assistant", f"{x} ÷ 2 = {x // 2}."]])
            else:
                out.append([["user", f"What is double {x // 2}?"], ["assistant", f"{x // 2} × 2 = {x}."]])
    return out


# ---------------------------------------------------------------- D: binding / relations
PASSIVES = [("helped", "needed help", "gave the help"), ("taught", "was the student", "was the teacher"),
            ("beaten", "lost", "won"), ("rescued", "was in danger", "did the rescuing"), ("hired", "got the job", "did the hiring"),
            ("forgiven", "had done something wrong", "did the forgiving"), ("interviewed", "answered the questions", "asked the questions"),
            ("invited", "received the invitation", "sent the invitation"), ("praised", "received the praise", "gave the praise"),
            ("followed", "was in front", "was behind")]
ROLES = ["doctor", "pilot", "teacher", "chef", "lawyer", "painter", "nurse", "engineer", "farmer", "singer"]
REL_INV = {"grandmother": ("grandson", "granddaughter"), "grandfather": ("grandson", "granddaughter"),
           "mother": ("son", "daughter"), "father": ("son", "daughter"), "aunt": ("nephew", "niece"),
           "uncle": ("nephew", "niece"), "teacher": ("student", "student"), "boss": ("employee", "employee")}


def gen_binding(n, rng, males, females):
    names = males + females
    out = []
    while len(out) < n:
        k = rng.randrange(8)
        a, b, c = rng.sample(names, 3)
        if k == 0:  # passive roles
            pp, pat, agt = rng.choice(PASSIVES)
            if rng.random() < 0.5:
                out.append([["user", f"{a} was {pp} by {b}. Who {agt}?"], ["assistant", f"{b}. {a} was {pp} by {b}, so {b} {agt}."]])
            else:
                out.append([["user", f"{a} was {pp} by {b}. Who {pat}?"], ["assistant", f"{a}. {a} was {pp} by {b}, so {a} {pat}."]])
        elif k == 1:  # respectively
            r1, r2 = rng.sample(ROLES, 2)
            ask = rng.choice([(r1, a), (r2, b)])
            out.append([["user", f"{a} and {b} are a {r1} and a {r2}, respectively. Who is the {ask[0]}?"],
                        ["assistant", f"{ask[1]}. \"Respectively\" pairs the names and roles in order, so {a} is the {r1} and {b} is the {r2}."]])
        elif k == 2:  # exclusion
            thing = rng.choice(["coffee", "tea", "spicy food", "swimming", "chess", "jazz", "hiking"])
            order = [a, b, c]
            rng.shuffle(order)
            out.append([["user", f"Of {order[0]}, {order[1]} and {order[2]}, {a} and {b} like {thing}, but {c} does not. Who does not like {thing}?"],
                        ["assistant", f"{c}. {a} and {b} like {thing}; {c} is the one who doesn't."]])
        elif k == 3:  # swap
            c1, c2 = rng.sample(["red", "blue", "green", "yellow", "white", "black"], 2)
            item = rng.choice(["ball", "cup", "key", "coin", "book"])
            x, y = rng.sample(["A", "B", "C", "D"], 2)
            ask = rng.choice([x, y])
            got = c2 if ask == x else c1
            out.append([["user", f"The {c1} {item} is in box {x} and the {c2} {item} is in box {y}. Then the two {item}s are swapped. Which {item} is in box {ask} now?"],
                        ["assistant", f"The {got} {item}. Before the swap box {x} had the {c1} {item} and box {y} the {c2} one; swapping exchanges them."]])
        elif k == 4:  # before / after
            acts = [("had lunch", "having lunch"), ("went to the market", "going to the market"), ("read a book", "reading a book"),
                    ("cleaned the kitchen", "cleaning the kitchen"), ("called a friend", "calling a friend"),
                    ("finished the homework", "finishing the homework"), ("went jogging", "going jogging"),
                    ("cooked dinner", "cooking dinner"), ("watched a film", "watching a film"), ("fed the cat", "feeding the cat")]
            (p1, g1), (p2, g2) = rng.sample(acts, 2)  # true order: p1, then p2
            if rng.random() < 0.5:
                out.append([["user", f"{a} {p2} after {g1}. What did {a} do first?"],
                            ["assistant", f"{a} {p1} first, then {p2}."]])
            else:
                out.append([["user", f"{a} {p1} before {g2}. What did {a} do second?"],
                            ["assistant", f"{a} {p2} second, after {g1}."]])
        elif k == 5:  # transitive chain
            adj, sup, low = rng.choice([("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest")])
            ask = rng.choice(["top", "bottom"])
            out.append([["user", f"{a} is {adj} than {b}, and {b} is {adj} than {c}. Who is the {sup if ask == 'top' else low}?"],
                        ["assistant", f"{a if ask == 'top' else c}. The order is {a}, then {b}, then {c}."]])
        elif k == 6:  # family / role reversal
            rel = rng.choice(list(REL_INV))
            male = rng.random() < 0.5
            b = rng.choice(males if male else females)
            inv = REL_INV[rel][0 if male else 1]
            out.append([["user", f"{a} is {b}'s {rel}. What is {b} to {a}?"], ["assistant", f"{b} is {a}'s {inv}."]])
        else:  # fit / inverse fit
            big, box, badj, sadj = rng.choice([("sofa", "doorway", "big", "narrow"), ("cake", "box", "large", "small"),
                                               ("painting", "frame", "wide", "small"), ("bicycle", "elevator", "long", "small"),
                                               ("bookshelf", "car", "tall", "small")])
            if rng.random() < 0.5:
                out.append([["user", f"The {big} would not fit in the {box} because the {box} was too {sadj}. What was too {sadj}?"],
                            ["assistant", f"The {box} was too {sadj}."]])
            else:
                out.append([["user", f"The {big} would not fit in the {box} because the {big} was too {badj}. What was too {badj}?"],
                            ["assistant", f"The {big} was too {badj}."]])
    return out


# ---------------------------------------------------------------- I: identity
ID_Q = ["Who are you?", "What is your name?", "What model are you?", "Who made you?", "Who trained you?",
        "Are you ChatGPT?", "Are you made by OpenAI?", "What are you?", "Introduce yourself.", "What can you do?",
        "How big are you?", "Are you open source?", "What are you good at?", "Tell me about yourself."]
ID_A = {
    "name": "I'm BubblSpace MoE-0.9B-A0.2B, a small open language model trained from scratch by BubblSpace.",
    "maker": "I was trained from scratch by BubblSpace. I'm not made by OpenAI, Google, Anthropic or Meta.",
    "size": "I'm a Mixture-of-Experts model with about 0.9 billion parameters in total, of which about 0.2 billion are active for each token.",
    "open": "Yes. My weights and training code are open, under the Apache 2.0 license.",
    "skills": "I'm built mainly for coding and tool use. I can write and explain code, call tools, and answer questions, but I'm a small model, so please double-check important answers.",
}


def gen_identity(rng, n=400):
    out = []
    for _ in range(n):
        q = rng.choice(ID_Q)
        if "name" in q or "Who are you" in q or "What are you" in q or "Introduce" in q or "yourself" in q:
            a = ID_A["name"] + " " + rng.choice([ID_A["skills"], ID_A["size"], ""])
        elif "made" in q or "trained" in q or "OpenAI" in q or "ChatGPT" in q:
            a = ("No. " if q.startswith("Are") else "") + ID_A["maker"]
        elif "big" in q:
            a = ID_A["size"]
        elif "open" in q:
            a = ID_A["open"]
        else:
            a = ID_A["skills"]
        out.append([["user", q], ["assistant", a.strip()]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts_parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--n_facts", type=int, default=150000)
    ap.add_argument("--n_math", type=int, default=120000)
    ap.add_argument("--n_binding", type=int, default=120000)
    ap.add_argument("--only", default=None, help="run 02: build just this set, e.g. own_facts_cmd (others untouched)")
    ap.add_argument("--n_facts_cmd", type=int, default=60000)
    a = ap.parse_args()
    rng = random.Random(20261127)
    banned = HANDTEST_NAMES | load_probe_names(a.repo_root)
    males = [m for m in MALE if m not in banned]
    females = [f for f in FEMALE if f not in banned]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.only == "own_realtime":
        sets = {"own_realtime": gen_realtime(3500, random.Random(20261129))}
    elif a.only == "own_facts_cmd":
        sets = {"own_facts_cmd": gen_facts_cmd(a.facts_parquet, a.n_facts_cmd, random.Random(20261128), skip=a.n_facts)}
    else:
        sets = {"own_facts": gen_facts(a.facts_parquet, a.n_facts, rng),
                "own_math": gen_math(a.n_math, rng, males + females),
                "own_binding": gen_binding(a.n_binding, rng, males, females),
                "own_identity": gen_identity(rng)}
    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for turns in rows:
                f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows):,} rows | e.g. {json.dumps(rows[len(rows) // 2], ensure_ascii=False)[:300]}")
    print(f"names used: {len(males)} male, {len(females)} female (excluded {len(banned)} hand-test/probe names)")


if __name__ == "__main__":
    main()
