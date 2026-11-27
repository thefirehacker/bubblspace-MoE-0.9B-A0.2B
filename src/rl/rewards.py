"""Phase 4 RL rewards: plain-code checkers, one per task type (doc/rl-primer-phase4.md §5). No LLM judges.

Every checker takes the model's reply text (and the task's answer key) and returns a float in [0, 1].
`score(task, reply, finished)` dispatches on task["kind"]; a reply cut off at the length limit (finished=False) gets 0
for every kind, so the policy is never paid for rambling past the budget.

Kinds and answer keys (task dicts are built by scripts/rl_build_tasks.py):
  number     {"answer": "19"}                  GSM8K / MATH with a numeric answer: last \\boxed{} / "answer is" / last number
  time       {"answer": "1:30"}                our time word problems: the reply's last clock time (H:MM, "1.30", "half past one")
  name       {"answer": "Anita", "names": [...]}  binding: the first of the candidate names the reply commits to
  ifeval     {"constraint": {...}}             instruction constraints (Tülu RLVR format: func_name + kwargs)
  code       {"tests": "...", "setup": "..."}   reward computed by the caller in the sandbox (see code_program)
  tool       {"calls": [...]} or {"calls": []}  gold tool calls (name + arguments); [] means "answer without a call"
  refusal    {"expect": "answer"|"refuse"|"no_live", "answer": "..."}
"""
from __future__ import annotations

import json
import re

# ------------------------------------------------------------------ shared helpers
_THINK = re.compile(r"<think>.*?</think>", re.S)
REFUSAL = re.compile(r"^\s*(I'?m sorry|I am sorry|I cannot|I can'?t|I apologi[sz]e|As an AI|Sorry,|Unfortunately, I|I'?m not able|"
                     r"I'?m unable|I won'?t|I will not)", re.I)
NO_LIVE = re.compile(r"(don'?t|do not|cannot|can'?t) (have )?(access|look up|check|browse|provide real-time)|real-time|"
                     r"live (data|information|access)|up-to-date (data|information)|current (data|information)", re.I)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _num(s: str):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


# ------------------------------------------------------------------ number (GSM8K / MATH)
def extract_number(reply: str):
    t = strip_think(reply).replace(",", "")
    t = re.sub(r"```output.*?```", "", t, flags=re.S)  # never trust a model-written "output" block
    for pat in (r"\\boxed\{\s*\$?(-?[\d.]+)", r"[Tt]he (?:final )?answer is[:\s]*\$?(-?[\d.]+)", r"####\s*\$?(-?[\d.]+)"):
        m = re.findall(pat, t)
        if m:
            return _num(m[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return _num(nums[-1]) if nums else None


def reward_number(reply: str, answer: str) -> float:
    pred, gold = extract_number(reply), _num(str(answer))
    return float(pred is not None and gold is not None and abs(pred - gold) < 1e-6 * max(1.0, abs(gold)))


# ------------------------------------------------------------------ time
_WORD_H = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def extract_times(reply: str) -> list[tuple[int, int]]:
    """All clock times in order: 1:30, 1.30, 1:30 pm, 7 o'clock, half past one, quarter to three."""
    t = strip_think(reply).lower()
    found = []
    for m in re.finditer(r"\b(\d{1,2})[:.](\d{2})\b|\b(\d{1,2}) ?o'?clock\b|\b(half|quarter) (past|to) (\d{1,2}|[a-z]+)\b", t):
        if m.group(1):
            h, mi = int(m.group(1)), int(m.group(2))
        elif m.group(3):
            h, mi = int(m.group(3)), 0
        else:
            hw = m.group(6)
            h = int(hw) if hw.isdigit() else _WORD_H.get(hw, -1)
            if h < 0:
                continue
            mins = 30 if m.group(4) == "half" else 15
            h, mi = (h, mins) if m.group(5) == "past" else ((h - 1) % 12 or 12, 60 - mins)
        if 0 <= mi < 60 and 0 <= h <= 24:
            found.append(((h - 1) % 12 + 1 if h else 12, mi))
    return found


def reward_time(reply: str, answer: str) -> float:
    h, mi = (int(x) for x in answer.split(":"))
    times = extract_times(reply)
    return float(bool(times) and times[-1] == ((h - 1) % 12 + 1, mi))


# ------------------------------------------------------------------ name (binding)
def reward_name(reply: str, answer: str, names: list[str]) -> float:
    """The first candidate name the reply mentions must be the answer (an answer that names both but leads with the
    wrong one is wrong)."""
    t = strip_think(reply)
    pos = {n: m.start() for n in names if (m := re.search(rf"\b{re.escape(n)}\b", t))}
    return float(bool(pos) and min(pos, key=pos.get) == answer)


# ------------------------------------------------------------------ instruction constraints (Tülu RLVR ground truth)
def _count_cmp(n: int, target: int, quantifier: str | None) -> bool:
    q = (quantifier or "").lower()
    if q in ("at least",):
        return n >= target
    if q in ("at most",):
        return n <= target
    if q in ("less than",):
        return n < target
    if q in ("more than",):
        return n > target
    if q in ("around", "about", "approximately"):
        return abs(n - target) <= max(1, round(0.1 * target))
    return n == target


def _words(t):
    return re.findall(r"\b\w+\b", t)


def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


IF_CHECKS = {
    "verify_keywords": lambda t, c: all(k.lower() in t.lower() for k in (c.get("keyword_list") or [])),
    "verify_keyword_frequency": lambda t, c: len(re.findall(rf"\b{re.escape(c['word'])}\b", t, re.I)) == c["N"],
    "validate_forbidden_words": lambda t, c: not any(re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in (c.get("forbidden_words") or [])),
    "verify_letter_frequency": lambda t, c: t.lower().count(str(c["letter"]).lower()) == c["N"],
    "verify_paragraph_count": lambda t, c: len([p for p in t.split("* * *") if p.strip()]) == c["N"],
    "validate_word_constraint": lambda t, c: _count_cmp(len(_words(t)), c["N"], c.get("quantifier")),
    "verify_sentence_constraint": lambda t, c: _count_cmp(len(_sentences(t)), c["N"], c.get("quantifier")),
    "validate_paragraphs": lambda t, c: (lambda ps: len(ps) == c["N"] and len(ps) >= c["i"]
                                         and ps[c["i"] - 1].split()[0].strip("*\"'").lower() == str(c["first_word"]).lower())(
                                             [p for p in re.split(r"\n\s*\n", t) if p.strip()]),
    "verify_postscript": lambda t, c: bool(re.search(rf"^\s*{re.escape(c['postscript_marker'])}", t, re.M)),
    "validate_placeholders": lambda t, c: len(re.findall(r"\[[^\[\]]+\]", t)) >= c["N"],
    "verify_bullet_points": lambda t, c: len(re.findall(r"^\s*[*-]\s+\S", t, re.M)) == c["N"],
    "validate_title": lambda t, c: bool(re.search(r"<<[^<>\n]+>>", t)),
    "validate_choice": lambda t, c: t.strip().strip(".\"'").lower() in [o.strip().lower() for o in _options(c.get("options"))],
    "validate_highlighted_sections": lambda t, c: len(re.findall(r"\*[^*\n]+\*", t)) >= c["N"],
    "validate_sections": lambda t, c: len(re.findall(rf"^\s*{re.escape(c.get('section_splitter') or 'SECTION')}\s*\d+", t, re.M)) == c["N"],
    "validate_json_format": lambda t, c: _is_json(t),
    "validate_repeat_prompt": lambda t, c: t.strip().startswith(str(c.get("original_prompt") or "").strip()),
    "validate_two_responses": lambda t, c: (lambda parts: len(parts) == 2 and all(p.strip() for p in parts)
                                            and parts[0].strip() != parts[1].strip())(t.split("******")),
    "validate_uppercase": lambda t, c: t == t.upper() and any(ch.isalpha() for ch in t),
    "validate_lowercase": lambda t, c: t == t.lower() and any(ch.isalpha() for ch in t),
    "validate_frequency_capital_words": lambda t, c: _count_cmp(len(re.findall(r"\b[A-Z]{2,}\b", t)), c["N"], c.get("quantifier")),
    "validate_end": lambda t, c: t.strip().endswith(str(c["end_phrase"]).strip()),
    "validate_quotation": lambda t, c: t.strip().startswith('"') and t.strip().endswith('"'),
    "validate_no_commas": lambda t, c: "," not in t,
}


def _options(o):
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [x for x in re.split(r"\(|\)|,|\bor\b|/", o.replace("My answer is", "")) if x.strip()] or [o]
    return []


def _is_json(t):
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def reward_ifeval(reply: str, constraint: dict) -> float:
    fn = IF_CHECKS.get(constraint.get("func_name"))
    if fn is None:
        raise KeyError(f"no checker for {constraint.get('func_name')}")
    try:
        return float(bool(fn(strip_think(reply), constraint)))
    except (KeyError, TypeError, IndexError, AttributeError):
        return 0.0


# ------------------------------------------------------------------ code (program text; execution happens in the sandbox)
def extract_code(reply: str) -> str:
    t = strip_think(reply)
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", t, re.S)
    if not blocks:
        m = re.search(r"```(?:python|py)?[^\n]*\n(.*)", t, re.S)
        return m.group(1) if m else ""
    return next((b for b in blocks if "def " in b), blocks[0])


def code_program(reply: str, task: dict) -> str | None:
    """The program to run in the sandbox (scripts/sandbox_exec.py run_in_sandbox); None if there is no code."""
    code = extract_code(reply)
    if "def " not in code:
        return None
    setup = task.get("setup") or ""
    return (setup + "\n" if setup else "") + code + "\n\n" + task["tests"] + "\n"


# ------------------------------------------------------------------ tool calls
def parse_calls(text: str) -> list[dict] | None:
    """All <tool_call> blocks as {"name", "arguments"}; None if any block is not valid JSON."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            return None
        args = obj.get("arguments", {}) if isinstance(obj, dict) else {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        calls.append({"name": obj.get("name") if isinstance(obj, dict) else None, "arguments": args if isinstance(args, dict) else {}})
    if "<tool_call>" in text and not calls:
        return None
    return calls


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def reward_tool(reply: str, gold_calls: list[dict]) -> float:
    """1.0: exactly the gold calls (any order; required argument values equal; no extra arguments beyond the gold ones
    and the optional ones listed in "optional"). 0.2: valid call JSON with the right tool names but wrong arguments.
    No-call tasks (gold_calls == []): 1.0 for answering without any <tool_call>, else 0."""
    calls = parse_calls(reply)
    if not gold_calls:
        return float("<tool_call>" not in reply and bool(strip_think(reply)))
    if not calls:
        return 0.0
    if sorted(c["name"] or "" for c in calls) != sorted(g["name"] for g in gold_calls):
        return 0.0
    remaining = list(gold_calls)
    for c in calls:
        hit = None
        for i, g in enumerate(remaining):
            allowed = set(g["arguments"]) | set(g.get("optional", []))
            if c["name"] == g["name"] and not (set(c["arguments"]) - allowed) and all(
                    k in c["arguments"] and _norm(c["arguments"][k]) == _norm(v) for k, v in g["arguments"].items()):
                hit = i
                break
        if hit is None:
            return 0.2
        remaining.pop(hit)
    return 1.0


# ------------------------------------------------------------------ refusal balance
def reward_refusal(reply: str, expect: str, answer: str | None = None) -> float:
    t = strip_think(reply)[:600]
    refused = bool(REFUSAL.search(t))
    if expect == "refuse":
        return float(refused)
    if expect == "no_live":
        return float(bool(NO_LIVE.search(t)))
    # "answer": must not refuse; if a gold answer is given it must appear
    if refused:
        return 0.0
    return float(answer is None or answer.lower() in t.lower())


# ------------------------------------------------------------------ dispatch
def score(task: dict, reply: str, finished: bool) -> float:
    """Reward in [0, 1] for one reply. Code tasks are scored by the caller (sandbox); this returns None for them."""
    if not finished:
        return 0.0
    k = task["kind"]
    if k == "number":
        return reward_number(reply, task["answer"])
    if k == "time":
        return reward_time(reply, task["answer"])
    if k == "name":
        return reward_name(reply, task["answer"], task["names"])
    if k == "ifeval":
        return reward_ifeval(reply, task["constraint"])
    if k == "tool":
        return reward_tool(reply, task["calls"])
    if k == "refusal":
        return reward_refusal(reply, task["expect"], task.get("answer"))
    if k == "code":
        return None
    raise KeyError(f"unknown task kind {k}")
