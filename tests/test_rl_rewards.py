"""Unit tests for the Phase 4 RL reward checkers (src/rl/rewards.py)."""
import json

from src.rl import rewards as R


def test_number():
    assert R.reward_number("3 * 8 = 24, 24 - 5 = 19. Thus \\boxed{19} apples.", "19") == 1
    assert R.reward_number("The answer is 1,104.", "1104") == 1
    assert R.reward_number("The answer is 20.", "19") == 0
    # a model-written fake "output" block must not count
    assert R.reward_number("```python\nprint(2**10)\n```\n```output\n1024\n```\nso it is 1000", "1024") == 0


def test_time():
    assert R.reward_time("The show started at 1:30.", "1:30") == 1
    assert R.reward_time("2:00 minus 30 minutes is half past one.", "1:30") == 1
    assert R.reward_time("It started at 1.30", "1:30") == 1
    assert R.reward_time("2 + 30 = 32 minutes late.", "1:30") == 0
    assert R.reward_time("It ended at 1 o'clock.", "1:00") == 1
    assert R.reward_time("It ends at 13:15.", "1:15") == 1  # 24-hour form folds onto the 12-hour answer
    assert R.reward_time("quarter to three", "2:45") == 1


def test_name():
    names = ["Anita", "Ravi", "Leo"]
    assert R.reward_name("Anita. Ravi gave Anita the red cap.", "Anita", names) == 1
    assert R.reward_name("Ravi gave the cap, so Anita has it.", "Anita", names) == 0  # leads with the wrong name
    assert R.reward_name("Nobody knows.", "Anita", names) == 0


def test_ifeval():
    assert R.reward_ifeval("* a\n* b\n* c", {"func_name": "verify_bullet_points", "N": 3}) == 1
    assert R.reward_ifeval("* a\n* b", {"func_name": "verify_bullet_points", "N": 3}) == 0
    assert R.reward_ifeval("all lower case here", {"func_name": "validate_lowercase"}) == 1
    assert R.reward_ifeval("Not lower", {"func_name": "validate_lowercase"}) == 0
    assert R.reward_ifeval("no commas here", {"func_name": "validate_no_commas"}) == 1
    assert R.reward_ifeval('{"a": 1}', {"func_name": "validate_json_format"}) == 1
    assert R.reward_ifeval("Main text.\nP.S. bye", {"func_name": "verify_postscript", "postscript_marker": "P.S."}) == 1
    assert R.reward_ifeval("<<My Title>>\nbody", {"func_name": "validate_title"}) == 1
    assert R.reward_ifeval("one two three", {"func_name": "validate_word_constraint", "N": 5, "quantifier": "less than"}) == 1


def test_ifeval_string_fields():
    # the dataset stores numbers as strings and lists as text
    assert R.reward_ifeval("* a\n* b\n* c", {"func_name": "verify_bullet_points", "N": "3"}) == 1
    c = {"func_name": "validate_forbidden_words", "forbidden_words": "['synedria', 'metalined']"}
    assert R.reward_ifeval("a normal reply", c) == 1 and R.reward_ifeval("Synedria are", c) == 0
    k = {"func_name": "verify_keywords", "keyword_list": "['hemigale', 'counterbattery']"}
    assert R.reward_ifeval("hemigale and counterbattery", k) == 1 and R.reward_ifeval("hemigale only", k) == 0
    ch = {"func_name": "validate_choice", "options": "('My answer is yes.', 'My answer is no.')"}
    assert R.reward_ifeval("My answer is yes.", ch) == 1 and R.reward_ifeval("Maybe.", ch) == 0
    s = {"func_name": "validate_sections", "N": "2", "section_splitter": "Section:"}
    assert R.reward_ifeval("Section: 1\nfoo\nSection: 2\nbar", s) == 1


def test_tool():
    gold = [{"name": "get_weather", "arguments": {"city": "Paris"}, "optional": ["unit"]}]
    call = lambda d: f"<think>x</think>\n<tool_call>\n{json.dumps(d)}\n</tool_call>"
    assert R.reward_tool(call({"name": "get_weather", "arguments": {"city": "paris"}}), gold) == 1
    assert R.reward_tool(call({"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}), gold) == 1
    assert R.reward_tool(call({"name": "get_weather", "arguments": {"city": "Rome"}}), gold) == 0.2
    assert R.reward_tool(call({"name": "get_current_time", "arguments": {"timezone": "Europe/Paris"}}), gold) == 0
    assert R.reward_tool("<tool_call>\n{bad json\n</tool_call>", gold) == 0
    assert R.reward_tool("I'm doing well, thanks!", []) == 1          # no call needed and none made
    assert R.reward_tool(call({"name": "get_weather", "arguments": {"city": "x"}}), []) == 0


def test_refusal():
    assert R.reward_refusal("I'm sorry, but I can't help with that.", "refuse") == 1
    assert R.reward_refusal("Step 1: gather...", "refuse") == 0
    assert R.reward_refusal("The capital of Japan is Tokyo.", "answer", "Tokyo") == 1
    assert R.reward_refusal("I'm sorry, I can't provide real-time data.", "answer", "Tokyo") == 0
    assert R.reward_refusal("I don't have access to live data, so I can't tell you.", "no_live") == 1
    assert R.reward_refusal("It is 10:00 PM in London.", "no_live") == 0


def test_score_unfinished_is_zero():
    assert R.score({"kind": "number", "answer": "19"}, "The answer is 19", finished=False) == 0


def test_code_program():
    mbpp = {"kind": "code", "tests": "assert add(2, 3) == 5", "setup": ""}
    assert "assert add(2, 3) == 5" in R.code_program("```python\ndef add(a, b):\n    return a + b\n```", mbpp)
    assert R.code_program("def add(a, b):\n    return a + b\n", mbpp) is not None   # no fences: still code
    assert R.code_program("The answer is 5.", mbpp) is None
    io = {"kind": "code", "tests_io": [["1 2\n", "3\n"]]}
    assert "TESTS" in R.code_program("```python\nprint(sum(map(int, input().split())))\n```", io)
