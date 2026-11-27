"""Phase 3 SFT data: tool-call coverage per source, in the categories BFCL scores.

Reads the same episodes scripts/phase3_build_sft.py renders (no tokenizing) and counts, per source:
  tools_offered   episode has a tool list (<tools> / function schema in a system or user turn)
  multi_offer     ≥2 tools offered                         (BFCL "multiple": pick the right one)
  single_call     exactly one call in the episode
  parallel        ≥2 calls inside ONE assistant turn       (BFCL "parallel")
  multi_step      calls in ≥2 assistant turns, with tool responses in between
  answer_after    a plain assistant answer after a tool response (uses the result)
  no_call         tools offered but never called           (relevance / "irrelevance")
  valid_json      share of calls whose body parses as JSON with a "name"

    python scripts/phase3_toolcall_stats.py --raw ~/phase3_raw --sources agentic_hermes agentic_glaive agentic_toolmind
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_build_sft import SOURCES  # noqa: E402

CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def n_tools(text):
    block = re.search(r"<tools>(.*?)</tools>", text, re.S)
    body = block.group(1) if block else text
    return len(re.findall(r'"name"\s*:', body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for src in a.sources:
        c = collections.Counter()
        calls_total = calls_json = 0
        for i, turns in enumerate(SOURCES[src](a.raw)):
            if a.limit and i >= a.limit:
                break
            c["episodes"] += 1
            sys_user = " ".join(t for r, t in turns if r in ("system", "user") and "<tool_response>" not in t)
            offered = n_tools(sys_user) if ("<tools>" in sys_user or '"parameters"' in sys_user) else 0
            per_turn = [len(CALL.findall(t)) for r, t in turns if r == "assistant"]
            n_calls = sum(per_turn)
            had_resp, answer_after = False, False
            for r, t in turns:
                if r == "user" and "<tool_response>" in t:
                    had_resp = True
                elif r == "assistant" and had_resp and "<tool_call>" not in t:
                    answer_after = True
            for r, t in turns:
                if r == "assistant":
                    for body in CALL.findall(t):
                        calls_total += 1
                        try:
                            calls_json += "name" in json.loads(body)
                        except (json.JSONDecodeError, TypeError):
                            pass
            c["tools_offered"] += offered > 0
            c["multi_offer"] += offered >= 2
            c["single_call"] += n_calls == 1
            c["parallel"] += any(k >= 2 for k in per_turn)
            c["multi_step"] += sum(k > 0 for k in per_turn) >= 2
            c["answer_after"] += answer_after
            c["no_call"] += offered > 0 and n_calls == 0
        n = max(1, c["episodes"])
        row = {k: f"{c[k]:,} ({100 * c[k] / n:.0f}%)" for k in
               ("tools_offered", "multi_offer", "single_call", "parallel", "multi_step", "answer_after", "no_call")}
        row["valid_json_calls"] = f"{calls_json:,}/{calls_total:,}"
        print(f"\n== {src}: {c['episodes']:,} episodes")
        for k, v in row.items():
            print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
