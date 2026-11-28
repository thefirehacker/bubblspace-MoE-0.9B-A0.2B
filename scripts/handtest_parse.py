"""Turn a non-interactive scripts/chat_repl.py session (prompts piped on stdin) into one line per prompt, full output.

    printf "/greedy\\n/rep 1.0\\n/ngram 0\\n/tokens 40\\n%s\\nexit\\n" "$(cat prompts.txt)" \\
        | python scripts/chat_repl.py --config ... --checkpoint ... 2>&1 | python scripts/handtest_parse.py <label> prompts.txt
Same format as benchmark/handtest/*_20261126.txt ("[label] 'prompt' -> 'output'", line breaks shown as " / "), but
neither the prompt nor the output is cut.
"""
import sys

label, pfile = sys.argv[1], sys.argv[2]
t = sys.stdin.read()
outs = [p[3:] for p in t.split("prompt> ") if p.startswith("=> ")]
prompts = [l for l in open(pfile).read().split("\n") if l.strip()]
if len(outs) != len(prompts):
    print(f"[{label}] WARNING: {len(prompts)} prompts but {len(outs)} outputs", file=sys.stderr)
for q, o in zip(prompts, outs):
    o = " / ".join(x.strip() for x in o.strip().splitlines() if x.strip())
    print("[%s] %r -> %r" % (label, q, o))
