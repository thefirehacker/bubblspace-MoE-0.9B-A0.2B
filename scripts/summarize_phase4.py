"""Summary table of the Phase 3 / Phase 4 chat benchmarks (scripts/phase4_benchmarks_a10.sh output dir)."""
import argparse
import json
from pathlib import Path

NAMES = {"phase3-step1500": "SFT run 01", "phase3-run02b-step1500-tokens40B": "SFT run 02b",
         "phase4-rlv1-step100-tokens40B": "RL v1 step 100", "phase4-rlv1-step300-tokens40B": "RL v1 step 300",
         "phase4-rlv1-step500-tokens40B": "RL v1 step 500"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []
    for m, name in NAMES.items():
        c = load(d / f"chat_{m}.json")
        if not c:
            continue
        x = load(d / f"codeexec_{m}.json") or {}
        r = c.get("refusal", {})
        rr = r.get("rows", [])
        cnt = lambda kind, key: (sum(z[key] for z in rr if z["kind"] == kind), sum(z["kind"] == kind for z in rr))
        passk = lambda task: (x.get("summary", {}).get(task) or {}).get("pass_at_1")
        b = c.get("bfcl", {})
        ife = c.get("ifeval", {}).get("scores", {})
        rows.append([name,
                     "%d/%d" % cnt("harmless", "refused"), "%d/%d" % cnt("harmful", "refused"), "%d/%d" % cnt("realtime", "no_live_data"),
                     passk("humaneval"), passk("mbpp"), c.get("gsm8k", {}).get("accuracy"),
                     ife.get("prompt_level_strict_acc"), ife.get("inst_level_strict_acc"),
                     *[b.get(k, {}).get("accuracy") for k in ("simple", "multiple", "parallel", "irrelevance")]])
    fmt = lambda v: "—" if v is None else (v if isinstance(v, str) else f"{100 * v:.1f}")
    head = ["Model", "Harmless refused", "Harmful refused", "Live data honest", "HumanEval %", "MBPP %", "GSM8K %",
            "IFEval prompt %", "IFEval instr. %", "BFCL simple %", "BFCL multiple %", "BFCL parallel %", "BFCL no-tool %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    # English 6-task: acc_norm for HellaSwag / PIQA / ARC, acc for WinoGrande / LAMBADA (same as every earlier phase)
    e6 = [("HellaSwag", "hellaswag", "acc_norm,none"), ("PIQA", "piqa", "acc_norm,none"),
          ("ARC-e", "arc_easy", "acc_norm,none"), ("ARC-c", "arc_challenge", "acc_norm,none"),
          ("WinoGrande", "winogrande", "acc,none"), ("LAMBADA", "lambada_openai", "acc,none")]
    eng = []
    for m in ("phase3-run02b-step1500-tokens40B", "phase4-rlv1-step500-tokens40B"):
        e = load(d / f"english6_{m}.json")
        if e:
            v = [100 * e["results"][t][k] for _, t, k in e6]
            eng.append(f"| {NAMES[m]} | " + " | ".join(f"{x:.1f}" for x in v) + f" | **{sum(v) / len(v):.2f}** |")
    eng_table = ("| Model | " + " | ".join(n for n, _, _ in e6) + " | Average |\n|" + "---|" * (len(e6) + 2) + "\n"
                 + "\n".join(eng)) if eng else "(not run)"
    out = ("# Phase 3 SFT vs Phase 4 RL v1: chat benchmarks (vLLM backend, greedy, A10, 28 Nov)\n\n" + "\n".join(lines)
           + "\n\nEnglish 6-task (forgetting check):\n\n" + eng_table + "\n")
    Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
