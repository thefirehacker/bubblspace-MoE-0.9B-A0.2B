"""Shared pieces of the RL pipeline: chat rendering, task loading, and batch scoring (code in parallel sandboxes)."""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.rl import rewards

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def render(messages) -> str:
    """Our SFT wire format: <|im_start|>{role}\\n{text}\\n<|im_end|> per turn, then the assistant header."""
    return "".join(f"<|im_start|>{r}\n{t}\n<|im_end|>" for r, t in messages) + "<|im_start|>assistant\n"


def load_tasks(path, areas=None, limit_per_area=0, seed=0):
    tasks = [json.loads(l) for l in open(path)]
    if areas:
        tasks = [t for t in tasks if t["area"] in areas]
    if limit_per_area:
        rng, by = random.Random(seed), defaultdict(list)
        for t in tasks:
            by[t["area"]].append(t)
        tasks = [t for v in by.values() for t in rng.sample(v, min(limit_per_area, len(v)))]
    return tasks


def score_batch(tasks: list[dict], replies: list[str], finished: list[bool], sandbox_workers: int = 16) -> list[float]:
    """One reward per reply. Non-code kinds are scored inline; code programs run in the docker sandbox in parallel
    (scripts/sandbox_exec.py: no network, 1 CPU, 512 MB, read-only, 10 s)."""
    out: list[float | None] = [rewards.score(t, r, f) for t, r, f in zip(tasks, replies, finished)]
    jobs = []
    for i, (t, r, f) in enumerate(zip(tasks, replies, finished)):
        if t["kind"] == "code":
            prog = rewards.code_program(r, t) if f else None
            if prog is None:
                out[i] = 0.0
            else:
                jobs.append((i, prog))
    if jobs:
        from sandbox_exec import run_in_sandbox
        with ThreadPoolExecutor(sandbox_workers) as ex:
            for (i, _), res in zip(jobs, ex.map(lambda j: run_in_sandbox(j[1]), jobs)):
                out[i] = 1.0 if res.get("passed") else 0.0
    return [float(x) for x in out]
