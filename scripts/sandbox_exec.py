"""HumanEval + MBPP, step 2 of 2: execute model-written programs in isolated containers.

Every program runs in its OWN fresh container and is passed on stdin. Nothing from the host is
mounted. Limits per container:
  --network none                 no network at all
  --cpus 1 --memory 512m         1 CPU, 512 MB RAM, no swap (--memory-swap 512m)
  --pids-limit 64                caps fork bombs
  --read-only + 16 MB noexec /tmp
  --user 65534:65534             nobody, non-root
  --cap-drop ALL, no-new-privileges
  --ulimit fsize=1MB             caps file writes inside /tmp
  10 s wall clock, then docker kill
Pass = exit code 0 (all asserts/check() passed). The image is pinned by digest.

Usage:
    python scripts/sandbox_exec.py --codegen benchmark/phase2/codegen_<tag>.json --out benchmark/phase2/codeexec_<tag>.json
    python scripts/sandbox_exec.py --selftest        # adversarial checks of the sandbox itself
"""

import argparse
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE = "python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535"  # python:3.11-slim
TIMEOUT_S = 10


def run_in_sandbox(program: str, timeout=TIMEOUT_S):
    name = f"sbx-{uuid.uuid4().hex[:12]}"
    cmd = ["sudo", "-n", "docker", "run", "--rm", "-i", "--name", name,
           "--network", "none", "--cpus", "1", "--memory", "512m", "--memory-swap", "512m",
           "--pids-limit", "64", "--read-only", "--tmpfs", "/tmp:rw,size=16m,noexec,nosuid,nodev",
           "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--ulimit", "fsize=1048576", "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", "/tmp",
           IMAGE, "python3", "-I", "-"]
    try:
        p = subprocess.run(cmd, input=program.encode(), capture_output=True, timeout=timeout)
        return {"passed": p.returncode == 0, "status": "ok" if p.returncode == 0 else f"exit_{p.returncode}",
                "stderr_tail": p.stderr.decode(errors="replace")[-300:]}
    except subprocess.TimeoutExpired:
        subprocess.run(["sudo", "-n", "docker", "kill", name], capture_output=True)
        subprocess.run(["sudo", "-n", "docker", "rm", "-f", name], capture_output=True)
        return {"passed": False, "status": "timeout", "stderr_tail": ""}


SELFTEST = {
    # name: (program, must_pass)
    "hello": ("print('ok')", True),
    "assert_fail": ("assert 1 == 2", False),
    "network_blocked": ("import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)", False),
    "dns_blocked": ("import socket; socket.gethostbyname('example.com')", False),
    "root_fs_readonly": ("open('/etc/evil', 'w').write('x')", False),
    "home_readonly": ("open('/evil', 'w').write('x')", False),
    "not_root": ("import os; assert os.getuid() != 0", True),
    "infinite_loop_killed": ("while True: pass", False),
    "memory_capped": ("x = bytearray(2 * 1024**3)", False),
    "fork_bomb_capped": ("import os\nfor _ in range(500):\n    os.fork()\nimport time; time.sleep(30)", False),
    "tmp_exec_blocked": ("import os,stat\nopen('/tmp/x.sh','w').write('#!/bin/sh\\necho hi')\nos.chmod('/tmp/x.sh',0o755)\nimport subprocess; subprocess.run(['/tmp/x.sh'], check=True)", False),
    "big_file_capped": ("open('/tmp/big','wb').write(b'0' * 50_000_000)", False),
}


def selftest():
    ok = True
    for name, (prog, must_pass) in SELFTEST.items():
        r = run_in_sandbox(prog)
        good = r["passed"] == must_pass
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<22} -> {r['status']}  {r['stderr_tail'][-90:].strip()!r}")
    print("SANDBOX SELFTEST:", "ALL GOOD" if ok else "PROBLEMS FOUND")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codegen")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    data = json.loads(Path(args.codegen).read_text())
    summary, per_task = {}, {}
    for task, rows in data["tasks"].items():
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda r: run_in_sandbox(r["program"]), rows))
        for r, x in zip(rows, res):
            x["task_id"] = r["task_id"]
        n_pass = sum(x["passed"] for x in res)
        statuses = {}
        for x in res:
            statuses[x["status"]] = statuses.get(x["status"], 0) + 1
        summary[task] = {"n": len(res), "passed": n_pass, "pass_at_1": n_pass / max(1, len(res)), "statuses": statuses}
        per_task[task] = res
        print(f"[{task}] pass@1 = {n_pass}/{len(res)} = {100 * n_pass / max(1, len(res)):.1f}%   {statuses}")
    out = {"checkpoint": data["checkpoint"], "step": data["step"], "image": IMAGE, "timeout_s": TIMEOUT_S,
           "summary": summary, "results": per_task}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[sandbox] saved {args.out}")


if __name__ == "__main__":
    main()
