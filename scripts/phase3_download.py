"""Phase 3 SFT data: download the raw datasets (parquet only, no loading scripts) for doc/plan-04-phase-3-sft.md §3.1.

Runs on the A10 (data prep never runs on the training box). Large repos are sliced to a few files first. The length
scan decides later how much of each is worth taking. After this, ClamAV scans the folder (project rule for downloads).

    python scripts/phase3_download.py --out ~/phase3_raw
"""
import argparse
import time

from huggingface_hub import HfApi, snapshot_download


def first_parquets(api, repo, prefix, n):
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    return files[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    api = HfApi()
    jobs = [
        ("allenai/tulu-3-sft-olmo-2-mixture-0225", ["data/*.parquet", "README.md"]),  # 1.3 GB, all of it
        ("openai/gsm8k", ["main/*.parquet", "README.md"]),
        ("PrimeIntellect/SYNTHETIC-1", first_parquets(api, "PrimeIntellect/SYNTHETIC-1", "data/", 8) + ["README.md"]),
        ("nvidia/Nemotron-Terminal-Corpus", first_parquets(api, "nvidia/Nemotron-Terminal-Corpus", "", 4) + ["README.md"]),
        ("open-thoughts/AgentTrove", first_parquets(api, "open-thoughts/AgentTrove", "data/", 4) + ["README.md"]),
    ]
    for repo, patterns in jobs:
        t = time.time()
        local = "%s/%s" % (a.out, repo.replace("/", "__"))
        snapshot_download(repo, repo_type="dataset", local_dir=local, allow_patterns=patterns)
        print("[%s] %s -> %s (%.0f s)" % (time.strftime("%H:%M:%S"), repo, local, time.time() - t), flush=True)
    print("DOWNLOADS DONE", flush=True)


if __name__ == "__main__":
    main()
