"""Push a runnable "basic model" branch to the HF model repo (TheFireHacker/bubblspace-moe0.9b-a0.2b), as done for
phase3-step1500-tokens40B: weights + model code + tokenizer (with the chat template) + generate.py + a model card.
Full training states (optimizer etc.) go to the training-states repo instead (scripts/hf_push_training_state.py).

The branch holds exactly these files (delete_patterns removes everything else inherited from main):
    model.safetensors     from an export dir of scripts/restore_checkpoint.py (hard-linked, not copied)
    training_config.yaml  the config the checkpoint was trained with
    src/                  src/__init__.py, src/config.py, src/model/*.py
    tokenizer/            bubblspace-tokenizer + chat template
    generate.py           scripts/hf_generate.py (chat mode, tools, plain text)
    requirements.txt, README.md

    python scripts/hf_push_inference_branch.py --export_dir ~/hf_export/phase3-run02b-step1500-tokens40B \
        --branch phase3-run02b-step1500-tokens40B --config configs/phase3_run02b.yaml \
        --title "Phase 3 SFT run 02b, step 1,500" --summary "..." --wandb_run 6dz3e9d5
"""
import argparse
import json
import os
import shutil
from pathlib import Path

REPO = "TheFireHacker/bubblspace-moe0.9b-a0.2b"
STATES = "TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states"
CHAT_TEMPLATE = ('{% for message in messages %}<|im_start|>{{ message["role"] }}\n{{ message["content"] }}\n<|im_end|>{% endfor %}'
                 '{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}')
REQS = "torch>=2.6.0\ntransformers>=4.45.0\nsafetensors>=0.4.0\npyyaml>=6.0\n"
CARD = """---
license: apache-2.0
tags:
  - mixture-of-experts
  - moe
  - from-scratch
{tags}
---

# BubblSpace MoE-0.9B-A0.2B: {title}

> **This branch:** `{branch}`. {summary}

| | |
|---|---|
| Parameters | ≈915M total / ≈211M active per token (64 experts, top-8) |
| Context | 16,384 tokens (RoPE base 1e6) |
| Files | `model.safetensors`, `training_config.yaml`, `src/` (model code), `tokenizer/` (with chat template), `generate.py` |
| Full training state (optimizer, schedulers, RNG) | branch [`{branch}`](https://huggingface.co/{states}/tree/{branch}) of the training-states repo |
| W&B | run [{wandb}](https://forge.coreweave.com/wandb/firehacker/moe0-9b-a0-2b/runs/{wandb}) |

## Try it

```bash
hf download {repo} --revision {branch} --local-dir {branch}
cd {branch} && pip install -r requirements.txt
python generate.py --chat --prompt "What is the capital of India?"
python generate.py --chat --tools tools.json --prompt "What's the weather in Paris?"   # tools.json: a list of tool definitions
```

The tokenizer includes the chat template, so `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` gives the exact training format.

## Caveats

A small model built for learning: it answers fluently and can be wrong. Don't rely on its outputs. For learning and research; see the disclaimer on the `main` model card.

Code, data pipeline and every training script: [github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B](https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True, help="has model.safetensors (scripts/restore_checkpoint.py export)")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--wandb_run", required=True)
    ap.add_argument("--tags", default="sft,chat,tool-calling")
    ap.add_argument("--stage_dir", default=str(Path.home() / "hf_infer"))
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    d = Path(a.stage_dir) / a.branch
    shutil.rmtree(d, ignore_errors=True)
    (d / "src" / "model").mkdir(parents=True)
    os.link(Path(a.export_dir) / "model.safetensors", d / "model.safetensors")
    shutil.copy(a.config, d / "training_config.yaml")
    for f in ("__init__.py", "config.py"):
        shutil.copy(root / "src" / f, d / "src" / f)
    for f in (root / "src" / "model").glob("*.py"):
        shutil.copy(f, d / "src" / "model" / f.name)
    shutil.copytree(root / "bubblspace-tokenizer", d / "tokenizer", ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    tc = json.loads((d / "tokenizer" / "tokenizer_config.json").read_text())
    tc["chat_template"] = CHAT_TEMPLATE
    (d / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False))
    shutil.copy(root / "scripts" / "hf_generate.py", d / "generate.py")
    (d / "requirements.txt").write_text(REQS)
    tags = "\n".join(f"  - {t.strip()}" for t in a.tags.split(",") if t.strip())
    (d / "README.md").write_text(CARD.format(title=a.title, branch=a.branch, summary=a.summary, states=STATES,
                                             wandb=a.wandb_run, repo=REPO, tags=tags))
    from huggingface_hub import HfApi
    api = HfApi()
    if a.branch not in [b.name for b in api.list_repo_refs(REPO).branches]:
        api.create_branch(REPO, branch=a.branch, revision="main")
    api.upload_folder(repo_id=REPO, folder_path=str(d), revision=a.branch, delete_patterns=["*"],
                      ignore_patterns=["**/__pycache__/**", "*.pyc"], commit_message=f"{a.title} ({a.branch})")
    files = sorted(api.list_repo_files(REPO, revision=a.branch))
    print("uploaded", a.branch, files)


if __name__ == "__main__":
    main()
