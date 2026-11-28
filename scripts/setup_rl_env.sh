#!/bin/bash
# Set up the RL / vLLM environment on a GPU host (done on the A10 on 27 Nov; same steps for fire01):
#   ~/vllm_env      Python 3.12 venv (uv) with vLLM 0.33.0 (Apache-2.0, released 2 Nov 2026), ninja, and our plugin
#                   (vllm_plugin/, editable). The training venv (.venv) is not touched.
#   docker image    the pinned python:3.11-slim used by scripts/sandbox_exec.py for code rewards
# Usage: bash scripts/setup_rl_env.sh      (from the repo root)
set -e
cd "$(dirname "$0")/.."
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || python3 -m pip install --user -q uv
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -x ~/vllm_env/bin/python ] || $UV venv ~/vllm_env --python 3.12
VIRTUAL_ENV=~/vllm_env $UV pip install -q vllm==0.33.0 ninja pyyaml
VIRTUAL_ENV=~/vllm_env $UV pip install -q -e vllm_plugin
~/vllm_env/bin/python -c "import vllm, importlib.metadata as m; print('vllm', vllm.__version__, [e.name for e in m.entry_points(group='vllm.general_plugins')])"
IMAGE=$(grep -oP 'IMAGE = "\K[^"]+' scripts/sandbox_exec.py)
sudo -n docker image inspect "$IMAGE" > /dev/null 2>&1 || sudo -n docker pull "$IMAGE"
~/vllm_env/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from sandbox_exec import run_in_sandbox; print('sandbox:', run_in_sandbox('print(1)'))"
echo SETUP_RL_ENV_OK
