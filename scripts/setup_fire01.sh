#!/usr/bin/env bash
# Run on fire01 after SSH'ing in. Sets up the Python env, installs deps,
# probes torch._grouped_mm availability, and verifies the storage mount.
#
# The persistent filesystem is already mounted at MOUNT_POINT (Lambda only
# allows attaching a same-region filesystem at instance launch, so this
# being mounted already confirms the region match — see README.md "Lambda
# Cloud setup" for how this was verified).
set -euo pipefail

MOUNT_POINT="/lambda/nfs/h100-f1"

echo "== GPU check =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== Python env =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

echo "== torch._grouped_mm probe =="
python3 -c "
import torch
from src.model.moe import probe_grouped_mm
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('grouped_mm usable:', probe_grouped_mm())
"

echo "== Storage mount =="
if ! mountpoint -q "$MOUNT_POINT"; then
    echo "$MOUNT_POINT is not mounted. Check the Lambda console (Storage tab) for this instance." >&2
    exit 1
fi
df -h "$MOUNT_POINT"
touch "$MOUNT_POINT/._writetest" && rm "$MOUNT_POINT/._writetest"
echo "Write access to $MOUNT_POINT confirmed."

echo "== wandb =="
echo "Run 'wandb login' (or set WANDB_API_KEY in .env) before training."
