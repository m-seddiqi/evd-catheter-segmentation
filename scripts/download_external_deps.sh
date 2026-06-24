#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/third_party"

if [ ! -d "${ROOT}/third_party/fvit" ]; then
  git clone https://github.com/apple/ml-fastvit.git "${ROOT}/third_party/fvit"
fi

python - <<'PY' "${ROOT}/third_party/fvit/models/fastvit.py"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = {
    "from models.modules.mobileone import MobileOneBlock": (
        "from third_party.fvit.models.modules.mobileone import MobileOneBlock"
    ),
    "from models.modules.replknet import ReparamLargeKernelConv": (
        "from third_party.fvit.models.modules.replknet import ReparamLargeKernelConv"
    ),
    "from mmcv.runner import _load_checkpoint": (
        "from mmcv.runner import load_checkpoint as _load_checkpoint"
    ),
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text)
PY

mkdir -p "${ROOT}/third_party/fvit/unfused_checkpoints"
for variant in fastvit_t8 fastvit_sa12 fastvit_sa36; do
  if [ ! -f "${ROOT}/third_party/fvit/unfused_checkpoints/${variant}.pth.tar" ]; then
    curl -L \
      -o "${ROOT}/third_party/fvit/unfused_checkpoints/${variant}.pth.tar" \
      "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/${variant}.pth.tar"
  fi
done

cat <<'MSG'

FastViT source and initialization checkpoints are ready under third_party/fvit.

MSG
