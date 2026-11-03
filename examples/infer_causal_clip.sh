#!/usr/bin/env bash
set -euo pipefail

# Research ablation only; the released VidDub model uses examples/infer_bidirectional.sh.
python -m viddub.causal.infer \
  --config configs/causal/dmd2.json \
  --checkpoint-dir "${1:?checkpoint directory}" \
  --step "${2:-100}" \
  --clip-id "${3:?prepared validation clip id}" \
  --audio "${4:?target wav}" \
  --output "${5:-outputs/causal_clip.mp4}"
