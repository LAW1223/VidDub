#!/usr/bin/env bash
set -euo pipefail

python scripts/infer.py \
  --video "${1:?source video}" \
  --audio "${2:?target wav}" \
  --model "${3:-AIPeanutman/VidDub}" \
  --weight-root "${VIDDUB_WEIGHT_ROOT:-weights}" \
  --output "${4:-outputs/dubbed.mp4}" \
  --mode auto \
  --overlap 17
