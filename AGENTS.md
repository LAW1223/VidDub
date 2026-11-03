# VidDub repository guide

This file is the primary orientation document for AI coding assistants working
in this repository.

## Project identity

- Paper: **VidDub: Replacement-Aware Distillation for Two-Step Video Dubbing**
- Paper URL: <https://arxiv.org/abs/2611.03404>
- Code: <https://github.com/LAW1223/VidDub>
- Project page: <https://law1223.github.io/viddub-project-page/>
- Model weights: <https://huggingface.co/AIPeanutman/VidDub>
- Benchmark: <https://huggingface.co/datasets/AIPeanutman/VidDub-Bench>
- Selected release model: `FULL2_DMD_R4@100`

VidDub is an offline, full-context video-dubbing editor. The released student
uses two denoising steps. It is not the causal diagnostic model and must not be
described as a persistent streaming system.

## Repository map

- `viddub/foundation/`: reduced Wan2.1-VACE backend required by VidDub.
- `viddub/full_context/`: selected teacher and full-context inference path.
- `viddub/causal/`: shared few-step objectives and optional causal ablations.
- `scripts/infer.py`: public inference entry point.
- `scripts/run_benchmark.py`: reusable-model VidDub-Bench inference runner.
- `scripts/validate_benchmark.py`: benchmark structure and integrity validator.
- `scripts/validate_training_dataset.py`: frozen training-data manifest/media validator.
- `scripts/stage_h3_dataset.py`: owner-authorized training-data package builder.
- `scripts/audit_release.py`: source/model release audit.
- `scripts/export_hf_models.py`: deterministic Hugging Face package exporter.
- `data/`: preprocessing pipeline and public data contract.
- `configs/full_context/`: selected ODE, fake-score, and DMD recipes.
- `configs/causal/`: causal ablation recipes; not the default release path.
- `docs/TRAINING.md`: training and distillation commands.
- `docs/ARCHITECTURE.md`: model architecture summary.
- `docs/BENCHMARK.md`: frozen benchmark download, schema, and execution guide.
- `docs/RELEASE_VALIDATION.md`: implementation-equivalence evidence.
- `tests/test_release_contract.py`: release invariants and CLI checks.

## Non-negotiable release contract

Do not change these values silently:

- `causal = false`
- `student_steps = 2`
- `inference_audio_cfg = 1`
- `num_frames = 81`
- training/model-ready clips: 24 FPS; public full-video inference preserves
  the source video's native timing
- long-video window stride / overlap: 64 / 17 frames
- final lineage:
  `sft-r2a-ema200 -> FULL2_ODE_R3@400 -> FULL2_DMD_R4@100`

The trained release components are DiT, VACE, and the audio adapter. Wan
VAE/T5/tokenizer and Whisper-small are external dependencies and are not
vendored in the model repository.

## Common commands

Install:

```bash
pip install -r requirements.txt
```

Inference:

```bash
python scripts/infer.py \
  --model AIPeanutman/VidDub \
  --weight-root ./weights \
  --video input.mp4 \
  --audio replacement.wav \
  --output output.mp4
```

Benchmark validation and inference:

```bash
python scripts/validate_benchmark.py --root data/VidDub-Bench
python scripts/run_benchmark.py \
  --benchmark data/VidDub-Bench \
  --model AIPeanutman/VidDub \
  --output outputs/VidDub-Bench
```

Validation before any public push:

```bash
python -m compileall -q viddub scripts data tests
python -m pytest -q
python scripts/audit_release.py --root .
```

## Editing rules

1. Treat `README.md`, `docs/TRAINING.md`, model config, and model card as one
   public contract. Update all affected surfaces together.
2. Never commit model weights, generated videos, private datasets, cluster
   paths, access tokens, experiment logs, or cached bytecode.
3. Keep source/model/data paths artifact-relative. Use the documented
   environment variables for external assets.
4. Preserve third-party notices and licenses under `dependencies/` and
   `viddub/foundation/`.
5. Keep causal experiments explicitly labeled as ablations. Public inference
   defaults to the full-context model.
6. Do not claim that automated synchronization metrics prove phoneme
   correctness or human preference.
7. Do not publish dataset media until its rights, consent, provenance, and
   redistribution review is complete.
8. Preserve VidDub-Bench item order, keys, fixed `cross_key` assignments,
   per-item seeds, rights records, and checksums. The benchmark is CC BY-NC 4.0;
   do not apply the source/model license to its media.
9. Preserve the 45,221-source / 85,435-window training-data selection. Its
   canonical Hub card is `data/TRAINING_DATA_CARD.md`; the media license is
   CC BY-NC 4.0 and is separate from the source/model license.

## Change checklist

For code changes:

- run the full test suite and release audit;
- verify fresh-clone paths rather than relying on workstation state;
- update config or checksum contracts when serialized formats change;
- document user-facing CLI changes in `README.md`.

For model-card changes:

- update `scripts/export_hf_models.py` first;
- regenerate the card rather than maintaining an unrelated manual variant;
- verify the arXiv, project-page, code, license, and weight links.

For documentation changes:

- use public names and URLs only;
- remove internal handoff notes before release;
- keep the paper title and citation synchronized with arXiv.
