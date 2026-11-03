# Evaluation scope

This repository ships the VidDub inference path, preprocessing contracts,
release-integrity checks, and implementation-parity tests. It does not bundle
third-party benchmark media or claim that a fresh clone alone reproduces every
paper table.

## Included checks

Run from the repository root:

```bash
python -m pytest -q
python scripts/audit_release.py --root .
python scripts/audit_release.py --root /path/to/exported/VidDub
```

These checks validate:

- the selected `FULL2_DMD_R4@100` lineage;
- full-context, two-step, single-audio-branch inference settings;
- the 81-frame and 17-frame-overlap release protocol;
- source/model package structure and required metadata;
- command-line entry points and component checksums.

`docs/RELEASE_VALIDATION.md` records the paired implementation-equivalence
tests used for the initial release.

## External evaluation inputs

Paper benchmarks use frozen inputs and pairings that are separate from this
source repository:

- HDTF-30
- VoxCeleb2-30
- TalkVid-Bench-500
- [VidDub-Bench-256](https://huggingface.co/datasets/AIPeanutman/VidDub-Bench)
- Long-20

Do not infer benchmark membership or cross-audio pairings from filesystem row
order. Use the official manifest for each released artifact. Third-party media
remains subject to its original terms and is not redistributed here.

VidDub-Bench contains synthetic media and is released separately under
CC BY-NC 4.0. Download, integrity validation, and batched same/cross inference
are documented in [BENCHMARK.md](BENCHMARK.md). The other paper benchmarks
retain their original access and redistribution terms.

## Protocol notes

- Model-ready calls use 81 frames at 24 FPS and 512x512 crops.
- Public full-video inference preserves the source video's native output timing.
- The paper's Long-20 inputs are 25 FPS and use 81-frame windows with 64-frame
  stride and 17 shared frames.
- Method-specific coverage and all-method common intersections are distinct
  reporting protocols.
- SyncNet confidence is a synchronization proxy, not proof of phoneme
  correctness or human preference.

For published results and visual comparisons, see the
[paper](https://arxiv.org/abs/2611.03404) and
[project page](https://law1223.github.io/viddub-project-page/).
