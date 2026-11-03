# VidDub-Bench

[VidDub-Bench](https://huggingface.co/datasets/AIPeanutman/VidDub-Bench) is
the frozen 256-source evaluation set introduced with VidDub. All identities are
synthetic. The benchmark contains 128 English and 128 Chinese clips and is
released separately from the source repository under CC BY-NC 4.0.

## Protocol

- 256 source-disjoint clips
- 81 frames, 24 FPS, 512x512
- one fixed same-audio reconstruction condition per source
- one fixed cross-audio replacement condition per source
- identical source frames, masks, reference frames, seeds, and durations
  between the two conditions
- no overlap with the released VidDub training selection

`manifest/items.jsonl` is authoritative. Do not derive replacement pairs from
file order: use `cross_key` and `audio_cross`.

## Download and validate

Install the repository first, then download the benchmark:

```bash
huggingface-cli download AIPeanutman/VidDub-Bench \
  --repo-type dataset \
  --local-dir data/VidDub-Bench

python scripts/validate_benchmark.py \
  --root data/VidDub-Bench
```

The validator checks all 256 item records, language counts, same/cross
assignments, rights records, safe relative paths, file presence, and SHA-256
receipts. For a slower codec-level check of every media file:

```bash
python scripts/validate_benchmark.py \
  --root data/VidDub-Bench \
  --media
```

`--media` requires FFmpeg/ffprobe.

## Package layout

```text
VidDub-Bench/
  README.md
  LICENSE
  clips/<shard>/<key>/
    video.mp4
    audio.wav
    mask.png
    ref.png
  manifest/
    items.jsonl
    rights.jsonl
    release.json
    checksums.sha256
```

Each item record contains:

- `key`, `language`, and the fixed inference `seed`;
- `video`, `mask`, and `reference`;
- `audio_same` for reconstruction;
- `audio_cross` and `cross_key` for replacement dubbing;
- the frozen clip specification (`fps`, `frames`, and `seconds`).

All paths are relative to the benchmark root.

## Run VidDub

After downloading the base and VidDub weights from the main
[Quick Start](../README.md#quick-start), run all 512 same/cross calls:

```bash
python scripts/run_benchmark.py \
  --benchmark data/VidDub-Bench \
  --model weights/VidDub \
  --weight-root weights \
  --output outputs/VidDub-Bench \
  --condition both
```

The model is loaded once and reused across the run. Outputs are stored as:

```text
outputs/VidDub-Bench/
  same/<key>.mp4
  cross/<key>.mp4
  results.jsonl
```

The receipt is flushed after every completed call, so the command can resume
without recomputing finished outputs. Use `--overwrite` to regenerate them.

For a smoke test:

```bash
python scripts/run_benchmark.py \
  --benchmark data/VidDub-Bench \
  --model weights/VidDub \
  --weight-root weights \
  --output outputs/VidDub-Bench-smoke \
  --condition both \
  --limit 1
```

Runs can be partitioned with `--language`, `--start-index`, and `--limit`.
After the package has passed validation once, `--skip-validation` avoids
rehashing the 289 MB package on every partition.

## Reporting

Report same-audio and cross-audio results separately. State whether a metric
uses:

1. each method's successfully evaluated outputs, or
2. the all-method common intersection.

Do not silently substitute failed outputs or alter the frozen pairings. SyncNet
confidence is a synchronization proxy, not proof of phoneme correctness or
human preference. The public runner generates outputs; paper metric
dependencies such as LatentSync and SyncNet remain separate assets described
in [Evaluation scope](EVALUATION.md).

## License and intended use

VidDub-Bench is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Use is intended for non-commercial research and evaluation of consented video
dubbing. Do not use the benchmark or derived outputs for impersonation, fraud,
harassment, non-consensual sexual content, or deceptive political/media use.
