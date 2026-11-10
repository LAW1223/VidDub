# VidDub data contract

The datasets are released as separate artifacts. This repository contains the
generation pipeline and schema, but intentionally contains no unreviewed person
video.

The frozen paper-selected synthetic source package is public as
[VidDub-Training-Data](https://huggingface.co/datasets/AIPeanutman/VidDub-Training-Data).
It contains 45,221 original-resolution generated source videos and an
authoritative 85,435-window selection under CC BY-NC 4.0. The corresponding
100.1 GB of materialized 81-frame media is available in 86 ZIP shards as
[VidDub-Training-Clips](https://huggingface.co/datasets/AIPeanutman/VidDub-Training-Clips).
Before processing a source-package download, validate its manifests:

```bash
python scripts/validate_training_dataset.py \
  --root data/VidDub-Training-Data \
  --manifests-only
```

The complete package is approximately 211.2 GiB before derived clips and
model-specific caches.

## Source manifest

Input is UTF-8 JSONL. Required fields:

```json
{"id":"clip_source_0001","video_path":"videos/0001.mp4"}
```

Recommended provenance fields:

```json
{
  "id": "clip_source_0001",
  "video_path": "videos/0001.mp4",
  "prompt": "optional visual description",
  "speech_text": "optional transcript",
  "language": "en",
  "source_url": "https://source.example/item",
  "source_dataset": "dataset-name",
  "license": "SPDX-or-source-specific-license",
  "consent_basis": "documented release basis",
  "redistribution_allowed": true
}
```

`s00_build_manifest.py` resolves relative paths and validates unique IDs.
Release media should remain at its original resolution. For the paper-selected
H3 package, source videos are 1344x768; the 512x512 representation is produced
only by `s02_make_clips.py`.

## Processed clip layout

Each accepted sample is transformed to:

```text
data/clips/<shard>/<clip_id>/
  video.mp4     # 81 RGB frames, 512x512, 24 fps
  audio.wav     # aligned mono PCM, 16 kHz
  mask.png      # 255 in the editable lower-face region
  ref.png       # appearance reference frame
  latents.pt    # VAE/T5 cache; optional in a media-only release
```

Manifests are written under `data/manifest/{s00,s01,s02,s03}`. Public manifests
must use artifact-relative paths; do not publish workstation paths.
Training reads the generated `s02/shard_*.jsonl` files and deterministically
splits source groups into train/validation. Each `dir` is relative to the
configured clips root, and each `clip_id` must appear in an `s03` completion
shard before cached training.

For a frozen release subset, pass its clip-level manifest to s02:

```bash
python -m data.s02_make_clips \
  --selection "$VIDDUB_DATA_ROOT/manifest/release_selection.jsonl" \
  --shard 0 --num_shards 1
```

Selection is clip-level rather than source-level. A source video can have two
candidate temporal windows while only one belongs to the released training
set. Unlisted windows must not be materialized.

`scripts/stage_h3_dataset.py` validates and stages an owner-authorized generated
media collection into this public data contract. It rejects processed 512x512
videos as release inputs and writes artifact-relative source and selection
manifests.

The paper's H3 membership is the 85,435 H3 keys in `weights_v1`, selected from
107,222 cached train clips. To rebuild that selection from the complete
pre-release pool, run:

```bash
python -m data.quality_score --rank 0 --world 1
python -m data.quality_weights \
  --name v1 --iqa-min 40 --off-max 4 --drop-frac 0.2
```

`quality_weights.py` first requires IQA >= 40 and absolute SyncNet offset < 4,
then removes the bottom 20% of the within-domain composite percentile of
SyncNet confidence, IQA, and mouth sharpness. Do not apply this filter again
to a separately distributed `VidDub-Training-Data` artifact:
`release_selection.jsonl` is already the frozen result.

The s01 implementation uses the MediaPipe `solutions` API and therefore pins
MediaPipe 0.10.21. On a headless machine, if another dependency overwrites
OpenCV with a GUI build, reinstall the pinned
`opencv-contrib-python-headless==4.11.0.86` package.

## Recommended release split

- `viddub-data-media`: video/audio/mask/ref and provenance manifest.
- `viddub-data-cache`: optional model-specific latent cache.
- `viddub-bench`: frozen evaluation inputs, only if each item is licensed.

Latent caches are derived from both the media and base model and may have
different redistribution constraints. Keep them separate.

## Mandatory review before upload

For every source collection, record:

- copyright/license and whether modified clips may be redistributed;
- voice/face/person consent or other lawful release basis;
- whether minors, biometric-sensitive content, or restricted identities occur;
- takedown contact and stable item IDs;
- train/validation/benchmark overlap policy;
- source URL and checksum where permitted.

HDTF, VoxCeleb2, scraped web video, and internal/generated person video must not
be assumed redistributable merely because they are used in research. If media
cannot be redistributed, publish download/preparation instructions and metadata
only, subject to the source terms.

The intended use is consented video dubbing and research. Prohibit impersonation,
fraud, harassment, non-consensual sexual content, and deceptive political/media
use in the dataset/model card and hosting terms.
