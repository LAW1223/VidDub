---
pretty_name: VidDub Training Data
license: cc-by-nc-4.0
task_categories:
- video-to-video
language:
- en
- zh
size_categories:
- 10K<n<100K
tags:
- video-dubbing
- talking-head
- audio-driven
- synthetic
---

# VidDub Training Data

The paper-selected synthetic H3 training media for
[VidDub](https://github.com/LAW1223/VidDub), released as original-resolution
source videos plus the exact clip-level selection used in the paper.

- [Paper](https://arxiv.org/abs/2611.03404)
- [Code and preprocessing](https://github.com/LAW1223/VidDub)
- [Project page](https://law1223.github.io/viddub-project-page/)
- [Model weights](https://huggingface.co/AIPeanutman/VidDub)
- [VidDub-Bench](https://huggingface.co/datasets/AIPeanutman/VidDub-Bench)

## Dataset summary

- **45,221** synthetic source videos
- **85,435** selected training windows
- Source media: **1344×768**, 24 FPS, approximately 4.46 seconds
- Model-ready clips: **81 frames**, 512×512, 24 FPS
- Model-ready audio: 16 kHz mono
- Languages by source: 27,630 English / 17,591 Chinese
- Languages by selected window: 52,658 English / 32,777 Chinese
- Logical source-media size: approximately **211.2 GiB**
- Frozen paper selection: `manifest/release_selection.jsonl`

The package deliberately stores original videos. The 512x512 face crops,
aligned 16 kHz audio, masks, reference frames, and optional latent caches are
derived locally by the source repository's `data/` pipeline. This avoids
distributing model-specific caches and allows users to inspect the original
generated media.

## Download

```bash
huggingface-cli download AIPeanutman/VidDub-Training-Data \
  --repo-type dataset \
  --local-dir data/VidDub-Training-Data
```

The full package requires at least 212 GiB before derived clips and caches.
Download to a filesystem with sufficient free space.

## Reproduce the processed clips

```bash
git clone https://github.com/LAW1223/VidDub.git
cd VidDub
pip install -r requirements.txt

export VIDDUB_DATA_ROOT=/path/to/VidDub-Training-Data
export VIDDUB_WEIGHT_ROOT=/path/to/weights
export VIDDUB_LATENTSYNC_ROOT=/path/to/LatentSync

python scripts/validate_training_dataset.py \
  --root "$VIDDUB_DATA_ROOT" \
  --manifests-only

python -m data.s01_analyze --shard 0 --num_shards 64
python -m data.s02_make_clips \
  --selection "$VIDDUB_DATA_ROOT/manifest/release_selection.jsonl" \
  --shard 0 --num_shards 64
python -m data.s03_cache_latents --shard 0 --num_shards 64
```

Launch every shard index from 0 through 63. `s02_make_clips` applies the frozen
release selection at clip level: 40,214 sources contribute two accepted
windows and 5,007 sources contribute one. Do not infer or materialize an
unlisted sibling window.

## Layout

```text
videos/<shard>/<source_id>.mp4        original-resolution media
manifest/s00_manifest.jsonl           artifact-relative source manifest
manifest/release_selection.jsonl      exact selected clip IDs and crop metadata
manifest/s01_reference.jsonl          reference preprocessing measurements
manifest/quality_selected.jsonl       quality records for selected clips
manifest/release.json                 release contract and counts
```

`manifest/s00_manifest.jsonl` uses artifact-relative `video_path` values.
`release_selection.jsonl` is authoritative for membership; `dir` is the
relative destination of a derived clip, not a pre-generated file in this
package.

## Selection protocol

The frozen selection is the H3 portion of the paper's `weights_v1` recipe.
Candidates first require IQA ≥ 40 and absolute SyncNet offset < 4. The bottom
20% of the within-domain composite percentile of SyncNet confidence, IQA, and
mouth sharpness is then removed. The published selection is already filtered;
do not apply the quality rule again.

## Data provenance and limitations

All videos are generated media with synthetic identities and are approved by
the dataset owner for this release. Text prompts intentionally vary language,
apparent age, gender presentation, ethnicity, setting, and emotion. These are
generation instructions—not verified attributes of real people. Generated
media may contain artifacts, stereotypes, demographic imbalance, inaccurate
lip motion, or other model biases. The dataset contains synthetic depictions
that include teenagers; it is not intended for identity recognition,
biometric inference, or claims about real demographic groups.

## License and responsible use

VidDub Training Data is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Use it for lawful non-commercial research, education, and evaluation of
consented video dubbing. Do not use it for impersonation, fraud, harassment,
non-consensual sexual content, biometric surveillance, or deceptive
political/media manipulation.

Report correction, removal, or licensing requests through the
[official VidDub repository](https://github.com/LAW1223/VidDub/issues).

## Citation

```bibtex
@article{liu2026viddub,
  title   = {VidDub: Replacement-Aware Distillation for Two-Step Video Dubbing},
  author  = {Liu, Yexin and Lin, Yunlong and Wang, Zehan and He, Guangxin and
             Xu, Xinli and Zha, Jiajun and Liu, Yaofu and Wang, Yueze and
             Ping, Bowen and Li, Yuanming and Wang, Amie and Ghanem, Bernard and
             Li, Bing and Yang, Harry},
  journal = {arXiv preprint arXiv:2611.03404},
  year    = {2026}
}
```
