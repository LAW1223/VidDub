---
pretty_name: VidDub Training Clips
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

# VidDub Training Clips

The processed, model-ready version of the frozen paper-selected VidDub
training set. It contains 85,435 synthetic talking-head windows derived from
[VidDub-Training-Data](https://huggingface.co/datasets/AIPeanutman/VidDub-Training-Data).

## Contents

Each sample contains:

- `video.mp4`: 81 RGB frames at 512×512 and 24 FPS;
- `audio.wav`: aligned 16 kHz mono speech;
- `mask.png`: editable lower-face region;
- `ref.png`: appearance reference frame.

The package is approximately 100.1 GB and is split into 86 uncompressed ZIP
shards of at most 1,000 clips each. Model-specific latent caches are not
included.

## Download and extract

```bash
huggingface-cli download AIPeanutman/VidDub-Training-Clips \
  --repo-type dataset \
  --local-dir data/VidDub-Training-Clips

cd data/VidDub-Training-Clips
sha256sum -c manifest/checksums.sha256
for archive in zips/*.zip; do unzip -q "$archive"; done
```

Extracting every shard creates:

```text
clips/<shard>/<clip_id>/
  video.mp4
  audio.wav
  mask.png
  ref.png
```

`manifest/items.jsonl` is authoritative for membership and preprocessing
metadata. Every ZIP also contains the rows for its own clips at
`manifest/shard-items-<index>.jsonl`.

## Relationship to the source dataset

The source dataset stores 45,221 original-resolution generated videos and the
exact clip-level selection. This package materializes only those 85,435
selected windows. It does not add identities or samples and remains disjoint
from VidDub-Bench.

## License and responsible use

VidDub Training Clips is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Use it for lawful non-commercial research, education, and evaluation of
consented video dubbing. Do not use it for impersonation, fraud, harassment,
non-consensual sexual content, biometric surveillance, or deceptive
political/media manipulation.

All identities are synthetic. Generated media may contain artifacts,
stereotypes, demographic imbalance, or inaccurate lip motion. Generation
prompts are not verified attributes of real people.

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
