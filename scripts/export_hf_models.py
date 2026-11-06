"""Export selected VidDub checkpoints into upload-ready HF directories."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "weights" / "staged"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export"
PAPER_TITLE = "VidDub: Replacement-Aware Distillation for Two-Step Video Dubbing"
PAPER_URL = "https://arxiv.org/abs/2611.03404"
PROJECT_URL = "https://law1223.github.io/viddub-project-page/"
OPENDUB_URL = "https://github.com/LAW1223/OpenDub"
ARXIV_ID = "2611.03404"
PAPER_AUTHORS = """Liu, Yexin and Lin, Yunlong and Wang, Zehan and He, Guangxin and
             Xu, Xinli and Zha, Jiajun and Liu, Yaofu and Wang, Yueze and
             Ping, Bowen and Li, Yuanming and Wang, Amie and Ghanem, Bernard and
             Li, Bing and Yang, Harry"""

MODELS = {
    "sft": {
        "repo": "VidDub-SFT",
        "source": "sft-r2a-ema200",
        "step": 200,
        "variant": "sft",
        "stage": "bidirectional_sft",
        "student_steps": 30,
        "sampling": "flow_euler",
        "teacher_cfg": None,
        "description": "Best bidirectional VidDub SFT checkpoint (r2a EMA@200).",
        "lineage": ["r2a EMA@200"],
    },
    "final": {
        "repo": "VidDub",
        "source": "FULL2_DMD_R4",
        "source_run": "FULL2_DMD_R4",
        "step": 100,
        "variant": "full_context_2step",
        "causal": False,
        "stage": "dmd",
        "student_steps": 2,
        "sampling": "flow_euler",
        "teacher_cfg": 2.0,
        "description": "Paper-selected full-context two-step checkpoint.",
        "lineage": ["r2a EMA@200", "FULL2_ODE_R3@400", "FULL2_DMD_R4@100"],
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(path):
    state = load_file(str(path), device="cpu")
    return len(state), sum(t.numel() for t in state.values()), {
        str(t.dtype).replace("torch.", "") for t in state.values()
    }


def convert_component(source, destination, component, dtype):
    state = load_file(str(source), device="cpu")
    converted = {}
    for name, tensor in state.items():
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Non-finite tensor: {component}.{name}")
        converted[name] = tensor.to(dtype=dtype).contiguous()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(
        converted,
        str(temporary),
        metadata={
            "format": "pt",
            "project": "VidDub",
            "component": component,
            "dtype": str(dtype).replace("torch.", ""),
        },
    )
    temporary.replace(destination)
    del state, converted


def public_config(spec):
    return {
        "_class_name": "VidDubModel",
        "format_version": 1,
        "model_type": "viddub",
        "variant": spec["variant"],
        "causal": bool(spec.get("causal", False)),
        "stage": spec["stage"],
        "base_model": "Wan2.1-VACE-1.3B",
        "audio_encoder": "openai/whisper-small",
        "fps": 24,
        "sample_rate": 16000,
        "num_frames": 81,
        "latent_chunk_size": 3,
        "num_reference_latents": 1,
        "shift": 5.0,
        "student_steps": spec["student_steps"],
        "inference_audio_cfg": 1.0,
        "training_teacher_cfg": spec["teacher_cfg"],
        "sampling": spec["sampling"],
        "paper_selected": spec["stage"] == "dmd",
        "source_run": spec.get("source_run"),
        "trained_component_parameters": 2440958285,
        "parameter_count_scope": "DiT + VACE + audio adapter; external VAE, T5, and Whisper excluded",
        "precision": {
            "dit": "bfloat16",
            "vace": "bfloat16",
            "audio_adapter": "float32",
        },
        "lineage": spec["lineage"],
        "components": {
            "dit": "dit/diffusion_pytorch_model.safetensors",
            "vace": "vace/diffusion_pytorch_model.safetensors",
            "audio_adapter": "audio_adapter/model.safetensors",
        },
    }


def model_card(spec, hf_namespace=None, github_url=None, license_id=None, paper_url=None):
    paper_url = paper_url or PAPER_URL
    github_url = github_url or "https://github.com/LAW1223/VidDub"
    steps = spec["student_steps"]
    mode = ("bidirectional SFT" if spec["stage"] == "bidirectional_sft"
            else f"full-context {steps}-step DMD")
    model_id = f"{hf_namespace}/{spec['repo']}" if hf_namespace else f"weights/{spec['repo']}"
    license_header = f"license: {license_id}\n" if license_id else ""
    if license_id == "other":
        license_header += (
            "license_name: viddub-individual-commercial-research-1.0\n"
            "license_link: LICENSE\n"
        )
    install = (
        f"git clone {github_url}\ncd VidDub\npip install -r requirements.txt"
        if github_url
        else "cd /path/to/VidDub\npip install -r requirements.txt"
    )
    citation = f"""```bibtex
@article{{liu2026viddub,
  title   = {{{PAPER_TITLE}}},
  author  = {{{PAPER_AUTHORS}}},
  journal = {{arXiv preprint arXiv:{ARXIV_ID}}},
  year    = {{2026}}
}}
```"""
    license_text = (
        "This release uses the [VidDub Individual Commercial and Research "
        "License](LICENSE). Individuals may use VidDub personally or "
        "commercially, including monetizing completed outputs. Non-commercial "
        "academic research is permitted. Company use, SaaS, APIs, hosted "
        "inference, web applications, bots, platform integrations, and other "
        "managed or remote services require separate written permission. "
        "Third-party terms apply independently."
        if license_id == "other"
        else f"This release is distributed under `{license_id}`. Base-model and "
             "auxiliary-weight terms apply independently."
        if license_id
        else "The model owner must select and add the project/model license before "
             "public upload. Base-model and auxiliary-weight terms apply independently."
    )
    download_block = (
        f"""`--model` accepts either a Hugging Face repo ID or a local model directory.
To download this release for offline use:

```bash
huggingface-cli download {model_id} \\
  --local-dir weights/{spec['repo']}
```"""
        if hf_namespace
        else """This staging card uses a local model directory. Supply `--hf-namespace`
to the exporter when preparing a Hugging Face upload card."""
    )
    step_note = (f"The full-context sampling step count is fixed to **{steps}** by the model config. "
                 "Inference audio CFG is fixed to **1**."
                 if spec["stage"] != "bidirectional_sft"
                 else f"The default SFT sampling step count is **{steps}** and can be changed with `--sft-steps`.")
    return f"""---
{license_header}library_name: viddub
pipeline_tag: video-to-video
base_model: Wan-AI/Wan2.1-VACE-1.3B
arxiv: {ARXIV_ID}
tags:
- video-dubbing
- lip-sync
- audio-driven-video
- wan
---

# {spec['repo']}

**{PAPER_TITLE}**

{spec['description']}

- [Paper]({paper_url})
- [Project page]({PROJECT_URL})
- [Code]({github_url})
- [OpenDub application]({OPENDUB_URL})
- [VidDub-Bench](https://huggingface.co/datasets/AIPeanutman/VidDub-Bench)

## News

- **2026-11-06:** [OpenDub]({OPENDUB_URL}), a browser-based end-to-end video
  translation and dubbing workbench, released.
- **2026-11-03:** Training code released.
- **2026-11-03:** Model weights released.
- **2026-11-02:** Paper released on [arXiv]({paper_url}).

## Quick Start

Requirements: Linux, an NVIDIA CUDA GPU, Python 3.10+, Git, and FFmpeg.

### 1. Install VidDub

```bash
{install}
```

### 2. Download shared base dependencies

The model repository contains the trained VidDub components only. Download
the shared Wan VAE/T5 and Whisper encoder once:

```bash
mkdir -p weights
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \\
  --local-dir weights/Wan2.1-VACE-1.3B
huggingface-cli download openai/whisper-small \\
  --local-dir weights/whisper-small
```

### 3. Run full-video dubbing

The source video must contain a detectable, mostly front-facing face and be at
least 81 frames long. The target audio is automatically resampled to 16 kHz.

```bash
python scripts/infer.py \\
  --model {model_id} \\
  --weight-root ./weights \\
  --video input.mp4 \\
  --audio target.wav \\
  --output output.mp4
```

{download_block}

{step_note}
The output keeps the input frame rate and resolution; generation is performed
with the paper's overlapping-window protocol and pasted back into the
source video. Exactly 81 frames use one full-context call; longer videos use 81-frame
windows with stride 64 and 17 shared frames. Each window has its own seed,
overlaps are linearly blended, and lower-face color harmonization is enabled.

### Useful options

```text
--mode auto       One call for 81 frames; bounded windows for longer videos
--overlap 17      Paper protocol: 17 shared frames, stride 64
--no-color-harmonization
                  Disable lower-face Lab correction (ablation)
--feather 15      Mouth-region paste-back feather radius
--seed 0          Sampling seed
--device cuda     Inference device
--sft-steps 30    SFT only: override denoising steps
```

## OpenDub application

[OpenDub]({OPENDUB_URL}) is a separate Apache-2.0 browser workbench for ASR,
translation, TTS, timeline editing, media composition, and pluggable lip-sync
providers. It does not redistribute or currently wrap the released VidDub
full-context inference stack; use the CLI above for this model. A future or
custom VidDub adapter remains subject to the VidDub license, and third-party
access through a hosted service, website, API, bot, or managed deployment
requires separate written permission.

## Benchmark

Download the frozen 256-source same/cross benchmark and validate it with the
code repository:

```bash
huggingface-cli download AIPeanutman/VidDub-Bench \\
  --repo-type dataset \\
  --local-dir data/VidDub-Bench
python scripts/validate_benchmark.py --root data/VidDub-Bench
```

Use `scripts/run_benchmark.py` to run all fixed same- and cross-audio
conditions while loading this model only once. See the
[benchmark guide]({github_url}/blob/main/docs/BENCHMARK.md) for the exact
protocol and resumable execution options.

## Model

- Mode: `{mode}`
- Video rate: 24 FPS
- Audio rate: 16 kHz mono
- Window: 81 frames
- Long-video stride / overlap: 64 / 17 frames
- Inference audio CFG: 1
- Sampling steps: {steps}
- Base dependency: Wan2.1-VACE-1.3B
- Audio dependency: Whisper-small

The repository stores the trained DiT, VACE and VidDub audio adapter. The
base VAE, T5/tokenizer and Whisper checkpoint are not duplicated here.
The released trained components contain **2,440,958,285 parameters** in total;
“1.3B” refers only to the Wan backbone class and is not the complete inference
parameter count.

## Files

- `dit/diffusion_pytorch_model.safetensors`: BF16 Wan DiT
- `vace/diffusion_pytorch_model.safetensors`: BF16 VACE
- `audio_adapter/model.safetensors`: FP32 VidDub audio adapter
- `config.json`: public architecture and sampling configuration
- `checksums.json`: SHA256 and tensor metadata
- `LICENSE`: VidDub Individual Commercial and Research License 1.0

## Limitations

The current full-video implementation is offline/windowed rather than a
persistent real-time streaming server. Large head motion, profile faces,
occlusions, multiple faces and shots shorter than 81 frames are not the primary
operating regime.

## Responsible use

Use only with media and identities for which you have permission. Do not use
for impersonation, fraud, harassment, non-consensual sexual content, or
deceptive political/media manipulation. Generated or modified media should be
clearly disclosed where appropriate.

## License

{license_text}

## Citation

{citation}
"""


def verified_sources(spec, checkpoint_root):
    source = checkpoint_root / spec["source"]
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing staged checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("step") != spec["step"] or manifest.get("stage") != spec["stage"]:
        raise ValueError(f"Checkpoint manifest does not match export spec: {manifest_path}")
    sources = {}
    for component in ("dit", "vace", "adapter"):
        entry = manifest["files"][component]
        path = source / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for staged {component}: {path}")
        sources[component] = path
    return sources, manifest


def export_one(spec, checkpoint_root, output_root, force=False, *,
               hf_namespace=None, github_url=None, license_id=None, paper_url=None):
    output = output_root / spec["repo"]
    if output.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "LICENSE", output / "LICENSE")

    step = spec["step"]
    raw_sources, source_manifest = verified_sources(spec, checkpoint_root)
    sources = {
        "dit": raw_sources["dit"],
        "vace": raw_sources["vace"],
        "audio_adapter": raw_sources["adapter"],
    }

    destinations = {
        "dit": output / "dit/diffusion_pytorch_model.safetensors",
        "vace": output / "vace/diffusion_pytorch_model.safetensors",
        "audio_adapter": output / "audio_adapter/model.safetensors",
    }
    convert_component(sources["dit"], destinations["dit"], "dit", torch.bfloat16)
    convert_component(sources["vace"], destinations["vace"], "vace", torch.bfloat16)
    convert_component(sources["audio_adapter"], destinations["audio_adapter"], "audio_adapter", torch.float32)

    config = public_config(spec)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model_index.json").write_text(json.dumps({
        "_class_name": "VidDubPipeline",
        "_diffusers_version": None,
        "dit": ["viddub", "WanModel"],
        "vace": ["viddub", "VaceWanModel"],
        "audio_adapter": ["viddub", "DubAudioAdapter"],
    }, indent=2) + "\n")
    (output / "README.md").write_text(
        model_card(spec, hf_namespace, github_url, license_id, paper_url)
    )

    receipt = {
        "schema": 1,
        "model": spec["repo"],
        "source_selection": {
            "stage": spec["stage"],
            "step": step,
            "source_run": spec.get("source_run"),
            "staged_artifact": spec["source"],
            "lineage": spec["lineage"],
            "raw_files": source_manifest["files"],
        },
        "files": {},
    }
    for component, path in destinations.items():
        tensors, parameters, dtypes = tensor_summary(path)
        receipt["files"][component] = {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tensors": tensors,
            "parameters": parameters,
            "dtypes": sorted(dtypes),
        }
    (output / "checksums.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                        help="Portable staged checkpoints containing per-artifact manifests")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["final"])
    parser.add_argument("--hf-namespace")
    parser.add_argument("--github-url")
    parser.add_argument("--license", dest="license_id", default="other",
                        help="Owner-approved SPDX/Hugging Face license identifier")
    parser.add_argument("--paper-url")
    parser.add_argument("--public", action="store_true",
                        help="Require owner-controlled metadata for an upload-ready export")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.public:
        missing = [
            name for name, value in (
                ("--hf-namespace", args.hf_namespace),
                ("--github-url", args.github_url),
                ("--license", args.license_id),
                ("--paper-url", args.paper_url),
            ) if not value
        ]
        if missing:
            parser.error("--public requires " + ", ".join(missing))
    for name in args.models:
        output = export_one(
            MODELS[name],
            args.checkpoint_root.resolve(),
            args.output_root.resolve(),
            args.force,
            hf_namespace=args.hf_namespace,
            github_url=args.github_url,
            license_id=args.license_id,
            paper_url=args.paper_url,
        )
        print(output, flush=True)


if __name__ == "__main__":
    main()
