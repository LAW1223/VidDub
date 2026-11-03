"""Run clip or bounded-window VidDub inference from a local/HF model repo.

Example:
    python scripts/infer.py --model organization/VidDub \
        --video input.mp4 --audio target.wav --output output.mp4
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def import_runtime():
    """Import GPU/media dependencies only after CLI parsing succeeds."""
    global decord, librosa, np, torch, snapshot_download, load_file
    global CausalConfig, CausalWanDub, autocast, rollout, schedule
    global DubAudioAdapter, full_context_runtime, FPS, SR, ZeroLatents
    global build_inputs, load_models, to_video_tensor, write_mp4, dub_video

    import decord
    import librosa
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from viddub.causal.model import CausalConfig, CausalWanDub
    from viddub.causal.runtime import autocast
    from viddub.causal.sampling import rollout, schedule
    from viddub.full_context.audio import DubAudioAdapter
    from viddub.full_context import runtime as full_context_runtime
    from viddub.full_context.runtime import (
        FPS, SR, ZeroLatents, build_inputs, load_models, to_video_tensor, write_mp4,
    )
    from viddub.full_context.inference import dub_video


def resolve_model(value, revision=None):
    local = Path(value).expanduser()
    if local.is_dir():
        return local.resolve()
    return Path(snapshot_download(
        repo_id=value, revision=revision,
        allow_patterns=["*.json", "*.md", "*.safetensors",
                        "dit/*", "vace/*", "audio_adapter/*"]))


def component_paths(model_dir, config):
    paths = {name: model_dir / relative
             for name, relative in config["components"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    return paths


def load_release(model_dir, device):
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "viddub":
        raise ValueError("The model directory is not a VidDub release")
    required = {
        "stage", "student_steps", "shift", "fps", "latent_chunk_size",
        "num_reference_latents", "inference_audio_cfg", "components",
    }
    missing_keys = sorted(required - config.keys())
    if missing_keys:
        raise ValueError(f"Incomplete VidDub config: {missing_keys}")
    if float(config["inference_audio_cfg"]) != 1.0:
        raise ValueError("Published VidDub checkpoints require inference audio CFG 1")
    if config["stage"] != "bidirectional_sft":
        if (
            config["stage"] != "dmd"
            or config.get("paper_selected") is not True
            or config.get("causal") is not False
            or config.get("source_run") != "FULL2_DMD_R4"
        ):
            raise ValueError(
                "The released two-step path requires the paper-selected "
                "full-context FULL2_DMD_R4 checkpoint")
    paths = component_paths(model_dir, config)
    pipe, whisper = load_models(device)
    pipe.dit.load_state_dict(load_file(str(paths["dit"]), device=str(device)), strict=True)
    pipe.vace.load_state_dict(load_file(str(paths["vace"]), device=str(device)), strict=True)
    adapter = DubAudioAdapter().to(device)
    adapter.load_state_dict(
        load_file(str(paths["audio_adapter"]), device=str(device)), strict=True)
    adapter.eval()
    return config, pipe, whisper, adapter


def two_step_sampler(model, config):
    @torch.no_grad()
    def sample(pipe, whisper, adapter, zero, clip, audio, steps=None,
               audio_cfg=1.0, seed=0, fps=FPS):
        trained_steps = int(config["student_steps"])
        if steps is not None and int(steps) != trained_steps:
            raise ValueError(
                f"This checkpoint was trained for {trained_steps} steps; got {steps}")
        if float(audio_cfg) != 1.0:
            raise ValueError("VidDub DMD checkpoints require audio CFG 1")
        num_frames = len(clip["frames"])
        video = to_video_tensor(clip["frames"], pipe)
        keep = torch.from_numpy(~clip["mask"]).to(video)[None, None, None]
        latents = {
            "x0": pipe.vae.encode(video, device=pipe.device)[0],
            "inactive": pipe.vae.encode(video * keep, device=pipe.device)[0],
            "ref": pipe.vae.encode(
                to_video_tensor(clip["ref"][None], pipe), device=pipe.device)[0],
        }
        x0, vace_context, context = build_inputs(
            latents, clip["mask"], num_frames, zero,
            drop_ref=int(config["num_reference_latents"]) == 0)
        whisper_hs = whisper([audio], pipe.device, torch.bfloat16)
        model.config = CausalConfig(
            int(config["latent_chunk_size"]),
            int(config["num_reference_latents"]), float(fps))
        conditions = {
            "context": context.to(torch.bfloat16),
            "vace_context": vace_context.to(torch.bfloat16),
            "whisper_hs": whisper_hs.to(torch.bfloat16),
            "num_frames": num_frames,
        }
        noise = torch.randn(
            x0.shape,
            generator=torch.Generator(device=x0.device).manual_seed(seed),
            device=x0.device, dtype=torch.bfloat16)
        with autocast(x0.device, "bf16"):
            output = rollout(
                model, x0.to(torch.bfloat16), conditions,
                schedule(trained_steps, float(config["shift"]), x0.device),
                noise=noise, use_cache=bool(config.get("causal", False)),
                audio_cfg=1.0)
        video = pipe.vae.decode(
            output[:, :, int(config["num_reference_latents"]):].to(pipe.torch_dtype),
            device=pipe.device)
        return ((video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1)
                * 127.5).round().byte().cpu().numpy()

    return sample


def load_inference_runtime(model, weight_root, device="cuda", revision=None,
                           sft_steps=None):
    """Load the release model once for one or more inference calls."""
    import_runtime()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full VidDub inference")
    weight_root = Path(weight_root).expanduser().resolve()
    full_context_runtime.WEIGHTS = weight_root
    full_context_runtime.VACE_DIR = weight_root / "Wan2.1-VACE-1.3B"
    required = [
        full_context_runtime.VACE_DIR / "diffusion_pytorch_model.safetensors",
        full_context_runtime.VACE_DIR / "Wan2.1_VAE.pth",
        full_context_runtime.VACE_DIR / "models_t5_umt5-xxl-enc-bf16.pth",
        full_context_runtime.VACE_DIR / "google/umt5-xxl",
        weight_root / "whisper-small/model.safetensors",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing base dependencies. Follow the model-card Quick Start. "
            f"Missing: {missing}")
    model_dir = resolve_model(model, revision)
    torch_device = torch.device(device)
    config, pipe, whisper, adapter = load_release(model_dir, torch_device)
    zero = ZeroLatents(pipe)

    sample_fn = None
    steps = int(config["student_steps"])
    if config["stage"] != "bidirectional_sft":
        causal = bool(config.get("causal", False))
        inference_model = CausalWanDub(
            pipe.dit, pipe.vace, adapter,
            CausalConfig(
                int(config["latent_chunk_size"]),
                int(config["num_reference_latents"]),
                float(config["fps"]),
            ),
            causal=causal,
        ).eval().requires_grad_(False)
        sample_fn = two_step_sampler(inference_model, config)
    elif sft_steps is not None:
        steps = int(sft_steps)
    return {
        "models": (pipe, whisper, adapter, zero),
        "config": config,
        "steps": steps,
        "sample_fn": sample_fn,
    }


def infer_file(runtime, video, audio, output, seed=0, mode="auto", overlap=17,
               feather=15, harmonize=True):
    """Run one media pair with an already loaded inference runtime."""
    video = str(video)
    audio = str(audio)
    output = Path(output)
    reader = decord.VideoReader(video, num_threads=2)
    fps = float(reader.get_avg_fps())
    frames = reader[:].asnumpy()
    waveform = librosa.load(audio, sr=SR, mono=True)[0]
    audio_samples = round(len(frames) / fps * SR)
    waveform = np.pad(
        waveform[:audio_samples],
        (0, max(0, audio_samples - len(waveform))),
    )
    generated = dub_video(
        runtime["models"], frames, waveform,
        steps=runtime["steps"], audio_cfg=1.0, feather=feather, seed=seed,
        fps=fps, overlap=overlap, mode=mode, harmonize=harmonize,
        sample_fn=runtime["sample_fn"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(output, generated, waveform, fps=fps)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Local model folder or Hugging Face repo ID")
    parser.add_argument("--revision")
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-root", type=Path,
        default=Path(os.environ.get("VIDDUB_WEIGHT_ROOT", PROJECT / "weights")),
        help="Directory containing Wan2.1-VACE-1.3B/ and whisper-small/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("auto", "clip", "windowed"), default="auto",
        help="auto uses clip mode for 81 frames and windowed mode otherwise")
    parser.add_argument(
        "--overlap", type=int, default=17,
        help="windowed shared frames (paper protocol: 17, giving stride 64)")
    parser.add_argument(
        "--no-color-harmonization", action="store_true",
        help="disable the paper's lower-face Lab correction and source feathering")
    parser.add_argument("--feather", type=int, default=15)
    parser.add_argument("--sft-steps", type=int,
                        help="Override SFT sampling steps; default from config")
    args = parser.parse_args()
    runtime = load_inference_runtime(
        args.model,
        args.weight_root,
        device=args.device,
        revision=args.revision,
        sft_steps=args.sft_steps,
    )
    infer_file(
        runtime,
        args.video,
        args.audio,
        args.output,
        seed=args.seed,
        mode=args.mode,
        overlap=args.overlap,
        feather=args.feather,
        harmonize=not args.no_color_harmonization,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
