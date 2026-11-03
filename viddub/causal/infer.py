"""Generate one prepared face clip for same/cross audio comparison.

This CLI outputs a cropped clip; full-video detection/paste-back remains in the
the full-video inference entry point. It does not pretend to be an end-to-end streaming server.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .runtime import (atomic_json, autocast, checkpoint_paths, load_components,
                      local_imports, read_config, require_causal_metadata)
from .sampling import rollout, schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--clip-id", required=True, help="Prepared validation clip id in s02 manifest")
    p.add_argument("--audio", help="Optional replacement WAV; original clip audio otherwise")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, help="AR defaults to 30; ODE/DMD requires trained step count")
    p.add_argument("--audio-cfg", type=float, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="Reference recomputation implementation")
    a = p.parse_args()
    cfg = read_config(a.config)
    meta = require_causal_metadata(a.checkpoint_dir, a.step, cfg)
    steps = a.steps if a.steps is not None else (30 if meta["stage"] == "ar" else cfg["student_steps"])
    if meta["stage"] != "ar":
        if steps != meta["config"]["student_steps"] or cfg["shift"] != meta["config"]["shift"]:
            raise ValueError("Few-step inference must use the training schedule")
        if a.audio_cfg != 1:
            raise ValueError("Guidance is distilled into ODE/DMD weights; use audio-cfg=1")
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    local_imports()
    import librosa
    from viddub.full_context import runtime as full_context_runtime
    from viddub.full_context.runtime import ZeroLatents, build_inputs, compute_latents, load_clip, load_rows, write_mp4
    if cfg.get("clips_root"):
        full_context_runtime.CLIPS = Path(cfg["clips_root"])
    rows = {r["clip_id"]: r for r in load_rows("val")}
    if a.clip_id not in rows:
        raise ValueError("clip-id must be from the validation split")
    if not torch.cuda.is_available():
        raise RuntimeError("Full inference requires CUDA")
    device = torch.device("cuda")
    pipe, whisper, model = load_components(checkpoint_paths(a.checkpoint_dir, a.step), cfg, device)
    model.requires_grad_(False).eval()
    clip = load_clip(rows[a.clip_id], cfg["num_frames"])
    audio = librosa.load(a.audio, sr=16000, mono=True)[0] if a.audio else clip["audio"]
    length = round(cfg["num_frames"] / cfg["fps"] * 16000)
    audio = np.pad(audio[:length], (0, max(0, length - len(audio))))
    started = time.monotonic()
    cached = full_context_runtime.CLIPS / rows[a.clip_id]["dir"] / cfg["latents_name"]
    if cached.is_file():
        lat = torch.load(cached, map_location="cpu", weights_only=False)
    else:
        if pipe.text_encoder is None:
            raise ValueError("No cached latents: unset VIDDUB_SKIP_TEXT_ENCODER for text encoding")
        pipe.text_encoder.to(device)
        lat = compute_latents(pipe, clip)
        pipe.text_encoder.cpu()
    x0, vc, ctx = build_inputs(lat, clip["mask"], cfg["num_frames"], ZeroLatents(pipe),
                                drop_ref=cfg["num_ref"] == 0)
    dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32
    hs = whisper([audio], device, torch.bfloat16)
    conditions = {"vace_context": vc.to(dtype), "context": ctx.to(dtype),
                  "whisper_hs": hs.to(dtype), "num_frames": cfg["num_frames"]}
    torch.manual_seed(a.seed)
    torch.cuda.synchronize()
    denoise_start = time.monotonic()
    with torch.no_grad(), autocast(device, cfg["precision"]):
        if cfg['sampler']=='cf_renoise':
            from .cf.sampling import cf_rollout
            latents=cf_rollout(model,x0.to(dtype),conditions,steps=steps,shift=cfg['shift'],use_cache=not a.no_cache)
        else:
            latents = rollout(model, x0.to(dtype), conditions, schedule(steps, cfg["shift"], device),
                              use_cache=not a.no_cache, audio_cfg=a.audio_cfg)
    torch.cuda.synchronize()
    denoise_seconds = time.monotonic() - denoise_start
    video = pipe.vae.decode(latents[:, :, cfg["num_ref"]:].to(pipe.torch_dtype), device=device)
    frames = ((video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    write_mp4(a.output, frames, audio, cfg["fps"])
    report = {"clip_id": a.clip_id, "step": a.step, "stage": meta["stage"], "sampling_steps": steps,
              "cached": not a.no_cache, "sampler":cfg['sampler'], "denoise_seconds": denoise_seconds,
              "total_seconds": time.monotonic() - started,
              "frames": len(frames), "note": "Offline clip; excludes model load, no streaming latency claim"}
    atomic_json(Path(a.output).with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
