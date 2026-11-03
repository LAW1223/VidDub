"""Generate auditable causal teacher rollouts; never initialize from ground-truth video."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import (atomic_json, autocast, checkpoint_paths, fingerprint, load_components,
                      local_imports, make_dataset, prepare_clip, read_config, require_causal_metadata)
from .sampling import rollout, schedule


def safe_tensor_tree(value):
    """Convert trajectory payloads to weights_only-safe primitive/tensor trees."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {k: safe_tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(safe_tensor_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsafe trajectory payload type: {type(value).__name__}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="ODE-stage config; schedule must match student training")
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--teacher-step", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--teacher-steps", type=int, default=32)
    p.add_argument("--max-clips", type=int, default=256)
    p.add_argument("--audio-mode", choices=["same", "cross", "both"], default="both")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    cfg = read_config(a.config)
    paths = checkpoint_paths(a.teacher_dir, a.teacher_step)
    require_causal_metadata(a.teacher_dir, a.teacher_step, cfg, ("ar",))
    if a.teacher_steps < cfg["student_steps"] or a.teacher_steps % cfg["student_steps"]:
        raise ValueError("teacher-steps must be a positive multiple of student_steps")
    if a.max_clips < 1:
        raise ValueError("max-clips must be positive")
    if a.shards < 1 or not 0 <= a.shard < a.shards:
        raise ValueError("invalid shard assignment")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty trajectory output directory")
    dataset, rows = make_dataset(cfg)
    if a.audio_mode != "same" and len(dataset) < 2:
        raise ValueError("Cross-audio trajectories need at least two clips")
    if a.check_only:
        print(json.dumps({"status": "inputs_validated_no_gpu_execution", "clips": min(a.max_clips, len(dataset)),
                          "teacher_steps": a.teacher_steps, "student_steps": cfg["student_steps"]}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Trajectory generation requires CUDA")
    local_imports()
    from viddub.full_context import runtime as full_context_runtime
    from viddub.full_context.runtime import ZeroLatents
    from .runtime import PixelClipDataset
    if not isinstance(dataset, PixelClipDataset):
        dataset = PixelClipDataset(dataset, full_context_runtime.CLIPS)
    device = torch.device("cuda")
    pipe, whisper, model = load_components(paths, cfg, device)
    model.requires_grad_(False).eval()
    zero = ZeroLatents(pipe)
    sigmas = schedule(a.teacher_steps, cfg["shift"], device)
    interval = a.teacher_steps // cfg["student_steps"]
    manifest = {"schema": 1, "complete": False, "config": cfg,
                "student_steps": cfg["student_steps"], "teacher_sigmas": sigmas.cpu().tolist(),
                "boundary_indices": list(range(0, a.teacher_steps + 1, interval)),
                "teacher": fingerprint(paths), "teacher_cfg": cfg["teacher_cfg"], "records": []}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", manifest)
    for index in range(a.shard, min(a.max_clips, len(dataset)), a.shards):
        modes = ("same", "cross") if a.audio_mode == "both" else (a.audio_mode,)
        for mode in modes:
            clip = dataset[index]
            audio_id = clip["clip_id"]
            if mode == "cross":
                partner_id = rows[index].get('audio_cross_id')
                if partner_id is not None:
                    candidates = [j for j, row in enumerate(rows) if row['clip_id'] == partner_id]
                    if len(candidates) != 1:
                        raise ValueError('Frozen cross-audio partner missing or ambiguous')
                    other = dataset[candidates[0]]
                else:
                    other = dataset[(index + 1) % len(dataset)]
                clip["audio"], audio_id = other["audio"], other["clip_id"]
            x0, conditions = prepare_clip(clip, pipe, whisper, zero, cfg)
            conditions['cross'] = mode == 'cross'
            torch.manual_seed(cfg["seed"] + index * 2 + int(mode == "cross"))
            with torch.no_grad(), autocast(device, cfg["precision"]):
                generated, chunks = rollout(model, x0, conditions, sigmas, use_cache=cfg["use_cache"],
                                            audio_cfg=cfg["teacher_cfg"], trajectory=True)
            # Persist reference + shape only, never use GT video as trajectory input.
            shape_ref = torch.zeros_like(x0)
            shape_ref[:, :, :cfg["num_ref"]] = x0[:, :, :cfg["num_ref"]]
            name = f"{index:06d}_{mode}.pt"
            data = {"clip_id": clip["clip_id"], "audio_id": audio_id, "mode": mode,
                    "shape_reference": shape_ref.cpu(), "gt_latents": x0.cpu(),
                    "generated": generated.cpu(), "chunks": chunks,
                    "conditions": safe_tensor_tree(conditions)}
            torch.save(data, out / (name + ".tmp"))
            (out / (name + ".tmp")).replace(out / name)
            manifest["records"].append({"file": name, "clip_id": clip["clip_id"], "audio_id": audio_id,
                                        "mode": mode, "bytes": (out / name).stat().st_size})
            atomic_json(out / "manifest.json", manifest)
            print(json.dumps(manifest["records"][-1]), flush=True)
    manifest["complete"] = True
    atomic_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
