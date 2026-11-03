"""Local Wan-Dub data/checkpoint integration; imports heavy dependencies lazily."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import torch

from .model import CausalConfig, CausalWanDub

PROJECT = Path(__file__).resolve().parents[2]


def local_imports():
    """Compatibility hook retained for callers; package imports need no path edits."""


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read_config(path):
    # Normalize stored configs as well, so adding DISABLED optional features
    # does not prevent resuming an older, otherwise identical training state.
    cfg = dict(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
    allowed = {"stage", "causal", "chunk_size", "num_ref", "fps", "num_frames", "max_steps", "warmup",
               "lr", "lr_base", "weight_decay", "grad_accum", "grad_clip", "precision", "workers",
               "seed", "max_train_clips", "latents_name", "save_every", "log_every", "mouth_weight",
               "p_drop_audio", "p_cross", "shift", "student_steps", "teacher_cfg", "use_cache",
               "gradient_checkpointing", "fake_updates", "matching_min", "matching_max",
               "rows_file", "clips_root", "pixel_weight", "lpips_weight", "sync_weight",
               "keep_weight", "aux_sigma_max", "sync_sigma_max", "aux_max_latent_h", "aux_max_latent_w",
               "lr_schedule", "lr_min_ratio", "save_steps", "save_state_every", "ema_decay",
               "dmd_rollout_mode", "art_weight", "art_layer", "art_warmup", "art_lr"}
    allowed.update({"ode_gt_flow_weight", "rollout_quality_weight", "rollout_quality_cross",
                    "dmd_retention_weight"})
    allowed.update({"fake_lr", "fake_lr_base"})
    allowed.update({"dmd_score_offload", "dmd_split_quality_backward"})
    allowed.update({'cd_grid_steps','cd_ema_decay','cd_student_cache','cd_frozen_cache',
                    'cd_aux_scale','cd_all_chunks','beta1','beta2','sampler'})
    allowed.update({"ar_history_mode", "ar_history_prob_start", "ar_history_prob_end",
                    "ar_history_ramp_steps", "ar_history_steps", "ar_history_sigma_max",
                    "ar_blocks_per_sample", "ar_history_mouth_drop_prob",
                    "ar_history_mouth_drop_cross_only"})
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    defaults = dict(stage="ar", causal=True, chunk_size=3, num_ref=1, fps=24, num_frames=81,
                    max_steps=2000, warmup=100, lr=1e-5, lr_base=2e-6, weight_decay=0.01,
                    grad_accum=1, grad_clip=1.0, precision="bf16", workers=2, seed=1234,
                    max_train_clips=None, latents_name="latents.pt", save_every=250, log_every=10,
                    mouth_weight=3, p_drop_audio=0.1, p_cross=0.5, shift=5, student_steps=4, teacher_cfg=1,
                    use_cache=True, gradient_checkpointing=True, fake_updates=5,
                    matching_min=0.02, matching_max=0.98, rows_file=None, clips_root=None,
                    pixel_weight=0.0, lpips_weight=0.0, sync_weight=0.0, keep_weight=0.0,
                    aux_sigma_max=0.5, sync_sigma_max=None,
                    aux_max_latent_h=36, aux_max_latent_w=48,
                    lr_schedule='constant', lr_min_ratio=0.1, save_steps=[], save_state_every=0,
                    ema_decay=0.0, dmd_rollout_mode='legacy_early_exit',
                    art_weight=0.0, art_layer=8, art_warmup=50, art_lr=1e-4,
                    ode_gt_flow_weight=0.0, rollout_quality_weight=1.0,
                    rollout_quality_cross=False, dmd_retention_weight=0.0,
                    fake_lr=None, fake_lr_base=None, dmd_score_offload=False,
                    dmd_split_quality_backward=False)
    defaults.update(ar_history_mode='gt', ar_history_prob_start=0.0,
                    ar_history_prob_end=0.0, ar_history_ramp_steps=0,
                    ar_history_steps=4, ar_history_sigma_max=0.5,
                    ar_blocks_per_sample=1, ar_history_mouth_drop_prob=0.0,
                    ar_history_mouth_drop_cross_only=False)
    defaults.update(cd_grid_steps=48, cd_ema_decay=.99, cd_student_cache=False,
                    cd_frozen_cache=False, cd_aux_scale=1., cd_all_chunks=True,
                    beta1=.9, beta2=.999, sampler='euler')
    defaults.update(cfg)
    cfg = defaults
    if cfg['sync_sigma_max'] is None:
        cfg['sync_sigma_max'] = cfg['aux_sigma_max']
    CausalConfig(cfg["chunk_size"], cfg["num_ref"], cfg["fps"])
    if cfg["stage"] not in ("ar", "ode", "dmd", "cd") or cfg["precision"] not in ("bf16", "fp32"):
        raise ValueError("Unknown stage or precision")
    if cfg["num_frames"] < 1 or (cfg["num_frames"] - 1) % 4:
        raise ValueError("num_frames must be 4n+1")
    if cfg["fps"] != 24:
        raise ValueError("Existing cached training dataset is 24fps")
    for name in ("max_steps", "lr", "lr_base", "grad_accum", "grad_clip", "save_every",
                 "log_every", "shift", "student_steps", "fake_updates"):
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg["workers"] < 0 or cfg["warmup"] < 0 or not 0 <= cfg["p_drop_audio"] <= 1 or not 0 <= cfg["p_cross"] <= 1:
        raise ValueError("Invalid workers/warmup/audio dropout")
    if cfg["mouth_weight"] < 1 or cfg["teacher_cfg"] < 0:
        raise ValueError("Invalid mouth weight or guidance")
    if not 0 < cfg["matching_min"] < cfg["matching_max"] < 1:
        raise ValueError("Invalid DMD matching range")
    if cfg["stage"] == "dmd" and cfg["grad_accum"] != 1:
        raise ValueError("DMD alternating optimizers currently require grad_accum=1")
    if any(cfg[k] < 0 for k in ['pixel_weight','lpips_weight','sync_weight','keep_weight',
                                'ode_gt_flow_weight','rollout_quality_weight','dmd_retention_weight']):
        raise ValueError('Auxiliary weights must be nonnegative')
    if (cfg['lr_schedule'] not in ('constant','cosine') or
            not 0 < cfg['aux_sigma_max'] <= 1 or not 0 < cfg['sync_sigma_max'] <= 1):
        raise ValueError('Invalid scheduler/aux sigma')
    if not 0 <= cfg['ema_decay'] < 1:raise ValueError('Invalid EMA decay')
    if cfg['dmd_rollout_mode'] not in ('legacy_early_exit','final_step'):
        raise ValueError('Invalid DMD rollout mode')
    if not isinstance(cfg['dmd_split_quality_backward'], bool):
        raise ValueError('dmd_split_quality_backward must be boolean')
    if cfg['dmd_split_quality_backward'] and cfg['stage'] != 'dmd':
        raise ValueError('dmd_split_quality_backward is DMD-only')
    if (cfg['stage'] == 'dmd' and any(cfg[k] for k in
            ['pixel_weight','lpips_weight','sync_weight','keep_weight','dmd_retention_weight'])
            and cfg['dmd_rollout_mode'] != 'final_step'):
        raise ValueError('DMD quality/retention losses require final_step rollout')
    if cfg['art_weight'] < 0 or cfg['art_layer'] < 0 or cfg['art_warmup'] < 0 or cfg['art_lr'] <= 0:
        raise ValueError('Invalid Art configuration')
    cfg['fake_lr'] = cfg['lr'] if cfg['fake_lr'] is None else cfg['fake_lr']
    cfg['fake_lr_base'] = cfg['lr_base'] if cfg['fake_lr_base'] is None else cfg['fake_lr_base']
    if cfg['fake_lr'] <= 0 or cfg['fake_lr_base'] <= 0:
        raise ValueError('Fake-score learning rates must be positive')
    if cfg['ar_history_mode'] not in ('gt','resample','rollout'):
        raise ValueError('Invalid AR history mode')
    if not 0 <= cfg['ar_history_prob_start'] <= 1 or not 0 <= cfg['ar_history_prob_end'] <= 1:
        raise ValueError('AR history probabilities must lie in [0,1]')
    if cfg['ar_history_ramp_steps'] < 0 or cfg['ar_history_steps'] < 1:
        raise ValueError('Invalid AR history schedule')
    if not 0 < cfg['ar_history_sigma_max'] <= 1:
        raise ValueError('Invalid AR history sigma maximum')
    if not isinstance(cfg['ar_blocks_per_sample'], int) or cfg['ar_blocks_per_sample'] < 0:
        raise ValueError('ar_blocks_per_sample must be an integer >= 0 (0 means all blocks)')
    if cfg['stage'] != 'ar' and cfg['ar_blocks_per_sample'] != 1:
        raise ValueError('ar_blocks_per_sample is AR-only')
    if not 0 <= cfg['ar_history_mouth_drop_prob'] <= 1:
        raise ValueError('ar_history_mouth_drop_prob must lie in [0,1]')
    if not isinstance(cfg['ar_history_mouth_drop_cross_only'], bool):
        raise ValueError('ar_history_mouth_drop_cross_only must be boolean')
    if cfg['stage'] != 'ar' and (cfg['ar_history_mouth_drop_prob'] or
                                cfg['ar_history_mouth_drop_cross_only']):
        raise ValueError('AR history mouth dropout is AR-only')
    if cfg['ema_decay'] != 0:
        raise ValueError('EMA export is not implemented in this trainer; use ema_decay=0')
    if cfg['cd_grid_steps'] < 2 or not 0 <= cfg['cd_ema_decay'] < 1 or cfg['cd_aux_scale'] < 0:
        raise ValueError('Invalid consistency configuration')
    if not 0 <= cfg['beta1'] < 1 or not 0 <= cfg['beta2'] < 1:
        raise ValueError('Invalid optimizer betas')
    if cfg['sampler'] not in ('euler','cf_renoise'):
        raise ValueError('Invalid sampler')
    return cfg


def checkpoint_paths(directory, step):
    if not directory or step is None:
        raise ValueError("Provide a checkpoint directory AND step")
    paths = {name: Path(directory).resolve() / f"{name}_step{step:06d}.safetensors"
             for name in ("dit", "vace", "adapter")}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def fingerprint(paths):
    result = {}
    for name, path in paths.items():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(chunk)
        result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    return result


def load_components(paths, cfg, device):
    local_imports()
    from safetensors.torch import load_file
    from viddub.full_context import runtime as full_context_runtime
    # Explicit experiment-local cache, never changes the old training module
    # on disk. Other processes retain their existing paths.
    if os.environ.get("VIDDUB_WEIGHT_ROOT"):
        full_context_runtime.WEIGHTS = Path(os.environ["VIDDUB_WEIGHT_ROOT"])
        full_context_runtime.VACE_DIR = full_context_runtime.WEIGHTS / "Wan2.1-VACE-1.3B"
    from viddub.full_context.audio import DubAudioAdapter
    # A skipped T5 is safe only with the exact precomputed empty-prompt tensor.
    # If no cache was supplied, load/encode/unload T5 once instead of producing
    # a tokenizer=None failure on the first minibatch.
    empty_path = os.environ.get("VIDDUB_EMPTY_CONTEXT")
    empty_context = None
    if empty_path:
        empty_context = torch.load(empty_path, map_location="cpu", weights_only=True)
        if (not torch.is_tensor(empty_context) or empty_context.ndim != 2 or
                empty_context.shape[1] != 4096 or not 1 <= empty_context.shape[0] <= 512 or
                not torch.isfinite(empty_context).all()):
            raise ValueError("VIDDUB_EMPTY_CONTEXT must contain finite [L,4096] empty T5 embeddings")
    if os.environ.get("VIDDUB_SKIP_TEXT_ENCODER") == "1" and empty_context is not None:
        # Cached-data experiments never invoke T5; avoid loading its 11 GB
        # checkpoint for every short diagnostic/training run.
        from viddub.foundation.pipelines.wan_video import WanVideoPipeline, ModelConfig
        from viddub.full_context.audio import WhisperFeatures
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=[ModelConfig(path=str(full_context_runtime.VACE_DIR / "diffusion_pytorch_model.safetensors")),
                           ModelConfig(path=str(full_context_runtime.VACE_DIR / "Wan2.1_VAE.pth"))],
            tokenizer_config=None)
        for m in (pipe.dit, pipe.vace, pipe.vae):
            m.requires_grad_(False).eval()
        whisper = WhisperFeatures(str(full_context_runtime.WEIGHTS / "whisper-small")).to(device, torch.bfloat16)
    else:
        pipe, whisper = full_context_runtime.load_models(device)
        # Evaluation needs only the empty prompt. Encode it once while T5 is
        # resident on the correct device, then keep the frozen encoder off GPU.
        pipe._wan_dub_empty_context = full_context_runtime.encode_text(pipe, "").detach().cpu()
    if empty_context is not None:
        pipe._wan_dub_empty_context = empty_context
    adapter = DubAudioAdapter().to(device)
    for name, module in (("dit", pipe.dit), ("vace", pipe.vace), ("adapter", adapter)):
        # Preserve FP32 fine-tuning updates when loading a causal checkpoint;
        # loading into a BF16 base module first silently rounds those updates.
        module.float()
        module.load_state_dict(load_file(str(paths[name])), strict=True)
    model = CausalWanDub(
        pipe.dit,
        pipe.vace,
        adapter,
        CausalConfig(cfg["chunk_size"], cfg["num_ref"], cfg["fps"]),
        causal=cfg.get("causal", True),
    )
    # FP32 master weights in both modes; BF16 only applies to compute.
    model.float()
    # Training/inference here consumes cached text. Keep T5 off the GPU unless
    # an explicit encoding operation calls the pipeline's model manager.
    if pipe.text_encoder is not None:
        pipe.text_encoder.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pipe, whisper, model


def autocast(device, precision):
    if precision == "bf16":
        # Rollout switches no_grad/grad within one autocast scope. Reusing a
        # weight cast cached during no_grad can silently drop weight gradients
        # and changes checkpoint-recomputation saved tensor metadata.
        return torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)
    return nullcontext()


def export_weights(model, directory, step, cfg, provenance):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("dit", "vace", "adapter"):
        dst = directory / f"{name}_step{step:06d}.safetensors"
        temp = dst.with_suffix(".safetensors.tmp")
        # Keep fp32 masters; existing inference loaders may cast to bf16.
        save_file({k: v.detach().float().cpu().contiguous()
                   for k, v in getattr(model, name).state_dict().items()}, str(temp))
        temp.replace(dst)
    atomic_json(directory / f"causal_step{step:06d}.json",
                {"schema": 1, "step": step, "stage": cfg["stage"], "config": cfg,
                 "architecture": model.architecture(), "provenance": provenance,
                 "quality_validated": False})


def require_causal_metadata(directory, step, cfg, stages=None):
    path = Path(directory) / f"causal_step{step:06d}.json"
    data = json.loads(path.read_text())
    if data.get("architecture", {}).get(
        "causal", data["config"].get("causal", True)
    ) != cfg["causal"]:
        raise ValueError("Checkpoint/config causal architecture mismatch")
    for key in ("chunk_size", "num_ref", "fps", "num_frames", "shift"):
        if data["config"][key] != cfg[key]:
            raise ValueError(f"Checkpoint/config mismatch for {key}")
    if stages and data["stage"] not in stages:
        raise ValueError(f"Expected stages {stages}, got {data['stage']}")
    if data["stage"] in ("ode", "dmd", "cd") and data["config"]["student_steps"] != cfg["student_steps"]:
        raise ValueError("Checkpoint/config mismatch for student_steps")
    if data['stage'] in ('ode','dmd','cd') and data['config'].get('sampler','euler') != cfg.get('sampler','euler'):
        raise ValueError('Checkpoint/config sampler mismatch')
    return data


def make_dataset(cfg):
    local_imports()
    from viddub.full_context import runtime as full_context_runtime
    from viddub.full_context.runtime import CachedClipDataset, load_rows, cached_clip_ids
    if cfg.get("clips_root"):
        full_context_runtime.CLIPS = Path(cfg["clips_root"])
    rows = json.loads(Path(cfg["rows_file"]).read_text()) if cfg.get("rows_file") else load_rows("train")
    sources = sorted({r.get('source', 'h3') for r in rows})
    ids = cached_clip_ids(sources)
    if cfg.get('rows_file'):
        missing = [r['clip_id'] for r in rows if r['clip_id'] not in ids]
        if missing:
            raise ValueError(f'Explicit manifest contains {len(missing)} uncached clips: {missing[:5]}')
    rows = [r for r in rows if r["clip_id"] in ids]
    if cfg["max_train_clips"] is not None:
        rows = rows[:cfg["max_train_clips"]]
    if not rows:
        raise ValueError("No cached training clips")
    dataset = CachedClipDataset(rows, cfg["num_frames"], cfg["latents_name"])
    if any(cfg.get(k, 0) for k in ['pixel_weight','lpips_weight','sync_weight','keep_weight',
                                   'art_weight']):
        dataset = PixelClipDataset(dataset, full_context_runtime.CLIPS)
    return dataset, rows


class PixelClipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root):
        self.dataset, self.root = dataset, Path(root)
    def __len__(self):return len(self.dataset)
    def __getitem__(self, index):
        import decord
        value = self.dataset[index]
        path = self.root / self.dataset.rows[index]['dir'] / 'video.mp4'
        reader = decord.VideoReader(str(path), num_threads=1)
        value['gt_frames'] = torch.from_numpy(reader.get_batch(list(range(self.dataset.num_frames))).asnumpy())
        return value


def first_item(batch):
    return batch[0]


def prepare_clip(clip, pipe, whisper, zero_latents, cfg):
    local_imports()
    from viddub.full_context.runtime import build_inputs
    x0, vc, ctx = build_inputs(clip["latents"], clip["mask"], cfg["num_frames"], zero_latents,
                               drop_ref=cfg["num_ref"] == 0)
    if x0.shape[2] != (cfg["num_frames"] - 1) // 4 + 1 + cfg["num_ref"]:
        raise ValueError("Cached clip is shorter than configured training length")
    hs = whisper([clip["audio"]], pipe.device, torch.bfloat16)
    # Whisper remains frozen BF16; transformer input follows compute precision.
    dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32
    conditions = {"context": ctx.to(dtype), "vace_context": vc.to(dtype),
                  "whisper_hs": hs.to(dtype), "num_frames": cfg["num_frames"], 'num_ref':cfg['num_ref']}
    if 'gt_frames' in clip:
        from viddub.full_context.sync_loss import melspectrogram
        conditions.update(gt_frames=clip['gt_frames'], pixel_mask=torch.from_numpy(clip['mask']),
                          mel=melspectrogram(clip['audio']))
    return x0.to(dtype), conditions
