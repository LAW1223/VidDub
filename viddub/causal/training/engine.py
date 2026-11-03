"""Shared implementation behind the AR, ODE and DMD training entry points."""
import argparse
import copy
import datetime
import hashlib
import json
import random
import math
import time
from pathlib import Path

import torch
from torch.optim import _functional as optim_functional

from .stages import ARObjective, DMDObjective, FakeObjective, ODEObjective
from ..runtime import (atomic_json, autocast, checkpoint_paths, export_weights, fingerprint,
                      first_item, load_components, local_imports, make_dataset, prepare_clip,
                      read_config, require_causal_metadata)
from ..sampling import rollout, schedule


def optimizer(model, cfg):
    return torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": cfg["lr"]},
        {"params": list(model.dit.parameters()) + list(model.vace.parameters()), "lr": cfg["lr_base"]},
    ], weight_decay=cfg["weight_decay"], betas=(cfg.get('beta1',.9),cfg.get('beta2',.999)), foreach=False)


def lr_scheduler(opt, cfg, updates_per_step=1):
    """Schedule measured in generator updates, including repeated critic steps."""
    if updates_per_step < 1:
        raise ValueError("updates_per_step must be positive")
    def scale(step):
        step = step / updates_per_step
        warm = min(1.0, (step + 1) / max(1, cfg["warmup"]))
        if cfg.get('lr_schedule') == 'cosine' and step >= cfg['warmup']:
            progress = min(1., (step-cfg['warmup'])/max(1,cfg['max_steps']-cfg['warmup']))
            return cfg['lr_min_ratio']+(1-cfg['lr_min_ratio'])*0.5*(1+math.cos(math.pi*progress))
        return warm
    return torch.optim.lr_scheduler.LambdaLR(opt, scale)


def set_trainable(model, enabled):
    model.requires_grad_(enabled)
    if not enabled:
        for p in model.parameters():
            p.grad = None


def move_optimizer_state(opt, device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_optimizer_state_for_step(opt, device):
    """Restore offloaded Adam state after backward without retaining cache fragments.

    Production clips can leave hundreds of MiB in non-contiguous cached blocks after
    fake-score backward.  The Adam tensors need contiguous allocations and may OOM
    despite that memory being reported as reserved-but-free.  Releasing only the
    allocator cache here does not modify live tensors, gradients, optimizer values,
    RNG, or update order; it merely makes already-free blocks available to the CUDA
    allocator before the state transfer.
    """
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    move_optimizer_state(opt, device)


@torch.no_grad()
def streaming_offloaded_adamw_step(opt, device, chunk_elements=262144):
    """Apply the native AdamW update one parameter at a time.

    The optimizer state remains on CPU between parameters.  For each parameter we
    move bounded chunks of its first/second moments to the parameter device, call
    PyTorch's own Adam functional with the original parameter-group flags, then
    immediately move those chunks back to CPU. Every chunk sees the same old step;
    the persistent step counter advances once per parameter update. This preserves
    ordinary AdamW state_dict semantics without materializing a full parameter's,
    let alone the full model's, moments on CUDA.
    """
    base = getattr(opt, "optimizer", opt)
    if not isinstance(base, torch.optim.AdamW):
        raise TypeError("Streaming offload currently requires torch.optim.AdamW")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    for group in base.param_groups:
        if group.get("foreach") not in (False, None) or group.get("fused"):
            raise ValueError("Streaming offload requires foreach=False and fused=False")
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients")
            state = base.state[parameter]
            if not state:
                # Match Adam._init_group, except moments are born on CPU because
                # this optimizer is explicitly offloaded between updates.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    parameter, device="cpu", memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter, device="cpu", memory_format=torch.preserve_format)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        parameter, device="cpu", memory_format=torch.preserve_format)
            if not parameter.is_contiguous() or not gradient.is_contiguous():
                raise ValueError("Streaming AdamW requires contiguous parameters and gradients")
            parameter_flat = parameter.view(-1)
            gradient_flat = gradient.view(-1)
            exp_avg_cpu = state["exp_avg"].view(-1)
            exp_avg_sq_cpu = state["exp_avg_sq"].view(-1)
            max_cpu = state.get("max_exp_avg_sq")
            max_cpu = max_cpu.view(-1) if max_cpu is not None else None
            old_step = state["step"].clone()
            next_step = None
            for start in range(0, parameter.numel(), chunk_elements):
                end = min(parameter.numel(), start + chunk_elements)
                exp_avg = exp_avg_cpu[start:end].to(parameter.device)
                exp_avg_sq = exp_avg_sq_cpu[start:end].to(parameter.device)
                max_exp_avg_sq = (max_cpu[start:end].to(parameter.device)
                                  if max_cpu is not None else None)
                local_step = old_step.clone()
                optim_functional.adam(
                    [parameter_flat[start:end]], [gradient_flat[start:end]],
                    [exp_avg], [exp_avg_sq],
                    [max_exp_avg_sq] if max_exp_avg_sq is not None else [],
                    [local_step], foreach=False,
                    capturable=group["capturable"],
                    differentiable=group["differentiable"], fused=False,
                    grad_scale=getattr(base, "grad_scale", None),
                    found_inf=getattr(base, "found_inf", None),
                    has_complex=torch.is_complex(parameter),
                    decoupled_weight_decay=True, amsgrad=group["amsgrad"],
                    beta1=beta1, beta2=beta2, lr=group["lr"],
                    weight_decay=group["weight_decay"], eps=group["eps"],
                    maximize=group["maximize"])
                exp_avg_cpu[start:end].copy_(exp_avg.cpu())
                exp_avg_sq_cpu[start:end].copy_(exp_avg_sq.cpu())
                if max_exp_avg_sq is not None:
                    max_cpu[start:end].copy_(max_exp_avg_sq.cpu())
                next_step = local_step
            if next_step is not None:
                state["step"].copy_(next_step)


def finite_streaming_offloaded_update(accel, objective, opt, cfg):
    norm = accel.clip_grad_norm_(objective.parameters(), cfg["grad_clip"])
    valid = accel.reduce(torch.tensor(int(torch.isfinite(norm)), device=accel.device), "sum")
    if int(valid) != accel.num_processes:
        opt.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradients; stopped without advancing checkpoint")
    streaming_offloaded_adamw_step(opt, accel.device)
    opt.zero_grad(set_to_none=True)
    return float(norm)


def finite_update(accel, objective, opt, cfg):
    norm = accel.clip_grad_norm_(objective.parameters(), cfg["grad_clip"])
    valid = accel.reduce(torch.tensor(int(torch.isfinite(norm)), device=accel.device), "sum")
    if int(valid) != accel.num_processes:
        opt.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradients; stopped without advancing checkpoint")
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(norm)


def check_loss(accel, loss):
    finite = accel.reduce(torch.tensor(int(torch.isfinite(loss)), device=accel.device), "sum")
    if int(finite) != accel.num_processes:
        raise FloatingPointError("Non-finite loss on at least one rank")


def main(expected_stage=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(teacher_dir=None, teacher_step=None, fake_dir=None,
                   fake_step=None, trajectories=None)
    p.add_argument("--config", required=True)
    p.add_argument("--init-dir", required=True)
    p.add_argument("--init-step", required=True, type=int)
    p.add_argument("--output", required=True)
    if expected_stage in (None, "ode"):
        p.add_argument(
            "--trajectories",
            help="Trajectory directory produced by viddub.causal.make_trajectories",
        )
    if expected_stage in (None, "dmd"):
        p.add_argument("--teacher-dir", help="Original bidirectional VidDub checkpoint")
        p.add_argument("--teacher-step", type=int)
        p.add_argument("--fake-dir", help="Pretrained fake-score checkpoint")
        p.add_argument("--fake-step", type=int)
    p.add_argument("--resume", help="A complete state_stepNNNNNN directory")
    p.add_argument("--stop-after-step", type=int,
                   help="Operational stop after a saved step; does not alter the training config")
    p.add_argument("--check-only", action="store_true", help="Validate inputs, metadata and data; do not allocate GPUs")
    a = p.parse_args()
    cfg = read_config(a.config)
    if expected_stage is not None and cfg["stage"] != expected_stage:
        p.error(f"This entry point requires stage={expected_stage}, got {cfg['stage']}")
    if cfg['stage'] == 'cd':
        p.error('Use python -m viddub.causal.cf.train for consistency training')
    if a.stop_after_step is not None and not 0 < a.stop_after_step <= cfg['max_steps']:
        p.error('--stop-after-step must lie within 1..max_steps')
    run_until = cfg['max_steps'] if a.stop_after_step is None else a.stop_after_step
    paths = checkpoint_paths(a.init_dir, a.init_step)
    teacher_paths = None
    run_provenance = {"student_init": fingerprint(paths)}
    if cfg["stage"] == "ode":
        if cfg["causal"]:
            require_causal_metadata(a.init_dir, a.init_step, cfg, ("ar",))
        if not a.trajectories:
            p.error("ODE requires --trajectories")
    if cfg["stage"] == "dmd":
        require_causal_metadata(a.init_dir, a.init_step, cfg, ("ode", "dmd", "cd"))
        teacher_paths = checkpoint_paths(a.teacher_dir, a.teacher_step)
        run_provenance["teacher"] = fingerprint(teacher_paths)
        if bool(a.fake_dir) != (a.fake_step is not None):
            p.error("Provide both --fake-dir and --fake-step")
        fake_paths = checkpoint_paths(a.fake_dir, a.fake_step) if a.fake_dir else None
        if fake_paths is not None:
            run_provenance['fake_init'] = fingerprint(fake_paths)
    if a.resume:
        if Path(a.resume).resolve().parent != Path(a.output).resolve():
            raise ValueError("Resume state must belong to this output directory")
        resume_meta = json.loads((Path(a.resume) / "complete.json").read_text())
        if read_config(resume_meta["config"]) != cfg:
            raise ValueError("Resume requires exactly the saved config")
    else:
        resume_meta = None
    if not a.resume and Path(a.output).exists() and any(Path(a.output).iterdir()):
        raise ValueError("Output is not empty; choose a new output or explicitly --resume")
    if cfg["stage"] == "ode":
        from ..trajectories import TrajectoryDataset
        dataset = TrajectoryDataset(a.trajectories, cfg)
        trajectory_causal = dataset.manifest.get(
            "teacher_architecture", {}
        ).get("causal", dataset.manifest.get("config", {}).get("causal", True))
        if trajectory_causal != cfg["causal"]:
            raise ValueError("Trajectory/student causal architecture mismatch")
        for name in paths:
            if dataset.manifest["teacher"][name]["sha256"] != run_provenance["student_init"][name]["sha256"]:
                raise ValueError(f"Trajectory teacher differs from student initialization: {name}")
        run_provenance["trajectories"] = str(Path(a.trajectories).resolve())
        run_provenance["trajectory_manifest"] = fingerprint({"manifest": dataset.manifest_path})["manifest"]
        rows = None
    else:
        dataset, rows = make_dataset(cfg)
    if cfg["stage"] == "dmd" and cfg["p_cross"] > 0 and len(dataset) < 2:
        raise ValueError("Cross-audio DMD requires at least two training clips")
    if rows is not None:
        row_json = json.dumps(rows, sort_keys=True).encode()
        run_provenance["dataset_rows_sha256"] = hashlib.sha256(row_json).hexdigest()
    if resume_meta and resume_meta["provenance"] != run_provenance:
        raise ValueError("Resume checkpoint/data provenance differs")
    if a.check_only:
        print(json.dumps({"config": cfg, "checkpoint_files": {k: str(v) for k, v in paths.items()},
                          "samples": len(dataset), "status": "inputs_validated_no_gpu_execution"}, indent=2))
        return
    local_imports()
    import accelerate
    from safetensors.torch import load_file
    from viddub.full_context.runtime import ZeroLatents
    accel = accelerate.Accelerator(
        mixed_precision="no", gradient_accumulation_steps=cfg["grad_accum"],
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False, gradient_as_bucket_view=True),
                         accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))])
    if accel.device.type != "cuda":
        raise RuntimeError(
            "Production training needs CUDA; run `python -m unittest discover -s tests` "
            "for CPU-only contract verification"
        )
    if "DEEPSPEED" in str(accel.distributed_type) or "FSDP" in str(accel.distributed_type):
        raise ValueError("This first implementation supports plain single GPU/DDP only")
    random.seed(cfg["seed"] + accel.process_index)
    torch.manual_seed(cfg["seed"] + accel.process_index)
    pipe, whisper, student = load_components(paths, cfg, accel.device)
    student.requires_grad_(True).train()
    teacher = fake = anchor = fake_objective = fake_opt = fake_sched = None
    if cfg["stage"] == "dmd":
        if cfg['dmd_retention_weight']:
            anchor = copy.deepcopy(student).requires_grad_(False).eval()
        teacher = copy.deepcopy(student)
        teacher.causal = False
        for name, path in teacher_paths.items():
            getattr(teacher, name).load_state_dict(load_file(str(path)), strict=True)
        teacher.requires_grad_(False).eval()
        fake = copy.deepcopy(teacher)
        if fake_paths is not None:
            for name, path in fake_paths.items():
                getattr(fake, name).load_state_dict(load_file(str(path)), strict=True)
        fake.requires_grad_(True).train()
    kwargs = dict(mouth_weight=cfg["mouth_weight"], shift=cfg["shift"], student_steps=cfg["student_steps"],
                  teacher_cfg=cfg["teacher_cfg"], use_cache=cfg["use_cache"],
                  gradient_checkpointing=cfg["gradient_checkpointing"],
                  matching_min=cfg["matching_min"], matching_max=cfg["matching_max"])
    kwargs['dmd_rollout_mode'] = cfg['dmd_rollout_mode']
    kwargs['ode_gt_flow_weight'] = cfg['ode_gt_flow_weight']
    kwargs['rollout_quality_weight'] = cfg['rollout_quality_weight']
    kwargs['rollout_quality_cross'] = cfg['rollout_quality_cross']
    kwargs['dmd_retention_weight'] = cfg['dmd_retention_weight']
    kwargs['dmd_score_offload'] = cfg['dmd_score_offload']
    for key in ['ar_history_mode','ar_history_prob_start','ar_history_prob_end',
                'ar_history_ramp_steps','ar_history_steps','ar_history_sigma_max',
                'ar_history_mouth_drop_prob','ar_history_mouth_drop_cross_only']:
        kwargs[key]=cfg[key]
    quality = None
    if any(cfg.get(k,0) for k in ['pixel_weight','lpips_weight','sync_weight','keep_weight','art_weight']):
        from ..quality import GTQuality
        quality = GTQuality(pipe, cfg, accel.device)
    art = None
    if cfg.get('art_weight',0):
        if cfg['stage']!='ar':raise ValueError('Art integration is initially AR-only')
        from ..art_alignment import CausalArtAlignment
        art=CausalArtAlignment(student.dit.dim).to(accel.device)
    objective_class={"ar":ARObjective,"ode":ODEObjective,"dmd":DMDObjective}[cfg["stage"]]
    if cfg['stage']=='dmd' and cfg['sampler']=='cf_renoise':
        from ..cf.dmd import CFAsymmetricDMD
        objective_class=CFAsymmetricDMD
    objective = objective_class(student, teacher=teacher, fake=fake, anchor=anchor,
                          quality=quality,
                          art=art,art_weight=cfg['art_weight'],art_layer=cfg['art_layer'],
                          art_warmup=cfg['art_warmup'],**kwargs)
    opt = optimizer(student, cfg)
    if art is not None:opt.add_param_group({'params':art.parameters(),'lr':cfg['art_lr']})
    sched = lr_scheduler(opt, cfg)
    loader_kwargs = dict(batch_size=1, shuffle=True, collate_fn=first_item, num_workers=cfg["workers"])
    loader_kwargs["generator"] = torch.Generator().manual_seed(cfg["seed"])
    if cfg["workers"]:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    objective, opt, loader = accel.prepare(objective, opt, loader)
    accel.register_for_checkpointing(sched)
    if fake is not None:
        # The training loop owns fake residency through forward+backward.
        # Self-offloading inside Objective.fake would move checkpointed weights
        # to CPU before backward recomputation.
        fake_kwargs = dict(kwargs, dmd_score_offload=False)
        fake_objective = FakeObjective(fake, **fake_kwargs)
        fake_opt = optimizer(fake, dict(cfg, lr=cfg['fake_lr'], lr_base=cfg['fake_lr_base']))
        fake_sched = lr_scheduler(fake_opt, cfg, updates_per_step=cfg["fake_updates"])
        fake_objective, fake_opt = accel.prepare(fake_objective, fake_opt)
        accel.register_for_checkpointing(fake_sched)
        if cfg['dmd_score_offload']:
            teacher.cpu()
            if anchor is not None:anchor.cpu()
            fake.cpu()
            torch.cuda.empty_cache()
    provenance = run_provenance
    out = Path(a.output)
    if accel.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "run.json", {"config": cfg, "provenance": provenance,
                                      "world_size": accel.num_processes,
                                      "note": "DDP shards samples; all models are replicated per GPU."})
    accel.wait_for_everyone()
    provenance = json.loads((out / "run.json").read_text())["provenance"]
    step, epoch, consumed = 0, 0, 0
    if a.resume:
        if resume_meta["world_size"] != accel.num_processes:
            raise ValueError("Resume requires the same world size")
        accel.load_state(a.resume)
        if fake is not None and cfg['dmd_score_offload']:
            move_optimizer_state(fake_opt, 'cpu')
            fake.cpu()
            teacher.cpu()
            if anchor is not None:anchor.cpu()
            torch.cuda.empty_cache()
        step, epoch, consumed = (resume_meta[k] for k in ("step", "epoch", "consumed_batches"))
    zero_latents = ZeroLatents(pipe)
    # Auxiliary network initialization must not change the data/noise stream.
    if not a.resume:
        random.seed(cfg['seed']+accel.process_index)
        torch.manual_seed(cfg['seed']+accel.process_index)
    start_time = time.monotonic()
    row_index = {r["clip_id"]: i for i, r in enumerate(rows)} if rows is not None else {}
    metric_sums = {}
    metric_count = 0

    def save(epoch, consumed):
        state = out / f"state_step{step:06d}"
        # All ranks must finish the existence check BEFORE any rank calls
        # save_state (which creates the directory). Otherwise a faster rank
        # makes a slower rank mistake this very save for an overwrite.
        exists = accel.reduce(torch.tensor(int(state.exists()), device=accel.device), "sum")
        if int(exists):
            raise FileExistsError(f"Refusing to overwrite state {state}")
        full_state = not cfg.get('save_state_every') or step % cfg['save_state_every'] == 0 or step == cfg['max_steps']
        if full_state:accel.save_state(str(state))
        if accel.is_main_process:
            export_weights(student, out, step, cfg, provenance)
            if full_state:atomic_json(state / "complete.json", {"step": step, "epoch": epoch,
                        "consumed_batches": consumed, "config": cfg, "provenance": provenance,
                        "world_size": accel.num_processes})
        accel.wait_for_everyone()

    while step < run_until:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        # Seed permutation by epoch, independent of model/Whisper RNG use.
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(cfg["seed"] + epoch)
        active_loader = accel.skip_first_batches(loader, consumed) if consumed else loader
        for item in active_loader:
            consumed += 1
            record = None
            if cfg["stage"] == "ode":
                dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32
                x0 = item["shape_reference"].to(accel.device, dtype)
                c = {}
                for k, v in item["conditions"].items():
                    if torch.is_tensor(v):
                        c[k] = v.to(accel.device, dtype) if k in (
                            'context', 'vace_context', 'whisper_hs') else v
                    else:
                        c[k] = v
                record = item["record"]
            else:
                if cfg["stage"] in ("ar", "dmd") and random.random() < cfg["p_cross"]:
                    # Conditions stay on the source video; swap only audio.
                    other = dataset[random.randrange(len(dataset))]
                    if other["clip_id"] == item["clip_id"]:
                        source_index = row_index[item["clip_id"]]
                        other = dataset[(source_index + 1) % len(dataset)]
                    item = dict(item, audio=other["audio"], cross=True)
                x0, c = prepare_clip(item, pipe, whisper, zero_latents, cfg)
                c['cross'] = bool(item.get('cross', False))
            if cfg["stage"] == "ar":
                c['train_step']=step
                c['history_seed']=cfg['seed']+consumed*100003+accel.process_index
                c["drop_audio"] = torch.rand(x0.shape[0], device=x0.device) < cfg["p_drop_audio"]
                if c['cross']:
                    c["drop_audio"].zero_()
            if fake is not None:
                set_trainable(student, False)
                if cfg['dmd_score_offload']:
                    fake.to(accel.device)
                set_trainable(fake, True)
                fake.train()
                for _ in range(cfg["fake_updates"]):
                    # Keep the (large) Adam state on CPU while rolling out the
                    # student sample. After the first fake update those states
                    # otherwise overlap the causal rollout peak by only tens of
                    # MiB on 96-GB cards. This changes only residency timing:
                    # fake weights, samples, RNG, update count and objectives are
                    # unchanged.
                    with torch.no_grad(), autocast(accel.device, cfg["precision"]):
                        if cfg['sampler']=='cf_renoise':
                            from ..cf.dmd import shared_exit
                            from ..cf.sampling import cf_rollout
                            generated=cf_rollout(student,x0,c,steps=cfg['student_steps'],shift=cfg['shift'],
                                use_cache=cfg['use_cache'],grad_exit=(None if cfg['dmd_rollout_mode']=='final_step'
                                else shared_exit(cfg['student_steps'],x0.device)))
                        else:
                            generated = rollout(student, x0, c, schedule(cfg["student_steps"], cfg["shift"], x0.device),
                                                use_cache=cfg["use_cache"])
                    with autocast(accel.device, cfg["precision"]):
                        fake_loss, _ = fake_objective(x0, c, generated=generated)
                    check_loss(accel, fake_loss)
                    accel.backward(fake_loss)
                    if cfg['dmd_score_offload']:
                        finite_streaming_offloaded_update(
                            accel, fake_objective, fake_opt, cfg)
                    else:
                        finite_update(accel, fake_objective, fake_opt, cfg)
                    fake_sched.step()
                    del generated
                set_trainable(fake, False)
                fake.eval()
                if cfg['dmd_score_offload']:
                    fake.cpu()
                    torch.cuda.empty_cache()
                set_trainable(student, True)
            with accel.accumulate(objective):
                if cfg['stage'] == 'ar' and cfg['ar_blocks_per_sample'] != 1:
                    ranges = student.config.video_blocks(x0.shape[2])
                    count = len(ranges) if cfg['ar_blocks_per_sample'] == 0 else cfg['ar_blocks_per_sample']
                    if count > len(ranges):
                        raise ValueError(f'ar_blocks_per_sample={count} exceeds {len(ranges)} video blocks')
                    # Without replacement: every selected block provides a distinct
                    # temporal/audio location. Zero means the official-style all-block path.
                    indices = (list(range(len(ranges))) if count == len(ranges)
                               else torch.randperm(len(ranges))[:count].tolist())
                    metrics = {}
                    loss_value = x0.new_zeros((), dtype=torch.float32)
                    for chunk_index in indices:
                        with autocast(accel.device, cfg["precision"]):
                            chunk_loss, chunk_metrics = objective(x0, c, chunk_index=chunk_index)
                        check_loss(accel, chunk_loss)
                        accel.backward(chunk_loss / count)
                        loss_value = loss_value + chunk_loss.detach().float() / count
                        for key, value in chunk_metrics.items():
                            value = torch.as_tensor(value, device=x0.device, dtype=torch.float32).detach()
                            metrics[key] = metrics.get(key, 0.0) + value / count
                    loss = loss_value
                    metrics['blocks_per_sample'] = x0.new_tensor(float(count))
                    metrics['block_coverage_fraction'] = x0.new_tensor(float(count / len(ranges)))
                else:
                    split_quality = (cfg['stage'] == 'dmd' and cfg['dmd_split_quality_backward'] and
                                     any(cfg[k] for k in ['pixel_weight','lpips_weight','sync_weight','keep_weight']))
                    if split_quality:
                        cpu_rng = torch.random.get_rng_state()
                        cuda_rng = torch.cuda.get_rng_state(accel.device)
                        with autocast(accel.device, cfg["precision"]):
                            primary, metrics = objective(x0, c, record=record, dmd_mode='primary')
                        post_cpu_rng = torch.random.get_rng_state()
                        post_cuda_rng = torch.cuda.get_rng_state(accel.device)
                        check_loss(accel, primary)
                        accel.backward(primary)
                        torch.random.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state(cuda_rng, accel.device)
                        with autocast(accel.device, cfg["precision"]):
                            auxiliary, aux_metrics = objective(x0, c, record=record, dmd_mode='quality')
                        check_loss(accel, auxiliary)
                        accel.backward(auxiliary)
                        torch.random.set_rng_state(post_cpu_rng)
                        torch.cuda.set_rng_state(post_cuda_rng, accel.device)
                        loss = primary.detach().float() + auxiliary.detach().float()
                        metrics.update(aux_metrics)
                        metrics['split_quality_backward'] = x0.new_tensor(1.)
                    else:
                        with autocast(accel.device, cfg["precision"]):
                            loss, metrics = objective(x0, c, record=record)
                        check_loss(accel, loss)
                        accel.backward(loss)
                if accel.sync_gradients:
                    norm = finite_update(accel, objective, opt, cfg)
                    sched.step()
            metric_count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            if not accel.sync_gradients:
                continue
            step += 1
            if step % cfg["log_every"] == 0:
                value = float(accel.reduce(loss.detach(), "mean"))
                count = accel.reduce(torch.tensor(float(metric_count), device=accel.device), "sum")
                averaged = {}
                for key, total in metric_sums.items():
                    reduced = accel.reduce(torch.tensor(total, device=accel.device), "sum")
                    averaged[key] = float(reduced / count)
                log = {"step": step, "loss": value, "grad_norm": norm,
                       "elapsed_seconds": time.monotonic() - start_time,
                       "metric_window_steps": cfg["log_every"], **averaged}
                if fake is not None:
                    log["fake_loss"] = float(fake_loss.detach())
                accel.print(json.dumps(log), flush=True)
                if accel.is_main_process:
                    with (out / "train.jsonl").open("a") as f:
                        f.write(json.dumps(log) + "\n")
                metric_sums.clear();metric_count=0
            if step % cfg["save_every"] == 0 or step in cfg.get('save_steps',[]) or step == cfg["max_steps"]:
                save(epoch, consumed)
            if step == run_until:
                break
        epoch, consumed = epoch + 1, 0
    accel.end_training()


if __name__ == "__main__":
    main()
