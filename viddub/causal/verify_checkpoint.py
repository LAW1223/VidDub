"""Verify real Wan-Dub weights before spending GPU time on causal training.

Checks original bidirectional parity, future-video isolation, cache/no-cache
parity, and one AR backward pass. Does not step an optimizer or save weights.
"""
import argparse
from pathlib import Path
import time

import torch

from .runtime import (atomic_json, autocast, checkpoint_paths, load_components,
                      local_imports, make_dataset, prepare_clip, read_config)
from .model import HistoryCache
from .training.stages import ARObjective
from .sampling import commit, predict_chunk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", required=True, help="New JSON report path")
    p.add_argument("--clip-index", type=int, default=0)
    a = p.parse_args()
    cfg = read_config(a.config)
    paths = checkpoint_paths(a.checkpoint_dir, a.step)
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Real checkpoint verification requires CUDA")
    local_imports()
    from viddub.full_context.runtime import ZeroLatents, predict
    data, _ = make_dataset(cfg)
    device = torch.device("cuda")
    pipe, whisper, model = load_components(paths, cfg, device)
    model.eval().requires_grad_(False)
    x0, c = prepare_clip(data[a.clip_index], pipe, whisper, ZeroLatents(pipe), cfg)
    started = time.monotonic()
    # Native reference uses the project's own adapter ordering and attention backend.
    torch.manual_seed(cfg["seed"])
    xt = torch.randn_like(x0)
    tol = 0.04 if cfg["precision"] == "bf16" else 0.0003
    report = {"status": "running", "precision": cfg["precision"], "clip_id": data[a.clip_index]["clip_id"],
              "tolerance": {"atol": tol, "rtol": tol}, "checks": {}}
    try:
        with torch.no_grad(), autocast(device, cfg["precision"]):
            tok = model.adapter.tokens(c["whisper_hs"], cfg["num_frames"], cfg["num_ref"],
                                       torch.tensor([False], device=device))
            native = predict(pipe, model.adapter, xt, xt.new_tensor([500]), c["context"], c["vace_context"], tok)
            model.causal = False
            actual = model(xt, 500, c["context"], c["vace_context"], c["whisper_hs"], c["num_frames"])
            report["checks"]["bidirectional_max_abs"] = float((actual - native).abs().max())
            torch.testing.assert_close(actual, native, atol=tol, rtol=tol)
            model.causal = True
            full = model(xt, 500, c["context"], c["vace_context"], c["whisper_hs"], c["num_frames"])
            end = model.config.video_blocks(xt.shape[2])[0][1]
            changed, vc = xt.clone(), c["vace_context"].clone()
            changed[:, :, end:] += 20
            vc[:, :, end:] -= 20
            future = model(changed, 500, c["context"], vc, c["whisper_hs"], c["num_frames"])
            report["checks"]["future_max_abs"] = float((full[:, :, :end] - future[:, :, :end]).abs().max())
            torch.testing.assert_close(full[:, :, :end], future[:, :, :end], atol=tol, rtol=tol)
            cache = HistoryCache()
            n_ref = cfg["num_ref"]
            if n_ref:
                commit(model, x0[:, :, :n_ref], c, 0, cache)
            errors = []
            for start, end in model.config.video_blocks(x0.shape[2]):
                expected = predict_chunk(model, xt[:, :, start:end], 0.5, x0[:, :, :start], c, start)
                actual = predict_chunk(model, xt[:, :, start:end], 0.5, x0[:, :, :start], c, start, cache)
                errors.append(float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                commit(model, x0[:, :, start:end], c, start, cache)
            report["checks"]["cache_max_abs"] = max(errors)
        del cache, native, actual, full, changed, vc, future
        model.requires_grad_(True).train()
        with autocast(device, cfg["precision"]):
            loss, _ = ARObjective(model, shift=cfg["shift"])(x0, c, chunk_index=0, sigma=0.5)
        loss.backward()
        for name in ("dit", "vace", "adapter"):
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise AssertionError(f"Missing/nonfinite {name} gradients")
            report["checks"][name + "_grad_abs_sum"] = sum(float(g.abs().sum()) for g in grads)
        report["checks"]["ar_loss"] = float(loss.detach())
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(a.output, report)


if __name__ == "__main__":
    main()
