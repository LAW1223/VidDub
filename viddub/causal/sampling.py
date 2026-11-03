"""Flow schedules and clean-history autoregressive rollout.

This is a correctness reference: no cache eviction and no streaming VAE decoder.
All denoising passes READ history. One extra t=0 pass commits each final chunk.
"""
from contextlib import contextmanager

import torch

from .model import HistoryCache


def schedule(steps, shift=5.0, device="cpu"):
    if steps < 1 or shift <= 0:
        raise ValueError("steps and shift must be positive")
    s = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return shift * s / (1 + (shift - 1) * s)


def validate_schedule(sigmas):
    if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
        raise ValueError("Expected finite 1D schedule")
    if abs(float(sigmas[0]) - 1) > 1e-6 or float(sigmas[-1]) != 0:
        raise ValueError("Schedule must go from pure noise (1) to clean (0)")
    if not torch.all(sigmas[:-1] > sigmas[1:]):
        raise ValueError("Schedule must be strictly decreasing")


@contextmanager
def evaluating(model):
    old = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(old)


def predict_chunk(model, chunk, sigma, prefix, conditions, start, cache=None,
                  drop_audio=None, gradient_checkpointing=False, detach_history=False,
                  return_hidden_layer=None):
    """Cache-free and cached implementations have exactly the same semantics."""
    if cache is not None:
        vc_start = int(conditions.get("vace_start", 0))
        local_start = start-vc_start
        if local_start < 0 or local_start+chunk.shape[2] > conditions["vace_context"].shape[2]:
            raise ValueError("Current VACE condition does not cover the requested absolute block")
        x = chunk
        vc = conditions["vace_context"][:, :, local_start:local_start + chunk.shape[2]]
        offset = start
        ts = chunk.new_full((chunk.shape[0], chunk.shape[2]), float(sigma) * 1000)
    else:
        x = torch.cat([prefix, chunk], 2)
        vc, offset = conditions["vace_context"][:, :, :x.shape[2]], 0
        ts = x.new_zeros((x.shape[0], x.shape[2]))
        ts[:, start:] = float(sigma) * 1000
    out = model(x, ts, conditions["context"], vc, conditions["whisper_hs"],
                conditions["num_frames"], drop_audio, start_frame=offset, cache=cache,
                gradient_checkpointing=gradient_checkpointing,
                detach_history_before=start if detach_history and cache is None else 0,
                return_hidden_layer=return_hidden_layer,
                audio_tokens=conditions.get("audio_tokens"))
    if return_hidden_layer is not None:
        prediction, hidden = out
        return (prediction if cache is not None else prediction[:, :, start:]), hidden
    return out if cache is not None else out[:, :, start:]


def commit(model, chunk, conditions, start, cache, drop_audio=None):
    if cache is None:
        return
    vc_start = int(conditions.get("vace_start", 0))
    local_start = start-vc_start
    if local_start < 0 or local_start+chunk.shape[2] > conditions["vace_context"].shape[2]:
        raise ValueError("Current VACE condition does not cover the committed absolute block")
    with torch.no_grad():
        model(chunk.detach(), 0.0, conditions["context"],
              conditions["vace_context"][:, :, local_start:local_start + chunk.shape[2]],
              conditions["whisper_hs"], conditions["num_frames"], drop_audio,
              start_frame=start, cache=cache, commit_cache=True,
              audio_tokens=conditions.get("audio_tokens"))


def history_chunk(chunk, conditions, start, mode):
    """Inference-only intervention on what is committed as clean visual history.

    The returned tensor affects future chunks only; the visible output for the
    current chunk remains unchanged. VACE channels 0:C contain the inactive
    source latent and channels 32: encode the editable-region mask.
    """
    if mode == "full":
        return chunk.detach()
    vc_start = int(conditions.get("vace_start", 0))
    local_start = start-vc_start
    if local_start < 0 or local_start+chunk.shape[2] > conditions["vace_context"].shape[2]:
        raise ValueError("Current VACE condition does not cover the history block")
    inactive = conditions["vace_context"][:, :chunk.shape[1], local_start:local_start + chunk.shape[2]].to(chunk)
    if mode == "inactive":
        return inactive.detach()
    if mode == "mouth_suppressed":
        mask = conditions["vace_context"][:, 32:, local_start:local_start + chunk.shape[2]].amax(
            1, keepdim=True).float().clamp(0, 1).to(chunk)
        return (chunk.detach() * (1 - mask) + inactive * mask).detach()
    raise ValueError(f"Unknown history mode: {mode}")


def rollout(model, shape_reference, conditions, sigmas, *, noise=None,
            use_cache=True, audio_cfg=1.0, grad_chunk=None, grad_step=None,
            gradient_checkpointing=False, trajectory=False, history_mode="full",
            on_chunk=None, max_video_history_frames=None):
    """Generate all video chunks, retaining gradients for at most one chunk/step.

    Normal inference: grad_chunk=None. DMD: randomly choose a video chunk and
    an exit step; use its x0 estimate and detach every other rollout operation.
    Later chunks condition on that estimate, but gradients do not pass through
    cache/history. This deliberately implements truncated rollout gradients.
    shape_reference video values are NEVER read; only the clean reference prefix
    (if configured) is consumed. noise has the full [B,C,F,H,W] shape.
    """
    validate_schedule(sigmas)
    if audio_cfg < 0:
        raise ValueError("audio_cfg must be nonnegative")
    if history_mode not in ("full", "mouth_suppressed", "inactive"):
        raise ValueError("Unsupported history mode")
    n_ref = model.config.num_ref
    ranges = model.config.video_blocks(shape_reference.shape[2])
    if grad_chunk is not None and not 0 <= grad_chunk < len(ranges):
        raise ValueError("grad_chunk is outside video chunks")
    if grad_step is not None and not 0 <= grad_step < len(sigmas) - 1:
        raise ValueError("grad_step outside schedule")
    noise = torch.randn_like(shape_reference) if noise is None else noise
    if noise.shape != shape_reference.shape:
        raise ValueError("noise shape mismatch")
    cache = HistoryCache(max_video_history_frames=max_video_history_frames) if use_cache else None
    null_cache = HistoryCache(max_video_history_frames=max_video_history_frames) if use_cache and audio_cfg != 1 else None
    null = torch.ones(shape_reference.shape[0], device=shape_reference.device, dtype=torch.bool)
    prefix = shape_reference[:, :, :n_ref].detach()
    chunks, records = [prefix], []
    with evaluating(model):
        if n_ref:
            commit(model, prefix, conditions, 0, cache)
            if null_cache is not None:
                commit(model, prefix, conditions, 0, null_cache, null)
        for index, (start, end) in enumerate(ranges):
            current = noise[:, :, start:end]
            states = [current.detach().cpu()] if trajectory else None
            stop = grad_step if index == grad_chunk and grad_step is not None else len(sigmas) - 2
            for step in range(stop + 1):
                keep_grad = index == grad_chunk and step == stop
                with torch.set_grad_enabled(keep_grad):
                    pred = predict_chunk(model, current, sigmas[step], prefix.detach(), conditions,
                                         start, cache, gradient_checkpointing=gradient_checkpointing,
                                         detach_history=True)
                    if audio_cfg != 1:
                        pred_null = predict_chunk(model, current, sigmas[step], prefix.detach(), conditions,
                                                  start, null_cache, null,
                                                  gradient_checkpointing=gradient_checkpointing,
                                                  detach_history=True)
                        pred = pred_null + audio_cfg * (pred - pred_null)
                    if keep_grad and grad_step is not None:
                        current = current.float() - sigmas[step] * pred.float()
                    else:
                        current = current.float() + (sigmas[step + 1] - sigmas[step]) * pred.float()
                    current = current.to(shape_reference.dtype)
                if trajectory:
                    states.append(current.detach().cpu())
            chunks.append(current)
            if trajectory:
                records.append({"start": start, "end": end, "states": torch.stack(states)})
            committed = history_chunk(current, conditions, start, history_mode)
            commit(model, committed, conditions, start, cache)
            if null_cache is not None:
                commit(model, committed, conditions, start, null_cache, null)
            prefix = torch.cat([prefix.detach(), committed], 2)
            if on_chunk is not None:
                on_chunk(index=index, start=start, end=end, chunk=current.detach())
    out = torch.cat(chunks, 2)
    return (out, records) if trajectory else out


def streaming_rollout_blocks(model, reference, reference_vace, block_inputs,
                             conditions, sigmas, *, max_video_history_frames,
                             history_mode="full"):
    """Yield causal blocks without materializing full-length video/VACE tensors.

    ``block_inputs`` is an iterable of dictionaries containing absolute ``start``,
    initial ``noise`` [B,C,F,H,W], and local ``vace_context`` for exactly that
    block. ``conditions`` contains global text/audio tensors, ``num_frames`` and
    optional precomputed ``audio_tokens``. The reference and every generated block
    are committed once to one bounded clean-history cache.
    """
    validate_schedule(sigmas)
    if max_video_history_frames is None:
        raise ValueError("Streaming rollout requires an explicit history bound")
    if reference.shape[2] != model.config.num_ref:
        raise ValueError("Reference latent count differs from model configuration")
    if reference_vace.shape[2] != reference.shape[2]:
        raise ValueError("Reference VACE shape mismatch")
    cache = HistoryCache(max_video_history_frames=max_video_history_frames)
    base = dict(conditions)
    with evaluating(model):
        if model.config.num_ref:
            ref_conditions = dict(base, vace_context=reference_vace, vace_start=0)
            commit(model, reference, ref_conditions, 0, cache)
        expected = model.config.video_blocks(
            (base["num_frames"]-1)//4+1+model.config.num_ref)
        for index, item in enumerate(block_inputs):
            if index >= len(expected):
                raise ValueError("Too many streaming blocks")
            start, end = expected[index]
            noise = item["noise"]
            if item.get("start", start) != start or noise.shape[2] != end-start:
                raise ValueError("Streaming block does not match the absolute schedule")
            local = dict(base, vace_context=item["vace_context"], vace_start=start)
            current = noise
            for step in range(len(sigmas)-1):
                pred = predict_chunk(model, current, sigmas[step], reference, local,
                                     start, cache, detach_history=True)
                current = (current.float()+(sigmas[step+1]-sigmas[step])*pred.float()).to(noise.dtype)
            committed = history_chunk(current, local, start, history_mode)
            commit(model, committed, local, start, cache)
            yield {"index": index, "start": start, "end": end,
                   "chunk": current, "cache_frames": cache.frames,
                   "cache_retained_frames": cache.retained_frames}
        if len(expected) and cache.frames != expected[-1][1]:
            raise ValueError("Streaming input ended before the configured video length")
