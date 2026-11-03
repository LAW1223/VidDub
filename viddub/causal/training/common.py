"""Shared math and state used by causal training objectives."""
import torch
from torch import nn


def spatial_weight(vace_context, mouth_weight):
    if mouth_weight < 1:
        raise ValueError("mouth_weight must be >= 1")
    mask = vace_context[:, 32:].amax(1, keepdim=True).float()
    return 1 + (mouth_weight - 1) * mask


def weighted_mse(a, b, weight):
    return ((a.float() - b.float()).square() * weight).mean() / weight.mean()


def flow_noise(x0, sigma, noise):
    return (1 - sigma) * x0 + sigma * noise


def dmd_gradient(x0, real_x0, fake_x0, active, eps=1e-6):
    dims = tuple(range(1, x0.ndim))
    denom = (active * (x0 - real_x0).abs()).sum(dims, keepdim=True)
    count = active.expand_as(x0).sum(dims, keepdim=True).clamp_min(1)
    denom = (denom / count).clamp_min(eps)
    gradient = (fake_x0 - real_x0) / denom
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite DMD score difference")
    return gradient * active


class ObjectiveBase(nn.Module):
    """Common generator state; frozen score/quality networks stay unregistered."""
    def __init__(self, model, *, mouth_weight=3, shift=5, student_steps=4,
                 use_cache=True, gradient_checkpointing=True, quality=None,
                 **_ignored):
        super().__init__()
        self.model = model
        self.mouth_weight = mouth_weight
        self.shift = shift
        self.student_steps = student_steps
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.quality = [quality]


class ScoreMixin:
    def _score(self, score, x, timestep, conditions, drop_audio=None, grad=False):
        if grad and self.dmd_score_offload:
            raise ValueError("Trainable scores must remain resident through backward")
        if self.dmd_score_offload:
            score.to(x.device)
        try:
            return score(
                x, timestep, conditions["context"], conditions["vace_context"],
                conditions["whisper_hs"], conditions["num_frames"], drop_audio,
                gradient_checkpointing=grad and self.gradient_checkpointing)
        finally:
            if self.dmd_score_offload:
                score.cpu()
                torch.cuda.empty_cache()
