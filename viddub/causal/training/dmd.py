"""Distribution-matching objective for the few-step causal student."""
import torch

from ..sampling import rollout, schedule
from .common import (ObjectiveBase, ScoreMixin, dmd_gradient, flow_noise,
                     spatial_weight, weighted_mse)


class DMDObjective(ScoreMixin, ObjectiveBase):
    def __init__(self, model, *, teacher=None, fake=None, anchor=None,
                 teacher_cfg=1, matching_min=.02, matching_max=.98,
                 dmd_rollout_mode="legacy_early_exit",
                 rollout_quality_weight=1, rollout_quality_cross=False,
                 dmd_retention_weight=0, dmd_score_offload=False, **kwargs):
        super().__init__(model, **kwargs)
        if not 0 < matching_min < matching_max < 1:
            raise ValueError("Matching range must lie strictly within (0,1)")
        if dmd_rollout_mode not in ("legacy_early_exit", "final_step"):
            raise ValueError("Unknown DMD rollout mode")
        self.scores = [teacher, fake, anchor]
        self.teacher_cfg = teacher_cfg
        self.matching_min, self.matching_max = matching_min, matching_max
        self.dmd_rollout_mode = dmd_rollout_mode
        self.rollout_quality_weight = rollout_quality_weight
        self.rollout_quality_cross = rollout_quality_cross
        self.dmd_retention_weight = dmd_retention_weight
        self.dmd_score_offload = dmd_score_offload

    def _anchor_rollout(self, anchor, x0, conditions, sigmas, noise):
        if self.dmd_score_offload:
            anchor.to(x0.device)
        try:
            return rollout(anchor, x0, conditions, sigmas, noise=noise,
                           use_cache=self.use_cache)
        finally:
            if self.dmd_score_offload:
                anchor.cpu()
                torch.cuda.empty_cache()

    def forward(self, x0, conditions, *, chunk_index=None,
                dmd_mode="combined", **_):
        if dmd_mode not in ("combined", "primary", "quality"):
            raise ValueError("Unknown DMD backward mode")
        teacher, fake, anchor = self.scores
        if teacher is None or fake is None:
            raise ValueError("DMD requires both real and fake score models")
        ranges = self.model.config.video_blocks(x0.shape[2])
        index = int(torch.randint(len(ranges), ())) if chunk_index is None else chunk_index
        start, stop = ranges[index]
        exit_step = (int(torch.randint(self.student_steps, ()))
                     if self.dmd_rollout_mode == "legacy_early_exit" else None)
        sigmas = schedule(self.student_steps, self.shift, x0.device)
        rollout_noise = torch.randn_like(x0)
        generated = rollout(
            self.model, x0, conditions, sigmas, noise=rollout_noise,
            use_cache=self.use_cache, grad_chunk=index, grad_step=exit_step,
            gradient_checkpointing=self.gradient_checkpointing)

        if dmd_mode == "quality":
            if self.quality[0] is None:
                raise ValueError("DMD quality-only pass requires quality supervision")
            if conditions.get("cross", False) and not self.rollout_quality_cross:
                zero = generated[:, :, start:stop].sum() * 0
                return zero, {
                    "rollout_aux_loss": zero.detach(), "chunk": float(index),
                    "exit_step": float(self.student_steps - 1
                                       if exit_step is None else exit_step)}
            auxiliary, stats = self.quality[0](
                generated[:, :, :stop], conditions, start, stop, 0.)
            loss = self.rollout_quality_weight * auxiliary
            metrics = {f"rollout_{key}": value.detach()
                       for key, value in stats.items()}
            metrics.update(
                rollout_aux_loss=auxiliary.detach(), chunk=float(index),
                exit_step=float(self.student_steps - 1
                                if exit_step is None else exit_step))
            return loss, metrics

        with torch.no_grad():
            sigma = torch.empty((), device=x0.device).uniform_(
                self.matching_min, self.matching_max)
            noisy = flow_noise(generated.detach(), sigma, torch.randn_like(generated))
            timestep = noisy.new_full(
                (noisy.shape[0], noisy.shape[2]), float(sigma) * 1000)
            real_velocity = self._score(teacher, noisy, timestep, conditions)
            if self.teacher_cfg != 1:
                null = torch.ones(noisy.shape[0], dtype=torch.bool,
                                  device=noisy.device)
                null_velocity = self._score(
                    teacher, noisy, timestep, conditions, null)
                real_velocity = null_velocity + self.teacher_cfg * (
                    real_velocity - null_velocity)
            real_x0 = noisy.float() - sigma * real_velocity.float()
            fake_x0 = noisy.float() - sigma * self._score(
                fake, noisy, timestep, conditions).float()
            active = torch.zeros_like(generated[:, :1], dtype=torch.float32)
            active[:, :, start:stop] = 1
            direction = dmd_gradient(
                generated.float(), real_x0, fake_x0, active)
            target = generated.detach().float() - direction

        loss = .5 * weighted_mse(
            generated[:, :, start:stop], target[:, :, start:stop],
            spatial_weight(conditions["vace_context"][:, :, start:stop],
                           self.mouth_weight))
        metrics = {
            "dmd": loss.detach(),
            "direction_rms": direction.square().mean().sqrt(),
            "chunk": float(index),
            "exit_step": float(self.student_steps - 1
                               if exit_step is None else exit_step),
        }
        if self.dmd_retention_weight:
            if anchor is None:
                raise ValueError("DMD retention requires a frozen ODE anchor")
            with torch.no_grad():
                anchor_generated = self._anchor_rollout(
                    anchor, x0, conditions, sigmas, rollout_noise)
            retention = weighted_mse(
                generated[:, :, start:stop], anchor_generated[:, :, start:stop],
                spatial_weight(conditions["vace_context"][:, :, start:stop],
                               self.mouth_weight))
            loss = loss + self.dmd_retention_weight * retention
            metrics["retention"] = retention.detach()
        if (dmd_mode == "combined" and self.quality[0] is not None and
                (not conditions.get("cross", False) or self.rollout_quality_cross)):
            auxiliary, stats = self.quality[0](
                generated[:, :, :stop], conditions, start, stop, 0.)
            loss = loss + self.rollout_quality_weight * auxiliary
            metrics.update({f"rollout_{key}": value.detach()
                            for key, value in stats.items()})
            metrics["rollout_aux_loss"] = auxiliary.detach()
        return loss, metrics
