"""Trajectory-regression objective for few-step ODE distillation."""
import torch

from ..sampling import predict_chunk, rollout, schedule
from .common import ObjectiveBase, flow_noise, spatial_weight, weighted_mse


class ODEObjective(ObjectiveBase):
    def __init__(self, model, *, ode_gt_flow_weight=0,
                 rollout_quality_weight=1, rollout_quality_cross=False,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.ode_gt_flow_weight = ode_gt_flow_weight
        self.rollout_quality_weight = rollout_quality_weight
        self.rollout_quality_cross = rollout_quality_cross

    def forward(self, x0, conditions, *, record=None, **_):
        if record is None:
            raise ValueError("ODE stage requires a trajectory record")
        start, stop, sigma = record["start"], record["end"], record["sigma"]
        noisy = record["noisy"].to(x0)
        pred = predict_chunk(
            self.model, noisy, sigma, record["prefix"].to(x0), conditions, start,
            gradient_checkpointing=self.gradient_checkpointing)
        estimate = noisy.float() - float(sigma) * pred.float()
        loss = weighted_mse(
            estimate, record["target"].to(x0),
            spatial_weight(conditions["vace_context"][:, :, start:stop],
                           self.mouth_weight))
        metrics = {"ode_x0": loss.detach()}

        gt = record.get("gt_x0")
        if self.ode_gt_flow_weight:
            if gt is None:
                raise ValueError("ODE GT flow requires trajectories containing gt_latents")
            if not conditions.get("cross", False):
                gt = gt.to(x0)
                clean = gt[:, :, start:stop]
                u = torch.rand((), device=x0.device).clamp(1e-4, 1 - 1e-4)
                gt_sigma = self.shift * u / (1 + (self.shift - 1) * u)
                noise = torch.randn_like(clean)
                gt_noisy = flow_noise(clean, gt_sigma, noise)
                gt_pred = predict_chunk(
                    self.model, gt_noisy, gt_sigma, gt[:, :, :start].detach(),
                    conditions, start,
                    gradient_checkpointing=self.gradient_checkpointing)
                gt_flow = weighted_mse(
                    gt_pred, noise - clean,
                    spatial_weight(conditions["vace_context"][:, :, start:stop],
                                   self.mouth_weight))
                loss = loss + self.ode_gt_flow_weight * gt_flow
                metrics["ode_gt_flow"] = gt_flow.detach()
            else:
                metrics["ode_gt_flow"] = loss.detach() * 0

        if (self.quality[0] is not None and
                (not conditions.get("cross", False) or self.rollout_quality_cross)):
            ranges = self.model.config.video_blocks(x0.shape[2])
            chunk_index = next(
                index for index, pair in enumerate(ranges) if pair == (start, stop))
            generated = rollout(
                self.model, x0, conditions,
                schedule(self.student_steps, self.shift, x0.device),
                use_cache=self.use_cache, grad_chunk=chunk_index, grad_step=None,
                gradient_checkpointing=self.gradient_checkpointing)
            auxiliary, stats = self.quality[0](
                generated[:, :, :stop], conditions, start, stop, 0.)
            loss = loss + self.rollout_quality_weight * auxiliary
            metrics.update({f"rollout_{key}": value.detach()
                            for key, value in stats.items()})
            metrics["rollout_aux_loss"] = auxiliary.detach()
        return loss, metrics
