"""Fake-score flow-matching objective used by DMD."""
import torch

from .common import (ObjectiveBase, ScoreMixin, flow_noise, spatial_weight,
                     weighted_mse)


class FakeObjective(ScoreMixin, ObjectiveBase):
    def __init__(self, model, *, matching_min=.02, matching_max=.98,
                 dmd_score_offload=False, **kwargs):
        super().__init__(model, **kwargs)
        if not 0 < matching_min < matching_max < 1:
            raise ValueError("Matching range must lie strictly within (0,1)")
        self.matching_min, self.matching_max = matching_min, matching_max
        self.dmd_score_offload = dmd_score_offload

    def forward(self, x0, conditions, *, generated=None, **_):
        if generated is None:
            raise ValueError("Fake score training requires a student rollout")
        generated = generated.detach()
        sigma = torch.empty((), device=generated.device).uniform_(
            self.matching_min, self.matching_max)
        noise = torch.randn_like(generated)
        noisy = flow_noise(generated, sigma, noise)
        n_ref = self.model.config.num_ref
        timestep = generated.new_full(
            (generated.shape[0], generated.shape[2]), float(sigma) * 1000)
        pred = self._score(
            self.model, noisy, timestep, conditions, grad=True)
        loss = weighted_mse(
            pred[:, :, n_ref:], (noise - generated)[:, :, n_ref:],
            spatial_weight(conditions["vace_context"][:, :, n_ref:],
                           self.mouth_weight))
        return loss, {"fake_flow": loss.detach()}
