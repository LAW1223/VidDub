"""Autoregressive causal flow-matching objective."""
import torch

from ..sampling import evaluating, predict_chunk, schedule
from .common import ObjectiveBase, flow_noise, spatial_weight, weighted_mse


class ARObjective(ObjectiveBase):
    def __init__(self, model, *, art=None, art_weight=0, art_layer=8,
                 art_warmup=50, ar_history_mode="gt",
                 ar_history_prob_start=0, ar_history_prob_end=0,
                 ar_history_ramp_steps=0, ar_history_steps=4,
                 ar_history_sigma_max=.5, ar_history_mouth_drop_prob=0,
                 ar_history_mouth_drop_cross_only=False, **kwargs):
        super().__init__(model, **kwargs)
        self.art, self.art_weight = art, art_weight
        self.art_layer, self.art_warmup = art_layer, art_warmup
        self.ar_history_mode = ar_history_mode
        self.ar_history_prob_start = ar_history_prob_start
        self.ar_history_prob_end = ar_history_prob_end
        self.ar_history_ramp_steps = ar_history_ramp_steps
        self.ar_history_steps = ar_history_steps
        self.ar_history_sigma_max = ar_history_sigma_max
        self.ar_history_mouth_drop_prob = ar_history_mouth_drop_prob
        self.ar_history_mouth_drop_cross_only = ar_history_mouth_drop_cross_only

    def _history_probability(self, step):
        if self.ar_history_ramp_steps <= 0:
            return self.ar_history_prob_end
        alpha = min(1., max(0., float(step) / self.ar_history_ramp_steps))
        return self.ar_history_prob_start + alpha * (
            self.ar_history_prob_end - self.ar_history_prob_start)

    @torch.no_grad()
    def _generated_history(self, x0, conditions, end, generator):
        n_ref = self.model.config.num_ref
        prefix = x0[:, :, :n_ref].detach()
        if end <= n_ref:
            return prefix
        with evaluating(self.model):
            for start, stop in self.model.config.video_blocks(end):
                if self.ar_history_mode == "resample":
                    u_max = self.ar_history_sigma_max / (
                        self.shift - self.ar_history_sigma_max * (self.shift - 1))
                    u = torch.rand((), device=x0.device, generator=generator) * u_max
                    sigma = self.shift * u / (1 + (self.shift - 1) * u)
                    clean = x0[:, :, start:stop]
                    noise = torch.randn(clean.shape, device=clean.device,
                                        dtype=clean.dtype, generator=generator)
                    noisy = flow_noise(clean, sigma, noise)
                    pred = predict_chunk(
                        self.model, noisy, sigma, prefix, conditions, start,
                        drop_audio=conditions.get("drop_audio"))
                    chunk = (noisy.float() - float(sigma) * pred.float()).to(x0.dtype)
                elif self.ar_history_mode == "rollout":
                    current = torch.randn(
                        x0[:, :, start:stop].shape, device=x0.device,
                        dtype=x0.dtype, generator=generator)
                    sigmas = schedule(self.ar_history_steps, self.shift, x0.device)
                    for index in range(len(sigmas) - 1):
                        pred = predict_chunk(
                            self.model, current, sigmas[index], prefix,
                            conditions, start, drop_audio=conditions.get("drop_audio"))
                        current = (current.float() + (sigmas[index + 1] - sigmas[index])
                                   * pred.float()).to(x0.dtype)
                    chunk = current
                else:
                    raise ValueError(self.ar_history_mode)
                prefix = torch.cat([prefix, chunk.detach()], 2)
        return prefix

    def forward(self, x0, conditions, *, chunk_index=None, sigma=None,
                noise=None, **_):
        ranges = self.model.config.video_blocks(x0.shape[2])
        index = int(torch.randint(len(ranges), ())) if chunk_index is None else chunk_index
        start, stop = ranges[index]
        u = torch.rand((), device=x0.device).clamp(1e-4, 1 - 1e-4)
        sigma = self.shift * u / (1 + (self.shift - 1) * u) if sigma is None else sigma
        clean = x0[:, :, start:stop]
        noise = torch.randn_like(clean) if noise is None else noise
        noisy = flow_noise(clean, sigma, noise)
        prefix = x0[:, :, :start].detach()

        history_mouth_dropped = False
        eligible = (start > self.model.config.num_ref and
                    (not self.ar_history_mouth_drop_cross_only or conditions.get("cross", False)))
        if self.ar_history_mouth_drop_prob and eligible:
            generator = torch.Generator(device=x0.device).manual_seed(
                int(conditions.get("history_seed", 0)) + index * 1009 + 17)
            history_mouth_dropped = bool(
                torch.rand((), device=x0.device, generator=generator)
                < self.ar_history_mouth_drop_prob)
            if history_mouth_dropped:
                mask = conditions["vace_context"][:, 32:, :start].amax(
                    1, keepdim=True).float().clamp(0, 1)
                mask[:, :, :self.model.config.num_ref] = 0
                inactive = conditions["vace_context"][:, :x0.shape[1], :start].to(prefix)
                prefix = prefix * (1 - mask) + inactive * mask

        history_used = False
        if self.ar_history_mode != "gt" and start > self.model.config.num_ref:
            generator = torch.Generator(device=x0.device).manual_seed(
                int(conditions.get("history_seed", 0)))
            probability = self._history_probability(conditions.get("train_step", 0))
            history_used = bool(torch.rand((), device=x0.device, generator=generator) < probability)
            if history_used:
                prefix = self._generated_history(x0, conditions, start, generator)

        result = predict_chunk(
            self.model, noisy, sigma, prefix, conditions, start,
            drop_audio=conditions.get("drop_audio"),
            gradient_checkpointing=self.gradient_checkpointing,
            return_hidden_layer=self.art_layer if self.art is not None else None)
        pred, hidden = result if self.art is not None else (result, None)
        current_mask = conditions["vace_context"][:, 32:, start:stop].amax(
            1, keepdim=True).float().clamp(0, 1)
        weight = ((1 - current_mask) if conditions.get("cross", False)
                  else spatial_weight(conditions["vace_context"][:, :, start:stop],
                                      self.mouth_weight))
        loss = weighted_mse(pred, noise - clean, weight)
        metrics = {
            "fm": loss.detach(), "chunk": float(index), "sigma": float(sigma),
            "history_eligible": float(start > self.model.config.num_ref),
            "history_generated": float(history_used),
            "history_probability": float(self._history_probability(
                conditions.get("train_step", 0))),
            "cross": float(conditions.get("cross", False)),
            "history_mouth_dropped": float(history_mouth_dropped),
        }
        if self.quality[0] is not None:
            estimate = noisy.float() - float(sigma) * pred.float()
            auxiliary, stats = self.quality[0](
                torch.cat([prefix, estimate], 2), conditions, start, stop, sigma)
            loss = loss + auxiliary
            metrics.update({key: value.detach() for key, value in stats.items()})
            metrics["aux_loss"] = auxiliary.detach()
        if self.art is not None:
            target = self.quality[0].art_target(conditions, stop)
            art_loss = (sum(parameter.sum() * 0 for parameter in self.art.parameters())
                        if target is None else self.art(
                            hidden, conditions["pixel_mask"].to(hidden.device), target,
                            num_ref=self.model.config.num_ref, history_end=start,
                            warmup=conditions.get("train_step", 0) < self.art_warmup))
            loss = loss + self.art_weight * art_loss
            metrics["art"] = art_loss.detach()
        return loss, metrics
