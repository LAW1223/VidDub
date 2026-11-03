"""Adapted from official model/naive_consistency.py (Apache-2.0).

Changes: Wan-Dub conditions/reference; optional frozen history caches; chunkwise
backward to bound memory. All seven losses can be accumulated with full student
history gradients, matching the upstream teacher-forcing attention dependency.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

from .upstream import FlowMatchScheduler
from ..model import HistoryCache
from ..sampling import commit, evaluating, predict_chunk


def flow_to_x0(xt, velocity, timestep, scheduler):
    """Same nearest-grid and float64 arithmetic as official WanDiffusionWrapper."""
    grid = scheduler.timesteps.to(device=xt.device, dtype=torch.float64)
    index = (grid - torch.as_tensor(timestep, device=xt.device).double()).abs().argmin()
    sigma = scheduler.sigmas.to(device=xt.device, dtype=torch.float64)[index]
    return (xt.double() - sigma * velocity.double()).to(velocity.dtype)


class ModelEMA:
    """Frozen target model, updated every optimizer step and included in resume."""
    def __init__(self, model, decay):
        self.model = model.eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, student):
        for (_, target), (_, source) in zip(self.model.named_parameters(), student.named_parameters()):
            target.mul_(self.decay).add_(source.detach().to(target), alpha=1-self.decay)
        for (_, target), (_, source) in zip(self.model.named_buffers(), student.named_buffers()):
            target.copy_(source)
        self.updates += 1

    def state_dict(self):
        return {'model': self.model.state_dict(), 'decay': self.decay, 'updates': self.updates}

    def load_state_dict(self, state):
        if state['decay'] != self.decay:
            raise ValueError('EMA decay differs on resume')
        self.model.load_state_dict(state['model'], strict=True)
        self.updates = state['updates']


@torch.no_grad()
def clean_history_cache(model, clean, conditions, end):
    cache = HistoryCache()
    with evaluating(model):
        for a, b in model.config.blocks(clean.shape[2]):
            if b > end:
                break
            commit(model, clean[:, :, a:b], conditions, a, cache)
    if cache.frames != end:
        raise ValueError('History endpoint must be a block boundary')
    return cache


class CausalCD(nn.Module):
    def __init__(self, model, teacher, ema, cfg, quality=None):
        super().__init__()
        self.model = model
        # Do not register frozen targets in DDP or the student optimizer.
        self.targets = [teacher, ema]
        self.quality = [quality]
        self.cfg = cfg
        self.grid = FlowMatchScheduler(shift=cfg['shift'], sigma_min=0., extra_one_step=True)
        self.grid.set_timesteps(cfg['cd_grid_steps'])
        self.conversion_grid = FlowMatchScheduler(shift=cfg['shift'], sigma_min=0., extra_one_step=True)
        self.conversion_grid.set_timesteps(1000, training=True)

    def sample_input(self, clean, index, noise):
        if not 0 <= index < self.cfg['cd_grid_steps'] - 1:
            raise ValueError('CD adjacent index outside nonzero upstream grid')
        # Official add_noise uses the exact 48-point sigma; the transformer
        # timesteps are cast to the configured compute dtype (BF16 upstream).
        s = self.grid.sigmas[index].to(device=clean.device, dtype=torch.float32)
        xt = ((1-s.float()) * clean.float() + s.float()*noise.float()).to(clean.dtype)
        t = self.grid.timesteps[index].to(clean)
        tn = self.grid.timesteps[index+1].to(clean)
        return xt, t, tn

    def forward(self, clean, c, *, xt, timestep, next_timestep, chunk_index):
        a, b = self.model.config.video_blocks(clean.shape[2])[chunk_index]
        current = xt[:, :, a:b]
        prefix = clean[:, :, :a].detach()
        teacher, ema = self.targets
        with torch.no_grad():
            teacher_cache = clean_history_cache(teacher, clean, c, a) if self.cfg['cd_frozen_cache'] else None
            velocity = predict_chunk(teacher, current, float(timestep)/1000, prefix, c, a, teacher_cache)
            if self.cfg['teacher_cfg'] != 1:
                null_c = dict(c, drop_audio=torch.ones(clean.shape[0],device=clean.device,dtype=torch.bool))
                # Conditional and unconditional histories must be different.
                velocity_null = predict_chunk(teacher, current, float(timestep)/1000, prefix, c, a,
                                              drop_audio=null_c['drop_audio'])
                velocity = velocity_null + self.cfg['teacher_cfg'] * (velocity-velocity_null)
            dt = (timestep-next_timestep)/1000
            next_xt = current - dt * velocity
            del teacher_cache
            ema_cache = clean_history_cache(ema, clean, c, a) if self.cfg['cd_frozen_cache'] else None
            ema_v = predict_chunk(ema, next_xt, float(next_timestep)/1000, prefix, c, a, ema_cache)
            target = flow_to_x0(next_xt, ema_v, next_timestep, self.conversion_grid).detach()
            del ema_cache
        # No student cache by default: preserve parameter gradients through
        # clean-history forward, as in the official doubled-sequence TF mask.
        cache = clean_history_cache(self.model, clean, c, a) if self.cfg['cd_student_cache'] else None
        velocity = predict_chunk(self.model, current, float(timestep)/1000, prefix, c, a, cache,
                                 gradient_checkpointing=self.cfg['gradient_checkpointing'])
        estimate = flow_to_x0(current, velocity, timestep, self.conversion_grid)
        cd_loss = F.mse_loss(estimate.float(), target.float())
        loss = cd_loss
        stats = {'cd': cd_loss.detach(), 'sigma': float(timestep)/1000,
                 'chunk': float(chunk_index), 'aux_loss': 0.}
        if self.quality[0] is not None:
            available = torch.cat([prefix, estimate], 2)
            aux, aux_stats = self.quality[0](available, c, a, b, float(timestep)/1000)
            loss = loss + self.cfg['cd_aux_scale'] * aux
            stats.update({k: v.detach() for k,v in aux_stats.items()})
            stats['aux_loss'] = aux.detach()
        return loss, stats
