"""Official CF few-step rule: x0 prediction then independent re-noising.

Wan-Dub implementation of pipeline/causal_inference.py from the pinned source.
Unlike Euler sampling, this has independent Gaussian noise at each transition.
"""
import torch
from .upstream import FlowMatchScheduler
from .consistency import flow_to_x0
from ..model import HistoryCache
from ..sampling import evaluating, commit, predict_chunk


def cf_rollout(model, shape_reference, c, *, steps=4, shift=5., noise=None,
               generator=None, use_cache=True, grad_chunk=None, grad_exit=None,
               gradient_checkpointing=False):
    if steps < 1 or 1000 % steps:
        raise ValueError('CF schedule requires steps to divide 1000')
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    timesteps = scheduler.timesteps[torch.arange(0,1000,1000//steps)].to(shape_reference.device)
    noise = torch.randn(shape_reference.shape,device=shape_reference.device,dtype=shape_reference.dtype,
                        generator=generator) if noise is None else noise
    if noise.shape != shape_reference.shape:
        raise ValueError('Noise shape differs')
    cache = HistoryCache() if use_cache else None
    prefix = shape_reference[:, :, :model.config.num_ref].detach()
    output = [prefix]
    if grad_exit is not None and not 0 <= grad_exit < steps:
        raise ValueError('Invalid exit step')
    with evaluating(model):
        if prefix.shape[2]:
            commit(model, prefix, c, 0, cache)
        for index, (a,b) in enumerate(model.config.video_blocks(shape_reference.shape[2])):
            current = noise[:, :, a:b]
            final = steps-1 if grad_exit is None else grad_exit
            for j in range(final+1):
                keep = index == grad_chunk and j == final
                with torch.set_grad_enabled(keep):
                    # Upstream quantizes the warped timestep to integer at inference.
                    t = timesteps[j].long()
                    v = predict_chunk(model,current,float(t)/1000,prefix,c,a,cache,
                                      gradient_checkpointing=gradient_checkpointing,detach_history=True)
                    estimate = flow_to_x0(current,v,t,scheduler)
                if j < final:
                    with torch.no_grad():
                        tnext = timesteps[j+1].long()
                        step_index = (scheduler.timesteps.to(tnext.device)-tnext).abs().argmin()
                        sigma = scheduler.sigmas[step_index].to(estimate)
                        eps = torch.randn(estimate.shape,device=estimate.device,dtype=estimate.dtype,generator=generator)
                        current = ((1-sigma)*estimate.detach()+sigma*eps).to(shape_reference.dtype)
            output.append(estimate)
            commit(model,estimate,c,a,cache)
            prefix = torch.cat([prefix.detach(),estimate.detach()],2)
    return torch.cat(output,2)
