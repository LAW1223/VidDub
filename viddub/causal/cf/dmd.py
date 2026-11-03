"""Asymmetric DMD aligned to pinned CF sampling and score-query conventions.

The real/fake models remain bidirectional. A single exit step is used for all
chunks and both fake/generator rollouts use the same exit-step distribution.
Wan-Dub differences: audio CFG (standard null + cfg*(cond-null) convention),
optional mouth weighting/quality/ODE retention, extra clean reference excluded
from video loss. No pretrained T2V checkpoint is substituted for Wan-Dub.
"""
import torch
import torch.distributed as dist
from .sampling import cf_rollout
from .upstream import FlowMatchScheduler
from ..training.common import flow_noise, spatial_weight, weighted_mse
from ..training.dmd import DMDObjective


def shared_exit(steps, device):
    value=torch.randint(steps,(1,),device=device)
    if dist.is_available() and dist.is_initialized():dist.broadcast(value,src=0)
    return int(value)


def score_sigma(device, shift=5., minimum=.02, maximum=.98):
    # Official DMD samples integer indices, shifts, clamps, then add_noise
    # maps the timestep to the nearest point on the 1000-point scheduler.
    t=torch.randint(0,1000,(),device=device).float()/1000
    t=(shift*t/(1+(shift-1)*t)).clamp(minimum,maximum)*1000
    grid=FlowMatchScheduler(shift=shift,sigma_min=0.,extra_one_step=True)
    grid.set_timesteps(1000,training=True)
    index=(grid.timesteps.to(device)-t).abs().argmin()
    return grid.sigmas.to(device)[index]


class CFAsymmetricDMD(DMDObjective):
    def dmd(self,x0,c,chunk_index=None,dmd_mode='combined'):
        if dmd_mode != 'combined':
            raise ValueError('CF asymmetric DMD does not implement split-quality backward')
        teacher,fake,anchor=self.scores
        if teacher is None or fake is None or teacher.causal or fake.causal:
            raise ValueError('CF asymmetric DMD requires bidirectional real and fake scores')
        ranges=self.model.config.video_blocks(x0.shape[2])
        i=int(torch.randint(len(ranges),())) if chunk_index is None else chunk_index
        a,b=ranges[i]
        exit_step=(self.student_steps-1 if self.dmd_rollout_mode=='final_step'
                   else shared_exit(self.student_steps,x0.device))
        seed=int(torch.randint(2**31-1,(),device=x0.device))
        generator=torch.Generator(device=x0.device).manual_seed(seed)
        # Sample one chunk for parameter gradients. This is a documented
        # memory-bounded unbiased chunk estimator, not full upstream backprop.
        gen=cf_rollout(self.model,x0,c,steps=self.student_steps,shift=self.shift,
            generator=generator,use_cache=self.use_cache,grad_chunk=i,grad_exit=exit_step,
            gradient_checkpointing=self.gradient_checkpointing)
        with torch.no_grad():
            s=score_sigma(x0.device,self.shift,self.matching_min,self.matching_max)
            noisy=flow_noise(gen.detach(),s,torch.randn_like(gen))
            ts=noisy.new_full((noisy.shape[0],noisy.shape[2]),float(s)*1000)
            real_v=self._score(teacher,noisy,ts,c)
            if self.teacher_cfg!=1:
                null=torch.ones(noisy.shape[0],device=x0.device,dtype=torch.bool)
                null_v=self._score(teacher,noisy,ts,c,null)
                real_v=null_v+self.teacher_cfg*(real_v-null_v)
            real_x=noisy.float()-s*real_v.float()
            fake_x=noisy.float()-s*self._score(fake,noisy,ts,c).float()
            n=self.model.config.num_ref
            denom=(gen[:,:,n:].float()-real_x[:,:,n:]).abs().mean((1,2,3,4),keepdim=True).clamp_min(1e-6)
            direction=(fake_x-real_x)/denom
            if not torch.isfinite(direction).all():raise FloatingPointError('Nonfinite DMD direction')
            target=gen.detach().float()-direction
        weight=spatial_weight(c['vace_context'][:,:,a:b],self.mouth_weight)
        loss=.5*weighted_mse(gen[:,:,a:b],target[:,:,a:b],weight)
        stats={'dmd':loss.detach(),'chunk':float(i),'exit_step':float(exit_step),
               'score_sigma':float(s),'direction_rms':direction[:,:,n:].square().mean().sqrt()}
        if self.dmd_retention_weight:
            if anchor is None:raise ValueError('Missing frozen anchor')
            if self.dmd_score_offload:anchor.to(x0.device)
            with torch.no_grad():
                fixed=cf_rollout(anchor,x0,c,steps=self.student_steps,shift=self.shift,
                    generator=torch.Generator(device=x0.device).manual_seed(seed),
                    use_cache=self.use_cache,grad_exit=exit_step)
            if self.dmd_score_offload:anchor.cpu();torch.cuda.empty_cache()
            retain=weighted_mse(gen[:,:,a:b],fixed[:,:,a:b],weight)
            loss+=self.dmd_retention_weight*retain;stats['retention']=retain.detach()
        if self.quality[0] is not None and (not c.get('cross',False) or self.rollout_quality_cross):
            aux,aux_stats=self.quality[0](gen[:,:,:b],c,a,b,0.)
            loss+=self.rollout_quality_weight*aux
            stats.update({f'rollout_{k}':v.detach() for k,v in aux_stats.items()})
        return loss,stats

    def fake(self,generated,c):
        if generated is None:raise ValueError('Fake score requires generated samples')
        gen=generated.detach();s=score_sigma(gen.device,self.shift,self.matching_min,self.matching_max)
        noise=torch.randn_like(gen);noisy=flow_noise(gen,s,noise)
        ts=gen.new_full((gen.shape[0],gen.shape[2]),float(s)*1000)
        pred=self._score(self.model,noisy,ts,c,grad=True)
        n=self.model.config.num_ref
        loss=weighted_mse(pred[:,:,n:],(noise-gen)[:,:,n:],
                           spatial_weight(c['vace_context'][:,:,n:],self.mouth_weight))
        return loss,{'fake_flow':loss.detach(),'score_sigma':float(s)}
