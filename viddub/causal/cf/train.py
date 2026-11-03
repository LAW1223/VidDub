"""Finite, resumable CF++ causal-CD trainer with Wan-Dub conditioning.

Uses the official online adjacent teacher-step objective, AdamW and EMA.
All blocks contribute per clip; individual backward calls bound activation
memory without detaching student clean-history parameter paths by default.
"""
import argparse
import copy
import datetime
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .consistency import CausalCD, ModelEMA
from .upstream import provenance as upstream_provenance
from ..runtime import (atomic_json, autocast, checkpoint_paths, export_weights, fingerprint,
                       first_item, load_components, local_imports, make_dataset, prepare_clip,
                       read_config, require_causal_metadata)
from ..train import check_loss, finite_update, optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--init-dir',required=True)
    parser.add_argument('--init-step',type=int,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-step',type=int)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config)
    if cfg['stage'] != 'cd' or cfg['grad_accum'] != 1 or cfg['p_cross'] != 0 or cfg['art_weight']:
        raise ValueError('CD pilot requires stage=cd, grad_accum=1, same data, Art disabled')
    if cfg['warmup'] or cfg['lr_schedule'] != 'constant':
        raise ValueError('This CF CD trainer uses the official constant LR without warmup')
    final = cfg['max_steps'] if args.stop_after_step is None else args.stop_after_step
    if not 0 < final <= cfg['max_steps']:
        raise ValueError('Invalid operational stop')
    out = Path(args.output)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')
    paths = checkpoint_paths(args.init_dir,args.init_step)
    require_causal_metadata(args.init_dir,args.init_step,cfg,('ar',))
    dataset, rows = make_dataset(cfg)
    record = {'student_init': fingerprint(paths), 'upstream':upstream_provenance(),
              'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
              'adapter_sources':fingerprint({n:Path(__file__).parent/n for n in
                                           ['train.py','consistency.py','sampling.py','upstream.py']})}
    resume = None
    if args.resume:
        if Path(args.resume).resolve().parent != out.resolve():
            raise ValueError('Resume state must belong to this run')
        resume = json.loads((Path(args.resume)/'complete.json').read_text())
        if read_config(resume['config']) != cfg or resume['provenance'] != record:
            raise ValueError('Resume config, code or provenance differs')
    if args.check_only:
        print(json.dumps({'status':'validated_no_gpu','samples':len(dataset),'provenance':record},indent=2))
        return
    local_imports()
    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    from viddub.full_context.runtime import ZeroLatents
    accel = Accelerator(mixed_precision='no',gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False,gradient_as_bucket_view=True),
                        InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=20))])
    if accel.device.type != 'cuda':
        raise RuntimeError('Training requires CUDA')
    random.seed(cfg['seed']+accel.process_index);torch.manual_seed(cfg['seed']+accel.process_index)
    pipe, whisper, student = load_components(paths,cfg,accel.device)
    student.requires_grad_(True).eval()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    ema = ModelEMA(copy.deepcopy(student),cfg['cd_ema_decay'])
    quality = None
    if any(cfg[k] for k in ['pixel_weight','lpips_weight','keep_weight','sync_weight']):
        from ..quality import GTQuality
        quality = GTQuality(pipe,cfg,accel.device)
    obj = CausalCD(student,teacher,ema.model,cfg,quality)
    opt = optimizer(student,cfg)
    loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=first_item,
        num_workers=cfg['workers'],generator=torch.Generator().manual_seed(cfg['seed']))
    obj,opt,loader = accel.prepare(obj,opt,loader)
    accel.register_for_checkpointing(ema)
    step=epoch=consumed=0
    if resume:
        if resume['world_size'] != accel.num_processes:
            raise ValueError('Resume requires same world size')
        accel.load_state(args.resume)
        step,epoch,consumed=(resume[k] for k in ('step','epoch','consumed_batches'))
    else:
        random.seed(cfg['seed']+accel.process_index);torch.manual_seed(cfg['seed']+accel.process_index)
    if accel.is_main_process:
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'run.json',{'config':cfg,'provenance':record,'world_size':accel.num_processes,
                                   'scope':'diagnostic_CD_pilot_not_AR_promotion'})
    accel.wait_for_everyone()
    zero = ZeroLatents(pipe)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    def save():
        state=out/f'state_step{step:06d}'
        exists=accel.reduce(torch.tensor(int(state.exists()),device=accel.device),'sum')
        if int(exists):raise FileExistsError(state)
        full = step == cfg['max_steps'] or step == final or step % cfg['save_state_every'] == 0
        if full:accel.save_state(str(state))
        if accel.is_main_process:
            export_weights(student,out,step,cfg,record)
            export_weights(ema.model,out/'ema',step,cfg,{**record,'export':'ema','ema_updates':ema.updates})
            if full:
                atomic_json(state/'complete.json',{'step':step,'epoch':epoch,'consumed_batches':consumed,
                    'config':cfg,'provenance':record,'world_size':accel.num_processes,'ema_updates':ema.updates})
        accel.wait_for_everyone()
    while step < final:
        if hasattr(loader,'set_epoch'):loader.set_epoch(epoch)
        if getattr(loader,'generator',None) is not None:loader.generator.manual_seed(cfg['seed']+epoch)
        active=accel.skip_first_batches(loader,consumed) if consumed else loader
        for item in active:
            consumed += 1
            clean,c = prepare_clip(item,pipe,whisper,zero,cfg)
            c['cross']=False
            # Identical RNG consumption in pure CD and GT auxiliary variants.
            index=random.randrange(cfg['cd_grid_steps']-1)
            noise=torch.randn_like(clean)
            raw=accel.unwrap_model(obj)
            xt,t,tn=raw.sample_input(clean,index,noise)
            ranges=student.config.video_blocks(clean.shape[2])
            indices=list(range(len(ranges))) if cfg['cd_all_chunks'] else [int(torch.randint(len(ranges),()))]
            stats={};value=clean.new_zeros((),dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            for j,i in enumerate(indices):
                # Backward per chunk; accumulate parameter gradients before one
                # DDP all-reduce and optimizer step for the complete clip.
                context=accel.no_sync(obj) if j<len(indices)-1 else nullcontext()
                with context:
                    with autocast(accel.device,cfg['precision']):
                        loss,metrics=obj(clean,c,xt=xt,timestep=t,next_timestep=tn,chunk_index=i)
                        loss=loss/len(indices)
                    check_loss(accel,loss)
                    accel.backward(loss)
                value += loss.detach()
                for key,v in metrics.items():stats[key]=stats.get(key,0.)+float(v)/len(indices)
                del loss,metrics
            norm=finite_update(accel,obj,opt,cfg)
            ema.update(student)
            step += 1
            if step%cfg['log_every']==0 or step==final:
                keys=sorted(stats)
                values=torch.tensor([stats[k] for k in keys],device=accel.device)
                values=accel.reduce(values,'mean').tolist()
                log={'step':step,'loss':float(accel.reduce(value,'mean')),'grad_norm':norm,
                    'elapsed_seconds':time.monotonic()-started,'ema_updates':ema.updates,
                    'blocks_per_clip':len(indices),'grid_index':index,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,**dict(zip(keys,values))}
                accel.print(json.dumps(log),flush=True)
                if accel.is_main_process:
                    with (out/'train.jsonl').open('a') as handle:handle.write(json.dumps(log)+'\n')
            if step%cfg['save_every']==0 or step in cfg['save_steps'] or step==final:save()
            if step==final:break
        epoch,consumed=epoch+1,0
    accel.end_training()


if __name__=='__main__':main()
