"""Pretrain and export one reusable fake-score model on frozen student rollouts."""
import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from .training.stages import FakeObjective
from .runtime import (atomic_json, autocast, checkpoint_paths, export_weights, fingerprint,
                      first_item, load_components, make_dataset, prepare_clip, read_config,
                      require_causal_metadata)
from .sampling import rollout, schedule
from .training.engine import check_loss, finite_update, lr_scheduler, optimizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--student-dir',required=True);p.add_argument('--student-step',type=int,required=True)
    p.add_argument('--teacher-dir',required=True);p.add_argument('--teacher-step',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume')
    p.add_argument('--stop-after-step',type=int)
    a=p.parse_args();cfg=read_config(a.config)
    run_until = cfg['max_steps'] if a.stop_after_step is None else a.stop_after_step
    if not 0 < run_until <= cfg['max_steps']:
        raise ValueError('stop-after-step must lie within the configured training budget')
    if cfg['sampler'] != 'euler':
        raise ValueError('Shared fake pretraining currently supports only the Euler sampler')
    if cfg['stage']!='dmd':raise ValueError('Fake pretraining requires a DMD config')
    student_paths=checkpoint_paths(a.student_dir,a.student_step)
    require_causal_metadata(a.student_dir,a.student_step,cfg,('ode','dmd'))
    teacher_paths=checkpoint_paths(a.teacher_dir,a.teacher_step)
    if a.resume and Path(a.resume).resolve().parent != Path(a.output).resolve():
        raise ValueError('Resume state must belong to this output directory')
    if not a.resume and Path(a.output).exists() and any(Path(a.output).iterdir()):
        raise ValueError('Output is not empty; choose a new output or explicitly --resume')
    dataset,rows=make_dataset(cfg)
    from .runtime import local_imports
    local_imports()
    import accelerate
    from viddub.full_context.runtime import ZeroLatents
    accel=accelerate.Accelerator(mixed_precision='no',gradient_accumulation_steps=1,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=False,
            gradient_as_bucket_view=True),accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))])
    pipe,whisper,student=load_components(student_paths,cfg,accel.device)
    fake=copy.deepcopy(student)
    for name,path in teacher_paths.items():
        getattr(fake,name).load_state_dict(load_file(str(path)),strict=True)
    student.requires_grad_(False).eval();fake.causal=False;fake.requires_grad_(True).train()
    objective=FakeObjective(fake,mouth_weight=cfg['mouth_weight'],shift=cfg['shift'],
        student_steps=cfg['student_steps'],use_cache=cfg['use_cache'],
        gradient_checkpointing=cfg['gradient_checkpointing'],
        matching_min=cfg['matching_min'], matching_max=cfg['matching_max'])
    opt=optimizer(fake,dict(cfg,lr=cfg['fake_lr'],lr_base=cfg['fake_lr_base']))
    sched=lr_scheduler(opt,cfg)
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=first_item,
        num_workers=cfg['workers'],generator=torch.Generator().manual_seed(cfg['seed']))
    objective,opt,loader=accel.prepare(objective,opt,loader);accel.register_for_checkpointing(sched)
    provenance={'student':fingerprint(student_paths),'teacher_init':fingerprint(teacher_paths)}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if accel.is_main_process:atomic_json(out/'run.json',{'config':cfg,'provenance':provenance,
        'world_size':accel.num_processes,'kind':'shared_fake_pretrain'})
    step=epoch=consumed=0
    random.seed(cfg['seed']+accel.process_index)
    torch.manual_seed(cfg['seed']+accel.process_index)
    if a.resume:
        meta=json.loads((Path(a.resume)/'complete.json').read_text())
        if meta['provenance']!=provenance or meta['config']!=cfg:raise ValueError('Resume mismatch')
        if meta['world_size']!=accel.num_processes:raise ValueError('Resume requires same world size')
        accel.load_state(a.resume)
        step,epoch,consumed=(meta[k] for k in ('step','epoch','consumed_batches'))
    zero=ZeroLatents(pipe);start=time.monotonic()
    row_index={r['clip_id']:i for i,r in enumerate(rows)}
    while step<run_until:
        if hasattr(loader,'set_epoch'):loader.set_epoch(epoch)
        if getattr(loader,'generator',None) is not None:
            loader.generator.manual_seed(cfg['seed']+epoch)
        active_loader=accel.skip_first_batches(loader,consumed) if consumed else loader
        for item in active_loader:
            consumed+=1
            if random.random()<cfg['p_cross']:
                other=dataset[random.randrange(len(dataset))]
                if other['clip_id']==item['clip_id']:
                    other=dataset[(row_index[item['clip_id']]+1)%len(dataset)]
                item=dict(item,audio=other['audio'],cross=True)
            x0,c=prepare_clip(item,pipe,whisper,zero,cfg)
            c['cross']=bool(item.get('cross',False))
            with torch.no_grad(),autocast(accel.device,cfg['precision']):
                generated=rollout(student,x0,c,schedule(cfg['student_steps'],cfg['shift'],x0.device),
                                  use_cache=cfg['use_cache'])
            with autocast(accel.device,cfg['precision']):loss,metrics=objective(x0,c,generated=generated)
            check_loss(accel,loss);accel.backward(loss);norm=finite_update(accel,objective,opt,cfg);sched.step()
            step+=1
            if step%cfg['log_every']==0:accel.print(json.dumps({'step':step,'loss':float(loss.detach()),
                'grad_norm':norm,'elapsed_seconds':time.monotonic()-start,
                **{k:float(v) for k,v in metrics.items()}}),flush=True)
            if step%cfg['save_every']==0 or step==cfg['max_steps'] or step==run_until:
                state=out/f'state_step{step:06d}'
                exists=accel.reduce(torch.tensor(int(state.exists()),device=accel.device),'sum')
                if int(exists):raise FileExistsError(state)
                accel.save_state(str(state))
                if accel.is_main_process:
                    export_weights(accel.unwrap_model(objective).model,out,step,cfg,provenance)
                    atomic_json(state/'complete.json',{'step':step,'epoch':epoch,
                        'consumed_batches':consumed,'config':cfg,'provenance':provenance,
                        'world_size':accel.num_processes})
                accel.wait_for_everyone()
            if step==run_until:break
        epoch,consumed=epoch+1,0
    accel.end_training()


if __name__=='__main__':main()
