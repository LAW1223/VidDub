"""Wan-Dub training (accelerate DDP). Stage s1: train audio adapter only; stage s2: + VACE + DiT."""
import argparse
import datetime
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter

from .audio import DubAudioAdapter
from .runtime import (FPS, PROJECT, CachedClipDataset, ZeroLatents, build_inputs, cached_clip_ids, load_clip, load_models, load_rows,
                      mask_weight, predict, sample, syncnet_scorer, write_mp4)
from . import lora
from .repa import Repa
from .sync_loss import N_FRAMES, SyncLoss, mel_window, melspectrogram, mouth_box


def first_item(batch):
    return batch[0]


class DomainBalancedSampler(torch.utils.data.Sampler):
    """Endless indices: domain ~ weights, then a uniform group (source video / speaker) of it, then a uniform clip of
    that group, so neither big domains nor clip-rich videos dominate. Per-rank seed (not sharded by accelerate)."""

    def __init__(self, rows, weights, seed):
        self.groups = {}
        for i, r in enumerate(rows):
            self.groups.setdefault(r["domain"], {}).setdefault(r["group"], []).append(i)
        self.domains = sorted(self.groups)
        self.groups = {d: list(self.groups[d].values()) for d in self.domains}
        self.weights = [weights.get(d, 1.0) for d in self.domains]
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            g = rng.choice(self.groups[rng.choices(self.domains, self.weights)[0]])
            yield rng.choice(g)


class EpochSampler(torch.utils.data.Sampler):
    """Endless epochs over every clip (a domain's clips listed repeats[domain] times), one shuffle per epoch shared by
    all ranks, each rank taking its contiguous slice; `start` = samples this rank already consumed (resume)."""

    def __init__(self, rows, repeats, rank, world, start=0):
        self.idx = [i for i, r in enumerate(rows) for _ in range(int(repeats.get(r["domain"], 1)))]
        self.rank, self.per, self.start = rank, len(self.idx) // world, start

    def __iter__(self):
        k = self.start
        while True:
            ep, off = divmod(k, self.per)
            perm = self.idx[:]
            random.Random(ep).shuffle(perm)
            yield from perm[self.rank * self.per + off:(self.rank + 1) * self.per]
            k = (ep + 1) * self.per


class MixSampler(torch.utils.data.Sampler):
    """Global sample g: domain ~ shares, then the next clip of that domain's current pass, a weighted random permutation
    (keys u^(1/w): every clip once per pass, high-weight clips earlier). All ranks build the same stream and rank r takes
    g = r, r + world, ...; `start` = samples this rank already consumed (resume)."""

    def __init__(self, rows, shares, clip_w, rank, world, start=0, seed=0):
        by = {}
        for i, r in enumerate(rows):
            by.setdefault(r["domain"], []).append(i)
        self.domains = [d for d in sorted(by) if shares.get(d, 1.0) > 0]
        self.idx = [np.array(by[d]) for d in self.domains]
        self.w = [np.array([clip_w[i] for i in by[d]]) for d in self.domains]
        p = np.array([shares.get(d, 1.0) for d in self.domains], dtype=np.float64)
        self.p, self.rank, self.world, self.start, self.seed = p / p.sum(), rank, world, start, seed

    def _perm(self, k, n_pass):
        rng = np.random.default_rng([self.seed, k, n_pass])
        return self.idx[k][np.argsort(-rng.random(len(self.idx[k])) ** (1 / self.w[k]))]

    def __iter__(self):
        perms = [self._perm(k, 0) for k in range(len(self.domains))]
        cur, passes, g, first = [0] * len(self.domains), [0] * len(self.domains), 0, self.start * self.world + self.rank
        for block in itertools.count():
            for k in np.random.default_rng([self.seed, 1 << 20, block]).choice(len(self.domains), 1 << 16, p=self.p):
                if cur[k] == len(perms[k]):
                    passes[k] += 1
                    perms[k], cur[k] = self._perm(k, passes[k]), 0
                i = perms[k][cur[k]]
                cur[k] += 1
                if g >= first and g % self.world == self.rank:
                    yield int(i)
                g += 1


def domain_stats(rows, weights):
    by = {}
    for r in rows:
        d = by.setdefault(r["domain"], {"clips": 0, "groups": set()})
        d["clips"] += 1
        d["groups"].add(r["group"])
    tot = sum(weights.get(d, 1.0) for d in by)
    return {d: {"clips": v["clips"], "groups": len(v["groups"]), "share": weights.get(d, 1.0) / tot} for d, v in sorted(by.items())}


def interleave_domains(rows, n):
    """First n rows taking one per domain in turn (clip_id order within a domain)."""
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r)
    queues, out = [by[d] for d in sorted(by)], []
    for k in range(max(map(len, queues), default=0)):
        out += [q[k] for q in queues if k < len(q)]
    return out[:n]


def latent_box(lat, box, max_h=36, max_w=48):
    """Latent window (lt, lb, ll, lr) around a pixel box (+2 latent px, at most max_h x max_w around its centre) and the
    pixel box clipped to it."""
    t, b, l, r = box
    lt, lb = max(0, t // 8 - 2), min(lat.shape[3], -(-b // 8) + 2)
    ll, lr = max(0, l // 8 - 2), min(lat.shape[4], -(-r // 8) + 2)
    if lb - lt > max_h:
        lt = (lt + lb - max_h) // 2
        lb = lt + max_h
    if lr - ll > max_w:
        ll = (ll + lr - max_w) // 2
        lr = ll + max_w
    return (lt, lb, ll, lr), (max(t, 8 * lt), min(b, 8 * lb), max(l, 8 * ll), min(r, 8 * lr))


@torch.no_grad()
def decode_box(pipe, lat, box):
    """Decode only the latent window around `box`: ([3, T, h, w] in [-1, 1], box local to it, clipped global box)."""
    (lt, lb, ll, lr), (t, b, l, r) = latent_box(lat, box)
    video = pipe.vae.model.decode(lat[:, :, :, lt:lb, ll:lr].to(pipe.torch_dtype), pipe.vae.scale)[0]
    pipe.vae.model.clear_cache()
    return video, (t - 8 * lt, b - 8 * lt, l - 8 * ll, r - 8 * ll), (t, b, l, r)


def sync_term(pipe, syncnet, lat, box, mel, lat_gt=None, lpips_net=None, max_h=36, max_w=48):
    """SyncNet loss on the first 16 frames decoded from x0_hat latents [1,16,5,h,w], plus (if lat_gt is given) the L1 and
    (if lpips_net is given) the LPIPS to the decoded ground truth inside the mouth box. Only the mouth box (+2 latent px, at most max_h x max_w latent px around
    its centre) is decoded: bounds decoder memory."""
    (lt, lb, ll, lr), (t, b, l, r) = latent_box(lat, box, max_h, max_w)
    decode = lambda z: pipe.vae.model.decode(z, pipe.vae.scale)
    video = torch.utils.checkpoint.checkpoint(decode, lat[:, :, :, lt:lb, ll:lr].to(pipe.torch_dtype), use_reentrant=False)[0]  # [3, 17, h, w]
    t, b, l, r = t - 8 * lt, b - 8 * lt, l - 8 * ll, r - 8 * ll
    l_sync = syncnet(video[:, 1:1 + N_FRAMES], (t, b, l, r), mel)
    if lat_gt is None:
        return l_sync, None, None
    with torch.no_grad():
        gt = decode(lat_gt[:, :, :, lt:lb, ll:lr].to(pipe.torch_dtype))[0]
    pipe.vae.model.clear_cache()
    pred, gt = video[:, 1:1 + N_FRAMES, t:b, l:r].float(), gt[:, 1:1 + N_FRAMES, t:b, l:r].float()
    l_lpips = None
    if lpips_net is not None:
        with torch.autocast("cuda", enabled=False):
            l_lpips = lpips_net(pred.clamp(-1, 1).transpose(0, 1), gt.clamp(-1, 1).transpose(0, 1)).mean()
    return l_sync, (pred - gt).abs().mean(), l_lpips


class Trainable(torch.nn.Module):
    """Single DDP entry point: frozen whisper hidden states + DiffSynth conditions -> weighted flow-matching loss (+ SyncNet loss)."""

    def __init__(self, pipe, adapter, syncnet=None, repa=None, repa_layer=8, lpips_net=None):
        super().__init__()
        self.adapter, self.dit, self.vace, self.repa = adapter, pipe.dit, pipe.vace, repa
        self.pipe, self.syncnet, self.lpips = [pipe], [syncnet], [lpips_net]  # not submodules
        self.hidden = [None]
        if repa is not None:
            adapter.blocks[repa_layer].register_forward_hook(lambda m, i, o: self.hidden.__setitem__(0, o))

    def forward(self, x0, context, vace_context, whisper_hs, num_frames, n_ref, drop_audio, lam, sync=None):
        """sync: None or {box, mel, weight, sigma_max, cross}. cross = audio from another clip: no MSE inside the mouth, low-noise t only."""
        pipe = self.pipe[0]
        tok = self.adapter.tokens(whisper_hs, num_frames, n_ref, drop_audio)
        ts = pipe.scheduler.timesteps
        if sync and sync["cross"]:
            ts = ts[pipe.scheduler.sigmas <= sync["sigma_max"]]
        timestep = ts[torch.randint(0, len(ts), (1,))].to(x0.device, x0.dtype)
        sigma = float(pipe.scheduler.sigmas[torch.argmin((pipe.scheduler.timesteps - timestep.cpu().float()).abs())])
        noise = torch.randn_like(x0)
        xt = pipe.scheduler.add_noise(x0, noise, timestep)
        target = pipe.scheduler.training_target(x0, noise, timestep)
        pred = predict(pipe, self.adapter, xt, timestep, context, vace_context, tok, grad_ckpt=True)
        w = 1 - vace_context[:, 32:].amax(1, keepdim=True).float() if sync and sync["cross"] else mask_weight(vace_context, lam)
        loss = (w * (pred.float() - target.float()) ** 2).mean() / w.mean() * pipe.scheduler.training_weight(timestep)
        l_sync, l_pix, l_lpips = None, None, None
        if sync and sigma <= sync["sigma_max"]:
            x0_hat = xt.float() - sigma * pred.float()
            pix = (sync["pix_weight"] > 0 or sync["lpips_weight"] > 0) and not sync["cross"]
            l_sync, l_pix, l_lpips = sync_term(pipe, self.syncnet[0], x0_hat[:, :, n_ref:n_ref + 5], sync["box"], sync["mel"],
                                               x0[:, :, n_ref:n_ref + 5] if pix else None,
                                               self.lpips[0] if sync["lpips_weight"] > 0 else None)
            loss = loss + sync["weight"] * l_sync
            if l_pix is not None:
                loss = loss + sync["pix_weight"] * l_pix
            if l_lpips is not None:
                loss = loss + sync["lpips_weight"] * l_lpips
        l_repa = None
        if self.repa is not None and sync and (not sync["cross"] or any(r.cross for r in self.repa)):
            l_repa = self.repa_term(x0, vace_context, n_ref, sync)
            loss = loss + sync["repa_weight"] * l_repa
        return loss, l_sync, l_pix, l_repa, l_lpips

    def repa_term(self, x0, vace_context, n_ref, sync):
        F_, hh, ww = x0.shape[2], x0.shape[3] // 2, x0.shape[4] // 2
        h = self.hidden[0][0].view(F_, hh, ww, -1)[n_ref:]
        assert h.requires_grad, "hooked hidden state is detached"
        if sync["repa_detach"]:
            h = h.detach()
        video, box, (t, b, l, r) = (None, None, (0, 0, 0, 0)) if sync["cross"] else decode_box(self.pipe[0], x0[:, :, n_ref:], sync["box"])
        loss = 0
        for repa in self.repa:
            if sync["cross"] and not repa.cross:
                continue
            if repa.kind == "art":
                tok_mask = torch.nn.functional.max_pool2d(vace_context[0, 32:, -1].amax(0)[None].float(), 2)[0] > 0
                loss = loss + repa.art_loss(h, tok_mask, self.syncnet[0], video, box, sync["mel_full"], sync["cross"])
            else:
                loss = loss + repa.dino_loss(h, video, box, (t // 16, -(-b // 16), l // 16, -(-r // 16)))
        return loss


def validate(args, accel, pipe, whisper, adapter, zero_latents, val_clips, scorer, step, writer):
    adapter.eval()
    res = {"same": [], "cross": [], "mouth_l1": []}
    for i, clip in enumerate(val_clips):
        other = val_clips[(i + 1) % len(val_clips)]
        for kind, audio in (("same", clip["audio"]), ("cross", other["audio"])):
            frames = sample(pipe, whisper, adapter, zero_latents, clip, audio, steps=args.val_steps_sampling,
                            audio_cfg=args.val_audio_cfg, seed=i, lat=clip["lat"])
            s = scorer(frames, audio)
            res[kind].append(s["conf"] if s else 0.0)
            res.setdefault(f"{kind}/{clip['domain']}", []).append(res[kind][-1])
            if kind == "same":
                m = clip["mask"]
                res["mouth_l1"].append(float(np.abs(frames[:, m].astype(np.float32) - clip["frames"][:, m]).mean()))
            if i < args.val_save:
                write_mp4(Path(args.output) / "val" / f"step{step:06d}" / f"{clip['clip_id']}_{kind}.mp4",
                          np.concatenate([clip["frames"], frames], 2), audio)
    out = {k: float(np.mean(v)) for k, v in res.items()}
    per_domain = " ".join(f"{k[5:]} {v:.2f}" for k, v in out.items() if k.startswith("same/"))
    accel.print(f"[val step {step}] sync_conf same {out['same']:.3f} cross {out['cross']:.3f} mouth_l1 {out['mouth_l1']:.2f} | same: {per_domain}", flush=True)
    for k, v in out.items():
        writer.add_scalar(f"val/{k}", v, step)
    with open(Path(args.output) / "val_log.jsonl", "a") as f:
        f.write(json.dumps({"step": step, **out}) + "\n")
    adapter.train()


def save_weights(args, raw, out, step):
    save_file({k: v.detach().float().cpu().contiguous() for k, v in raw.adapter.state_dict().items()},
              str(out / f"adapter_step{step:06d}.safetensors"))
    if args.stage == "s2":
        dit_sd = lora.merged_state_dict(raw.dit) if args.dit_lora_rank else raw.dit.state_dict()
        save_file({k: v.detach().bfloat16().cpu().contiguous() for k, v in dit_sd.items()}, str(out / f"dit_step{step:06d}.safetensors"))
        save_file({k: v.detach().bfloat16().cpu().contiguous() for k, v in raw.vace.state_dict().items()},
                  str(out / f"vace_step{step:06d}.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--stage", choices=["s1", "s2"], default="s1")
    ap.add_argument("--init_adapter", default=None, help="adapter safetensors to start from (s2)")
    ap.add_argument("--init_dit", default=None, help="DiT safetensors (continue an s2 run in a new dir)")
    ap.add_argument("--init_vace", default=None, help="VACE safetensors (continue an s2 run in a new dir)")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--max_train_clips", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_base", type=float, default=1e-5, help="lr for DiT/VACE in s2")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--mask_lambda", type=float, default=2.0)
    ap.add_argument("--p_drop_text", type=float, default=1.0,
                    help="prob. of the empty-prompt context; 1 = never condition on the cached caption (inference has no text)")
    ap.add_argument("--data", default="h3", help="comma list of sources: h3, hallo3, ls_real, astri")
    ap.add_argument("--domain_weights", default="",
                    help="e.g. H3=1,TalkVid=2 (domains: H3 hallo3 HDTF TalkVid SpeakerVid VoxCeleb2; default 1 each). "
                         "Multi-source runs sample domain -> source video -> clip")
    ap.add_argument("--sampling", choices=["balanced", "epoch", "mix"], default="balanced",
                    help="epoch: every clip once per epoch, --domain_weights are integer repeat factors (e.g. VoxCeleb2=8); "
                         "mix: --domain_weights are sampling shares, clips of a domain cycle without replacement, ordered by --clip_weights")
    ap.add_argument("--clip_weights", default="", help="mix: json {clip_id: weight}; clips not listed are dropped")
    ap.add_argument("--p_drop_audio", type=float, default=0.1)
    ap.add_argument("--p_drop_ref", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_clips", type=int, default=8)
    ap.add_argument("--val_save", type=int, default=4)
    ap.add_argument("--val_steps_sampling", type=int, default=30)
    ap.add_argument("--val_audio_cfg", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latents_name", default="latents.pt", help="filename under each clip dir (s03 cache)")
    ap.add_argument("--sync_weight", type=float, default=0.0, help="StableSyncNet loss weight (0 = off)")
    ap.add_argument("--sync_sigma_max", type=float, default=0.8, help="SyncNet loss only when sigma <= this")
    ap.add_argument("--p_cross", type=float, default=0.0, help="prob. of driving with another clip's audio (SyncNet + non-mouth MSE)")
    ap.add_argument("--pix_weight", type=float, default=0.0, help="L1 on the decoded mouth box vs decoded GT (same-audio sync steps)")
    ap.add_argument("--repa", default="none", help="mouth-token representation alignment: none or a comma list of art, dino")
    ap.add_argument("--repa_cross", action="store_true", help="art: align cross-audio steps to the driving audio's SyncNet embedding")
    ap.add_argument("--repa_temporal", action="store_true", help="art: also align window-to-window differences")
    ap.add_argument("--repa_layer", type=int, default=8)
    ap.add_argument("--repa_weight", type=float, default=0.1)
    ap.add_argument("--repa_lr", type=float, default=1e-4)
    ap.add_argument("--repa_warmup", type=int, default=50, help="steps that only train the projector (hidden state detached)")
    ap.add_argument("--lpips_weight", type=float, default=0.0, help="LPIPS(vgg) on the decoded mouth box vs decoded GT (same-audio aux steps)")
    ap.add_argument("--dit_lora_rank", type=int, default=0, help="s2 + --freeze_dit: train a LoRA of this rank on the DiT (merged on save)")
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    ap.add_argument("--lr_schedule", choices=["const", "cosine"], default="const", help="cosine: decay to 0.1x lr at max_steps")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trained weights (0 = off); saved to <output>_ema/")
    ap.add_argument("--skip_val0", action="store_true")
    ap.add_argument("--freeze_dit", action="store_true", help="s2: keep the DiT frozen (bf16), train adapter + VACE only")
    args = ap.parse_args()

    accel = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum, step_scheduler_with_optimizer=False,
                                   kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True),
                                                    accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234 + accel.process_index)
    random.seed(1234 + accel.process_index)

    pipe, whisper = load_models(accel.device)
    pipe.scheduler.set_timesteps(1000, training=True)
    adapter = DubAudioAdapter().to(accel.device)
    if args.init_adapter:
        adapter.load_state_dict(load_file(args.init_adapter))
    groups = [{"params": list(adapter.parameters()), "lr": args.lr}]
    if args.stage == "s2":
        pipe.vace.requires_grad_(True).float()
        base = list(pipe.vace.parameters())
        if not args.freeze_dit:
            pipe.dit.requires_grad_(True).float()
            base += list(pipe.dit.parameters())
        groups.append({"params": base, "lr": args.lr_base})
    if args.init_dit:
        pipe.dit.load_state_dict(load_file(args.init_dit))
    if args.init_vace:
        pipe.vace.load_state_dict(load_file(args.init_vace))
    if args.dit_lora_rank:
        assert args.stage == "s2" and args.freeze_dit
        groups.append({"params": lora.inject(pipe.dit, args.dit_lora_rank, args.dit_lora_rank), "lr": args.lora_lr})
    ema_params = [p for g in groups for p in g["params"]]  # weights that end up in the checkpoints (not the repa projectors)
    # SyncNet decode step: needed by the sync loss and by every loss on the decoded mouth / mouth tokens
    aux = args.sync_weight > 0 or args.repa != "none" or args.pix_weight > 0 or args.lpips_weight > 0
    repa = None
    if args.repa != "none":
        repa = torch.nn.ModuleList([Repa(k, temporal=args.repa_temporal, cross=args.repa_cross).to(accel.device) for k in args.repa.split(",")])
        groups.append({"params": [p for r in repa for p in r.proj.parameters()], "lr": args.repa_lr})
    lpips_net = None
    if args.lpips_weight > 0:
        import lpips
        torch.hub.set_dir(os.environ.get("VIDDUB_TORCH_HUB", str(PROJECT / "weights/torch_hub")))
        lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(accel.device).eval().requires_grad_(False)
    model = Trainable(pipe, adapter, SyncLoss().to(accel.device) if aux else None, repa, args.repa_layer, lpips_net)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    def lr_lambda(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        if args.lr_schedule == "cosine":
            return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.max_steps - args.warmup))))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = [p.detach().float().clone() for p in ema_params] if args.ema > 0 else None

    sources = args.data.split(",")
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.domain_weights.split(",") if kv)}
    balanced = len(sources) > 1 or bool(weights)
    cached = cached_clip_ids(sources)
    train_rows = [r for r in load_rows("train", sources) if r["clip_id"] in cached][:args.max_train_clips]
    clip_w = None
    if args.sampling == "mix" and args.clip_weights:
        cw = json.load(open(args.clip_weights))
        accel.print(f"clip weights {args.clip_weights}: keep {sum(r['clip_id'] in cw for r in train_rows)}/{len(train_rows)} train clips")
        train_rows = [r for r in train_rows if r["clip_id"] in cw]
        clip_w = [float(cw[r["clip_id"]]) for r in train_rows]
    zero_latents = ZeroLatents(pipe)
    zero_latents.empty_context()
    dataset = CachedClipDataset(train_rows, args.num_frames, args.latents_name)
    if balanced:
        model, opt, sched = accel.prepare(model, opt, sched)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=first_item,
                                             num_workers=args.workers, persistent_workers=True, prefetch_factor=2)
        model, opt, loader, sched = accel.prepare(model, opt, loader, sched)
    accel.register_for_checkpointing(sched)

    step = 0
    state_dir = out / "state"
    if state_dir.exists():
        accel.load_state(str(state_dir))
        step = json.load(open(out / "step.json"))["step"]
        if ema is not None and (state_dir / "ema.pt").exists():
            ema = [t.to(accel.device) for t in torch.load(state_dir / "ema.pt")]
        accel.print(f"resumed from step {step}")
    if args.sampling == "epoch":
        clips = {}
        for r in train_rows:
            clips[r["domain"]] = clips.get(r["domain"], 0) + 1
        stats = domain_stats(train_rows, {d: n * int(weights.get(d, 1)) for d, n in clips.items()})
    else:
        stats = domain_stats(train_rows, weights)
    if balanced:
        if args.sampling == "mix":
            sampler = MixSampler(train_rows, weights, clip_w or [1.0] * len(train_rows), accel.process_index,
                                 accel.num_processes, step * args.grad_accum)
        elif args.sampling == "epoch":
            sampler = EpochSampler(train_rows, weights, accel.process_index, accel.num_processes, step * args.grad_accum)
            accel.print(f"epoch sampling: {sampler.per * accel.num_processes} samples/epoch, "
                        f"{sampler.per / args.grad_accum:.0f} steps/epoch")
        else:
            sampler = DomainBalancedSampler(train_rows, weights, seed=1234 + accel.process_index + 1000003 * step)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=first_item,
                                             num_workers=args.workers, persistent_workers=True, prefetch_factor=2)
        for d, s in stats.items():
            accel.print(f"  {d:<11} clips {s['clips']:>7} groups {s['groups']:>6} share {s['share']:.3f} "
                        f"steps/epoch {s['clips'] / (s['share'] * accel.num_processes * args.grad_accum):.0f}")

    writer = SummaryWriter(str(out / "tb")) if accel.is_main_process else None
    scorer, val_clips = None, None
    if accel.is_main_process:
        scorer = syncnet_scorer(accel.device)
        val_rows = interleave_domains(load_rows("val", sources), args.val_clips)
        val_clips = [load_clip(r, args.num_frames) for r in val_rows]
        for r, c in zip(val_rows, val_clips):
            c["lat"] = torch.load(PROJECT / "data/clips" / r["dir"] / args.latents_name) if r["clip_id"] in cached else None
            c["domain"] = r["domain"]
        json.dump(vars(args) | {"num_train_clips": len(train_rows), "world_size": accel.num_processes, "domains": stats,
                                "val_clip_ids": [r["clip_id"] for r in val_rows]}, open(out / "args.json", "w"), indent=1)
    accel.print(f"train clips {len(train_rows)}, world {accel.num_processes}, trainable params "
                f"{sum(p.numel() for g in groups for p in g['params']) / 1e6:.1f}M")

    raw = accel.unwrap_model(model)
    t0, losses, sync_losses, pix_losses, repa_losses, lpips_losses, last_val = time.time(), [], [], [], [], [], (0 if args.skip_val0 else -1)
    prev_audio = None
    while step < args.max_steps:
        for clip in loader:
            if step % args.val_every == 0 and step != last_val:
                if accel.is_main_process:
                    validate(args, accel, pipe, whisper, raw.adapter, zero_latents, val_clips, scorer, step, writer)
                accel.wait_for_everyone()
                last_val = step

            drop_ref = random.random() < args.p_drop_ref
            x0, vace_context, context = build_inputs(clip["latents"], clip["mask"], args.num_frames, zero_latents, drop_ref=drop_ref,
                                                     context=None if random.random() < args.p_drop_text else clip["latents"]["context"])
            cross = aux and prev_audio is not None and random.random() < args.p_cross
            audio = prev_audio if cross else clip["audio"]
            prev_audio = clip["audio"]
            hs = whisper([audio], accel.device, torch.bfloat16)
            drop_audio = torch.tensor([not cross and random.random() < args.p_drop_audio])
            sync = None
            if aux and not drop_audio.item():
                sync = {"box": mouth_box(clip["mask"], 0.1), "mel": mel_window(melspectrogram(audio), 1 / FPS).to(accel.device),
                        "weight": args.sync_weight, "sigma_max": args.sync_sigma_max, "cross": cross, "pix_weight": args.pix_weight, "lpips_weight": args.lpips_weight,
                        "repa_weight": args.repa_weight, "repa_detach": step < args.repa_warmup, "mel_full": melspectrogram(audio) if repa is not None else None}
            with accel.accumulate(model), torch.autocast("cuda", dtype=torch.bfloat16):
                loss, l_sync, l_pix, l_repa, l_lpips = model(x0, context, vace_context, hs, args.num_frames, 0 if drop_ref else 1, drop_audio, args.mask_lambda, sync)
                accel.backward(loss)
                if accel.sync_gradients:
                    norm = accel.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                    if not torch.isfinite(norm):  # same on every rank (post all-reduce): skip the update everywhere
                        opt.zero_grad(set_to_none=True)
                        accel.print(f"step {step}: non-finite grad norm, update skipped", flush=True)
                opt.step()
                if accel.sync_gradients:
                    sched.step()
                    if ema is not None:
                        with torch.no_grad():
                            torch._foreach_lerp_(ema, [p.detach().float() for p in ema_params], 1 - args.ema)
                opt.zero_grad(set_to_none=True)
            if l_sync is not None:
                pipe.vae.model.clear_cache()  # its feature cache keeps the decoder graph alive otherwise
                sync_losses.append(l_sync.item())
            if l_pix is not None:
                pix_losses.append(l_pix.item())
            if l_repa is not None:
                repa_losses.append(l_repa.item())
            if l_lpips is not None:
                lpips_losses.append(l_lpips.item())
            losses.append(loss.item())
            if not accel.sync_gradients:
                continue
            step += 1

            if step % args.log_every == 0:
                l = torch.tensor(np.mean(losses), device=accel.device)
                l = accel.reduce(l, "mean").item()
                ls = torch.tensor([sum(sync_losses), len(sync_losses)], device=accel.device, dtype=torch.float64)
                ls = accel.reduce(ls, "sum")
                ls = (ls[0] / ls[1].clamp(min=1)).item()
                lp = accel.reduce(torch.tensor([sum(pix_losses), len(pix_losses)], device=accel.device, dtype=torch.float64), "sum")
                lp = (lp[0] / lp[1].clamp(min=1)).item()
                lr_ = accel.reduce(torch.tensor([sum(repa_losses), len(repa_losses)], device=accel.device, dtype=torch.float64), "sum")
                lr_ = (lr_[0] / lr_[1].clamp(min=1)).item()
                lq = accel.reduce(torch.tensor([sum(lpips_losses), len(lpips_losses)], device=accel.device, dtype=torch.float64), "sum")
                lq = (lq[0] / lq[1].clamp(min=1)).item()
                dt = (time.time() - t0) / args.log_every
                accel.print(f"step {step} loss {l:.4f} sync {ls:.3f} pix {lp:.4f} repa {lr_:.4f} lpips {lq:.4f} lr {sched.get_last_lr()[0]:.2e} {dt:.2f}s/step "
                            f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
                if writer:
                    writer.add_scalar("train/loss", l, step)
                    writer.add_scalar("train/sync", ls, step)
                    writer.add_scalar("train/pix", lp, step)
                    writer.add_scalar("train/repa", lr_, step)
                    writer.add_scalar("train/lpips", lq, step)
                    writer.add_scalar("train/sec_per_step", dt, step)
                t0, losses, sync_losses, pix_losses, repa_losses, lpips_losses = time.time(), [], [], [], [], []

            if step % args.save_every == 0 or step == args.max_steps:
                accel.save_state(str(state_dir))
                if accel.is_main_process:
                    json.dump({"step": step}, open(out / "step.json", "w"))
                    save_weights(args, raw, out, step)
                    if ema is not None:
                        torch.save(ema, out / "state/ema.pt")
                        live = [p.detach().clone() for p in ema_params]
                        with torch.no_grad():
                            torch._foreach_copy_(ema_params, ema)
                        (out.parent / f"{out.name}_ema").mkdir(exist_ok=True)
                        save_weights(args, raw, out.parent / f"{out.name}_ema", step)
                        with torch.no_grad():
                            torch._foreach_copy_(ema_params, live)
                        del live
            if step >= args.max_steps:
                break
    if accel.is_main_process:
        validate(args, accel, pipe, whisper, raw.adapter, zero_latents, val_clips, scorer, step, writer)
    accel.wait_for_everyone()


if __name__ == "__main__":
    main()
