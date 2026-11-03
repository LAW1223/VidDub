"""Paired GT quality and audio supervision on a causal, trailing decoded window.

Frozen encoders are external to the trainable generator. Decode the complete
available temporal prefix, so an arbitrary latent slice is never treated as a
new VAE first frame. Spatial crop is shared by prediction and pixel GT.
"""
import torch
import numpy as np
import os
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sync_mel_window(mel, start_seconds):
    from viddub.full_context.sync_loss import mel_window
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy()
    if not isinstance(mel, np.ndarray):
        raise TypeError(f'Expected mel tensor/array, got {type(mel).__name__}')
    return mel_window(mel, start_seconds)


def trailing_window(latent_frames, size=16):
    frames = 4 * (latent_frames - 1) + 1
    return max(0, frames - size), frames


def roi_from_mask(mask, latent_height, latent_width, margin=2, max_h=36, max_w=48):
    ys, xs = torch.where(mask)
    if not len(ys):
        raise ValueError('Empty generation mask')
    t, b, l, r = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    lt, lb = max(0, t // 8 - margin), min(latent_height, (b + 7) // 8 + margin)
    ll, lr = max(0, l // 8 - margin), min(latent_width, (r + 7) // 8 + margin)
    if lb - lt > max_h:
        lt = (lt + lb - max_h) // 2; lb = lt + max_h
    if lr - ll > max_w:
        ll = (ll + lr - max_w) // 2; lr = ll + max_w
    return lt, lb, ll, lr


class GTQuality:
    def __init__(self, pipe, cfg, device):
        from .runtime import local_imports
        local_imports()
        self.pipe, self.cfg = pipe, cfg
        self.sync = self.lpips = None
        if cfg.get('sync_weight', 0) or cfg.get('art_weight', 0):
            from viddub.full_context import sync_loss
            from .runtime import PROJECT
            sync_loss.LS_ROOT = Path(os.environ.get(
                'VIDDUB_LATENTSYNC_ROOT', PROJECT / 'weights' / 'LatentSync'))
            self.sync = sync_loss.SyncLoss().to(device).eval().requires_grad_(False)
        if cfg.get('lpips_weight', 0):
            from viddub.full_context.runtime import PROJECT
            import lpips
            torch.hub.set_dir(str(Path(os.environ.get(
                'VIDDUB_TORCH_HUB', PROJECT / 'weights' / 'torch_hub'))))
            self.lpips = lpips.LPIPS(net='vgg', verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def art_target(self, conditions, end):
        first,last=trailing_window(end-conditions['num_ref'])
        if last-first!=16:return None
        device=conditions['vace_context'].device
        mask=conditions['pixel_mask'].to(device).bool()
        ys,xs=torch.where(mask)
        box=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1)
        if conditions.get('cross',False):
            mel=sync_mel_window(conditions['mel'],first/self.cfg['fps']).to(device)
            return self.sync.net.audio_encoder(mel).flatten(1).float()
        gt=conditions['gt_frames'][first:last].to(device).float().permute(3,0,1,2)/127.5-1
        return self.sync.net.visual_encoder(self.sync.crop(gt,box)).flatten(1).float()

    def decode(self, z):
        vae = self.pipe.vae
        # Checkpoint forward and recompute each own a fresh VAE feature cache.
        vae.model.clear_cache()
        out = vae.model.decode(z.to(self.pipe.torch_dtype), vae.scale)
        vae.model.clear_cache()
        return out

    def decode_window(self, z, grad_begin, first, last):
        """Exact prefix decode, but frozen GT-history latents require no graph.

        The original conv2 has temporal kernel 1. Processing history under
        no_grad is equivalent to decoding a detached history with frozen VAE
        weights, and avoids retaining dozens of irrelevant frame graphs.
        """
        vae = self.pipe.vae
        net = vae.model
        if net.conv2.kernel_size[0] != 1:
            raise ValueError('Window decode assumes temporal conv2 kernel 1')
        net.clear_cache()
        z = z.to(self.pipe.torch_dtype)
        mean, inv_std = (s.to(z) for s in vae.scale)
        values = []
        grad_enabled = torch.is_grad_enabled()
        for i in range(z.shape[2]):
            with torch.set_grad_enabled(grad_enabled and i >= grad_begin):
                latent = z[:, :, i:i+1] / inv_std.view(1,-1,1,1,1) + mean.view(1,-1,1,1,1)
                x = net.conv2(latent)
                decoded, net._feat_map, net._conv_idx = net.decoder(
                    x, feat_cache=net._feat_map, feat_idx=[0])
                frame_start = 0 if i == 0 else 4*i-3
                frame_end = frame_start + decoded.shape[2]
                lo, hi = max(first, frame_start), min(last, frame_end)
                if lo < hi:values.append(decoded[:, :, lo-frame_start:hi-frame_start])
        net.clear_cache()
        return torch.cat(values, 2)

    def __call__(self, estimate_prefix, conditions, start, end, sigma):
        zero = estimate_prefix.new_zeros((), dtype=torch.float32)
        stats = {'pixel': zero, 'lpips': zero, 'sync': zero, 'keep': zero,
                 'aux_active': zero, 'sync_active': zero}
        q_active = float(sigma) <= self.cfg['aux_sigma_max']
        sync_active = float(sigma) <= self.cfg['sync_sigma_max']
        if not q_active and not sync_active:
            return zero, stats
        if conditions.get('gt_frames') is None:
            raise ValueError('GT auxiliary loss requires original pixel frames')
        n_ref = conditions['num_ref']
        z = estimate_prefix[:, :, n_ref:end]
        if z.shape[2] < 1:return zero, stats
        mask = conditions['pixel_mask'].to(z.device).bool()
        lt, lb, ll, lr = roi_from_mask(mask, z.shape[-2], z.shape[-1],
                                      max_h=self.cfg['aux_max_latent_h'],
                                      max_w=self.cfg['aux_max_latent_w'])
        first, last = trailing_window(z.shape[2])
        z_roi = z[:, :, :, lt:lb, ll:lr]
        grad_begin = max(0, start - n_ref)
        decoded = checkpoint(self.decode_window, z_roi, grad_begin, first, last,
                             use_reentrant=False)[0].float()
        gt = conditions['gt_frames'][first:last, lt*8:lb*8, ll*8:lr*8].to(z.device).float()
        gt = gt.permute(3, 0, 1, 2) / 127.5 - 1
        if decoded.shape != gt.shape:
            raise ValueError(f'Decoded/GT window mismatch {decoded.shape} != {gt.shape}')
        local_mask = mask[lt*8:lb*8, ll*8:lr*8]
        # Q applies only where paired GT exists; current first-wave AR is same.
        if q_active and not conditions.get('cross', False):
            if self.cfg.get('pixel_weight', 0):
                stats['pixel'] = (decoded - gt).abs()[:, :, local_mask].mean()
            if self.lpips is not None:
                # Four evenly spaced frames keep memory bounded, including end.
                ids = torch.linspace(0, last-first-1, min(4, last-first), device=z.device).round().long()
                ys, xs = torch.where(local_mask)
                box = (int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1)
                t,b,l,r = box
                pred = decoded[:, ids, t:b, l:r].transpose(0,1)
                target = gt[:, ids, t:b, l:r].transpose(0,1)
                with torch.autocast(z.device.type, enabled=False):
                    stats['lpips'] = self.lpips(pred.clamp(-1,1), target).mean()
        if q_active and self.cfg.get('keep_weight', 0) and (~local_mask).any():
            stats['keep'] = (decoded - gt).abs()[:, :, ~local_mask].mean()
        dropped = conditions.get('drop_audio')
        if (sync_active and self.sync is not None and last-first == 16 and
                not (dropped is not None and bool(dropped.any()))):
            ys, xs = torch.where(local_mask)
            box = (int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1)
            mel = sync_mel_window(conditions['mel'], first / self.cfg['fps']).to(z.device)
            stats['sync'] = self.sync(decoded, box, mel)
            stats['sync_active'] = zero + 1
        stats['aux_active'] = zero + float(q_active)
        loss = sum(self.cfg.get(key+'_weight', 0) * stats[key] for key in ['pixel','lpips','sync','keep'])
        return loss, stats
