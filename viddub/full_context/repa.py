"""Representation alignment of DiT mouth tokens (hidden states after one DiT block) to frozen encoders.

art:  StableSyncNet embedding space. 17 windows of 16 frames (start 4k-3 -> latent frames k..k+3); a projector maps the
      concatenated mouth-pooled tokens of those 4 latent frames to 2048-d. Targets: the visual embedding of the ground-truth
      window (same audio) or, with `cross`, the audio embedding of the driving audio window (counterfactual steps, no GT
      video exists). `temporal` adds a loss on window-to-window differences.
dino: per-token alignment of the mouth-box tokens of every latent frame to DINOv2 patch features of the GT mouth box.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import FPS, WEIGHTS
from .sync_loss import mel_window

N_WIN, WIN_LAT = 17, 4
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class Repa(nn.Module):
    def __init__(self, kind, dim=1536, temporal=False, cross=False):
        super().__init__()
        self.kind, self.temporal, self.cross = kind, temporal and kind == "art", cross and kind == "art"
        if kind == "art":
            self.proj = mlp(WIN_LAT * dim, 2048, 2048)
        else:
            from transformers import Dinov2Model
            self.proj = mlp(dim, 2048, 768)
            self.dino = [Dinov2Model.from_pretrained(str(WEIGHTS / "dinov2-base")).eval().requires_grad_(False)]  # not a submodule

    def to(self, *a, **kw):
        if self.kind == "dino":
            self.dino[0].to(*a, **kw)
        return super().to(*a, **kw)

    def art_loss(self, h, tok_mask, syncnet, video_gt, box, mel, cross):
        """h: [F, 32, 32, C] (latent frames without ref); tok_mask: [32, 32] bool; video_gt: [3, 81, h, w] or None."""
        f = h[:, tok_mask].mean(1)  # F, C
        x = torch.stack([f[k:k + WIN_LAT].flatten() for k in range(1, N_WIN + 1)])
        p = F.normalize(self.proj(x).float(), dim=-1)
        with torch.no_grad():
            if cross:
                m = torch.cat([mel_window(mel, (4 * k - 3) / FPS) for k in range(1, N_WIN + 1)]).to(h.device)
                t = F.normalize(syncnet.net.audio_encoder(m).flatten(1).float(), dim=-1)
            else:
                v = torch.cat([syncnet.crop(video_gt[:, 4 * k - 3:4 * k + 13], box) for k in range(1, N_WIN + 1)])
                t = F.normalize(syncnet.net.visual_encoder(v).flatten(1).float(), dim=-1)
        loss = (1 - (p * t).sum(-1)).mean()
        if self.temporal:
            loss = loss + ((p[1:] - p[:-1]) - (t[1:] - t[:-1])).pow(2).sum(-1).mean()
        return loss

    def dino_loss(self, h, video_gt, box, tok_box):
        """h: [F, 32, 32, C]; video_gt: [3, 81, h, w] in [-1, 1]; box: pixel box in video_gt; tok_box: token box in h."""
        t, b, l, r = box
        frames = video_gt[:, ::4, t:b, l:r].transpose(0, 1).float()  # F, 3, bh, bw
        with torch.no_grad():
            x = F.interpolate((frames + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - DINO_MEAN.to(x)) / DINO_STD.to(x)
            target = F.normalize(self.dino[0](pixel_values=x).last_hidden_state[:, 1:].float(), dim=-1)  # F, 256, 768
        tt, tb, tl, tr = tok_box
        z = h[:, tt:tb, tl:tr].permute(0, 3, 1, 2).float()
        z = F.interpolate(z, size=(16, 16), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)  # F, 256, C
        p = F.normalize(self.proj(z.to(h.dtype)).float(), dim=-1)
        return (1 - (p * target).sum(-1)).mean()
