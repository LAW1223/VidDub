"""Shared pieces for Wan-Dub training / validation / inference."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import decord
import imageio_ffmpeg
import numpy as np
import soundfile
import torch
from einops import rearrange
from PIL import Image

from viddub.foundation.pipelines.wan_video import ModelConfig, WanVideoPipeline, WanVideoUnit_PromptEmbedder, model_fn_wan_video
from .audio import WhisperFeatures

PROJECT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("VIDDUB_WEIGHT_ROOT", PROJECT / "weights"))
DATA_ROOT = Path(os.environ.get("VIDDUB_DATA_ROOT", PROJECT / "data"))
CLIPS = DATA_ROOT / "clips"
VACE_DIR = WEIGHTS / "Wan2.1-VACE-1.3B"
SR = 16000
FPS = 24
TEXT_LEN = 512
MANIFEST = DATA_ROOT / "manifest"
SOURCES = {"h3": MANIFEST, "hallo3": MANIFEST / "hallo3", "ls_real": MANIFEST / "ls_real", "astri": MANIFEST / "astri"}


def load_rows(split, sources=("h3",)):
    """s02 rows of `sources`, each tagged with source / domain / group. val = md5(group) % 100 == 0, so no source video
    (HDTF person, VoxCeleb2 speaker, YouTube id) straddles train/val; groups shared with the public benches are dropped
    from both. Optional group maps are stored with the data manifests."""
    exclude_path = MANIFEST / "bench_exclude.json"
    excluded = ({g for v in json.load(open(exclude_path)).values() for g in v}
                if exclude_path.is_file() else set())
    out = []
    for src in sources:
        groups = json.load(open(SOURCES[src] / "groups.json")) if src != "h3" else None
        for p in sorted((SOURCES[src] / "s02").glob("shard_*.jsonl")):
            for l in open(p):
                if not l.strip():
                    continue
                r = json.loads(l)
                if "error" in r:
                    continue
                r["source"] = src
                r["domain"], r["group"] = groups[r["id"]] if groups else ("H3", r["id"])
                if r["group"] not in excluded and (int(hashlib.md5(r["group"].encode()).hexdigest(), 16) % 100 == 0) == (split == "val"):
                    out.append(r)
    return sorted(out, key=lambda r: r["clip_id"])


def cached_clip_ids(sources=("h3",)):
    ids = set()
    for src in sources:
        for p in (SOURCES[src] / "s03").glob("shard_*.txt"):
            ids.update(p.read_text().split())
    return ids


def load_clip(row, num_frames=81):
    d = CLIPS / row["dir"]
    frames = decord.VideoReader(str(d / "video.mp4"), num_threads=1).get_batch(list(range(num_frames))).asnumpy()
    audio, _ = soundfile.read(d / "audio.wav", dtype="float32")
    return {"frames": frames, "mask": np.array(Image.open(d / "mask.png")) > 127, "ref": np.array(Image.open(d / "ref.png").convert("RGB")),
            "audio": audio[:round(num_frames / FPS * SR)], "prompt": row["prompt"], "clip_id": row["clip_id"]}


class CachedClipDataset(torch.utils.data.Dataset):
    """Precomputed latents (s03) + mask + audio."""

    def __init__(self, rows, num_frames, latents_name="latents.pt"):
        self.rows, self.num_frames, self.latents_name = rows, num_frames, latents_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = CLIPS / self.rows[i]["dir"]
        audio, _ = soundfile.read(d / "audio.wav", dtype="float32")
        return {"latents": torch.load(d / self.latents_name), "mask": np.array(Image.open(d / "mask.png")) > 127,
                "audio": audio[:round(self.num_frames / FPS * SR)], "clip_id": self.rows[i]["clip_id"]}


def load_models(device):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=str(VACE_DIR / "diffusion_pytorch_model.safetensors")),
                       ModelConfig(path=str(VACE_DIR / "models_t5_umt5-xxl-enc-bf16.pth")),
                       ModelConfig(path=str(VACE_DIR / "Wan2.1_VAE.pth"))],
        tokenizer_config=ModelConfig(path=str(VACE_DIR / "google/umt5-xxl")),
    )
    for m in (pipe.text_encoder, pipe.dit, pipe.vace, pipe.vae):
        m.requires_grad_(False).eval()
    whisper = WhisperFeatures(str(WEIGHTS / "whisper-small")).to(device, torch.bfloat16)
    return pipe, whisper


def to_video_tensor(frames, pipe):
    """uint8 [T,H,W,3] -> [1,3,T,H,W] in [-1,1]."""
    return (torch.from_numpy(frames).to(pipe.device, pipe.torch_dtype).permute(3, 0, 1, 2)[None] / 127.5 - 1)


@torch.no_grad()
def encode_text(pipe, prompt):
    """T5 context trimmed to its real length: [L, 4096]."""
    emb = WanVideoUnit_PromptEmbedder().encode_prompt(pipe, prompt)[0]
    n = int((emb.abs().sum(-1) > 0).sum())
    return emb[:max(n, 1)]


@torch.no_grad()
def compute_latents(pipe, clip):
    """VAE latents of gt video, masked (inactive) video, reference frame; plus T5 context. Matches WanVideoUnit_VACE."""
    video = to_video_tensor(clip["frames"], pipe)
    keep = torch.from_numpy(~clip["mask"]).to(video)[None, None, None]
    return {"x0": pipe.vae.encode(video, device=pipe.device)[0].cpu(),
            "inactive": pipe.vae.encode(video * keep, device=pipe.device)[0].cpu(),
            "ref": pipe.vae.encode(to_video_tensor(clip["ref"][None], pipe), device=pipe.device)[0].cpu(),
            "context": encode_text(pipe, clip["prompt"]).cpu()}


class ZeroLatents:
    """VAE latent of an all-zero video (= VACE 'reactive' part, since the masked region is blanked)."""

    def __init__(self, pipe):
        self.pipe, self.cache, self._empty_context = pipe, {}, None

    def empty_context(self):
        """T5 context of "": the only text condition (dubbing is audio-driven; inference never has a prompt)."""
        if self._empty_context is None:
            cached = getattr(self.pipe, "_wan_dub_empty_context", None)
            self._empty_context = (cached.to(self.pipe.device, self.pipe.torch_dtype)
                                   if cached is not None else encode_text(self.pipe, ""))
        return self._empty_context

    def __call__(self, num_frames, h, w):
        key = (num_frames, h, w)
        if key not in self.cache:
            z = torch.zeros(1, 3, num_frames, h, w, device=self.pipe.device, dtype=self.pipe.torch_dtype)
            self.cache[key] = self.pipe.vae.encode(z, device=self.pipe.device)[0]
        return self.cache[key]


def build_inputs(lat, mask, num_frames, zero_latents, drop_ref=False, context=None):
    """Cached latents -> (x0 [1,16,F,h,w], vace_context [1,96,F,h,w], context [1,512,4096]) as WanVideoUnit_VACE would.
    context=None -> empty-prompt context (the cached caption context lat["context"] is not used unless passed explicitly)."""
    device, dtype = zero_latents.pipe.device, zero_latents.pipe.torch_dtype
    n_lat = (num_frames - 1) // 4 + 1
    x0 = lat["x0"][:, :n_lat].to(device, dtype)
    H, W = mask.shape
    vace = torch.cat([lat["inactive"][:, :n_lat].to(device, dtype), zero_latents(num_frames, H, W)[:, :n_lat]], 0)
    m = rearrange(torch.from_numpy(mask).to(device, dtype), "(H P) (W Q) -> (P Q) 1 H W", P=8, Q=8).expand(-1, n_lat, -1, -1)
    if not drop_ref:
        ref = lat["ref"].to(device, dtype)
        x0 = torch.cat([ref, x0], 1)
        vace = torch.cat([torch.cat([ref, torch.zeros_like(ref)], 0), vace], 1)
        m = torch.cat([torch.zeros_like(m[:, :1]), m], 1)
    ctx = zero_latents.empty_context() if context is None else context
    ctx = torch.cat([ctx, ctx.new_zeros(TEXT_LEN - len(ctx), ctx.shape[1])]).to(device, dtype)[None]
    return x0[None], torch.cat([vace, m], 0)[None], ctx


def predict(pipe, adapter, latents, timestep, context, vace_context, audio_tokens, grad_ckpt=False):
    return model_fn_wan_video(dit=pipe.dit, vace=pipe.vace, latents=latents, timestep=timestep, context=context,
                              vace_context=vace_context, vace_scale=1, audio_adapter=adapter, audio_tokens=audio_tokens,
                              use_gradient_checkpointing=grad_ckpt)


def mask_weight(vace_context, lam):
    """Latent-resolution mouth mask from VACE context (channels 32: are the 8x8-unshuffled pixel mask)."""
    return 1 + lam * vace_context[:, 32:].amax(1, keepdim=True).float()


@torch.no_grad()
def sample(pipe, whisper, adapter, zero_latents, clip, audio, steps=30, audio_cfg=1.0, seed=0, lat=None, drop_ref=False, fps=FPS):
    """Returns uint8 frames [T,H,W,3] generated for `clip` (masked, with ref) driven by `audio`."""
    device, dtype = pipe.device, pipe.torch_dtype
    n = len(clip["frames"])
    n_ref = 0 if drop_ref else 1
    lat = compute_latents(pipe, clip) if lat is None else lat
    x0, vace_context, context = build_inputs(lat, clip["mask"], n, zero_latents, drop_ref=drop_ref)
    hs = whisper([audio], device, dtype)
    latents = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed)).to(device, dtype)
    pipe.scheduler.set_timesteps(steps, shift=5.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = adapter.tokens(hs, n, n_ref, torch.tensor([False]), fps)
        tok_null = adapter.tokens(hs, n, n_ref, torch.tensor([True]), fps)
        for t in pipe.scheduler.timesteps:
            tt = t.unsqueeze(0).to(dtype=dtype, device=device)
            pred = predict(pipe, adapter, latents, tt, context, vace_context, tok)
            if audio_cfg != 1.0:
                pred_u = predict(pipe, adapter, latents, tt, context, vace_context, tok_null)
                pred = pred_u + audio_cfg * (pred - pred_u)
            latents = pipe.scheduler.step(pred, t, latents)
    video = pipe.vae.decode(latents[:, :, n_ref:].to(dtype), device=device)
    pipe.scheduler.set_timesteps(1000, training=True)
    return ((video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def write_mp4(path, frames, audio=None, fps=FPS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    tmp_wav = None
    if audio is not None:
        tmp_wav = str(path) + ".wav"
        soundfile.write(tmp_wav, audio, SR)
        # 48k stereo: 16k mono AAC plays silent in many embedded players
        # Pad resampled AAC to the video duration before -shortest. Without
        # apad, fractional/native frame rates can make audio end a fraction
        # early and FFmpeg silently drops the final video frame.
        cmd += [
            "-i", tmp_wav, "-af", "apad",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest",
        ]
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    if tmp_wav:
        Path(tmp_wav).unlink()


def syncnet_scorer(device):
    from data.s01_analyze import detect_landmarks, load_syncnet, syncnet_crops, syncnet_score
    net = load_syncnet(device)

    def score(frames, audio):
        lm, nfaces = detect_landmarks(frames)
        if (nfaces > 0).mean() < 0.5:
            return None
        crops = syncnet_crops(frames, lm)
        idx25 = (np.arange(int(len(frames) * 25 / FPS)) * FPS / 25).astype(int)
        off, dist, conf = syncnet_score(net, crops[idx25], (audio * 32767).astype(np.int16), device)
        return {"offset": off, "dist": dist, "conf": conf}
    return score
