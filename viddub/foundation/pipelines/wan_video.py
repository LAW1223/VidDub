"""Minimal Wan2.1-VACE-1.3B pipeline for VidDub.

This module intentionally exposes only the interfaces used by VidDub. It is
derived from DiffSynth-Studio commit 7686e54d41d25c0e8ed5f1318acc23b6bb832654
under Apache-2.0; unrelated model families and Wan extensions are removed.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from safetensors import safe_open

from ..core import gradient_checkpoint_forward
from ..diffusion import FlowMatchScheduler
from ..models.wan_video_dit import WanModel, precompute_freqs_cis_3d, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_vae import WanVideoVAE


@dataclass
class ModelConfig:
    path: str


def _load_torch(path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if len(state) == 1 and next(iter(state)) in ("state_dict", "model_state", "module"):
        state = state[next(iter(state))]
    return state


def _load_safetensors_subset(path, predicate):
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if predicate(key):
                state[key] = handle.get_tensor(key)
    return state


def _load_assign(model, state):
    """Materialize a meta-initialized module directly from one state dict."""
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {type(model).__name__}: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}")
    return model


class WanVideoPipeline(torch.nn.Module):
    def __init__(self, device="cpu", torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.scheduler = FlowMatchScheduler("Wan")
        self.dit = self.vace = self.vae = self.text_encoder = None
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, torch_dtype=torch.bfloat16, device="cpu",
                        model_configs=None, tokenizer_config=None, **_):
        configs = [Path(config.path) for config in (model_configs or [])]
        diffusion = next((p for p in configs if p.suffix == ".safetensors"), None)
        vae_path = next((p for p in configs if "VAE" in p.name.upper()), None)
        text_path = next((p for p in configs if "T5" in p.name.upper()), None)
        if diffusion is None or vae_path is None:
            raise ValueError("Wan2.1-VACE diffusion and VAE checkpoints are required")

        pipe = cls(device=device, torch_dtype=torch_dtype)
        with torch.device("meta"):
            pipe.dit = WanModel(
                has_image_input=False, patch_size=(1, 2, 2), in_dim=16,
                dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096,
                out_dim=16, num_heads=12, num_layers=30, eps=1e-6)
            pipe.vace = VaceWanModel()
        composite = _load_safetensors_subset(diffusion, lambda key: not key.startswith("vace"))
        _load_assign(pipe.dit, composite)
        pipe.dit.freqs = precompute_freqs_cis_3d(pipe.dit.dim // 12)
        del composite
        composite = _load_safetensors_subset(diffusion, lambda key: key.startswith("vace"))
        _load_assign(pipe.vace, composite)
        del composite

        with torch.device("meta"):
            pipe.vae = WanVideoVAE()
        vae_state = _load_torch(vae_path)
        if not any(key.startswith("model.") for key in vae_state):
            vae_state = {"model." + key: value for key, value in vae_state.items()}
        _load_assign(pipe.vae, vae_state)
        mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
        std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
        pipe.vae.mean, pipe.vae.std = mean, std
        pipe.vae.scale = [mean, 1.0 / std]
        del vae_state

        if text_path is not None:
            with torch.device("meta"):
                pipe.text_encoder = WanTextEncoder()
            _load_assign(pipe.text_encoder, _load_torch(text_path))
        if tokenizer_config is not None:
            pipe.tokenizer = HuggingfaceTokenizer(
                tokenizer_config.path, seq_len=512, clean="whitespace")
        pipe.to(device=device, dtype=torch_dtype)
        return pipe

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return super().to(*args, **kwargs)

    def preprocess_image(self, image, min_value=-1, max_value=1):
        array = torch.as_tensor(__import__("numpy").array(image), device=self.device)
        return array.to(self.torch_dtype).permute(2, 0, 1)[None] * ((max_value-min_value)/255) + min_value

    def preprocess_video(self, video, min_value=-1, max_value=1):
        return torch.stack([self.preprocess_image(frame, min_value, max_value)[0]
                            for frame in video], dim=1)[None]


class WanVideoUnit_PromptEmbedder:
    def encode_prompt(self, pipe, prompt):
        if pipe.text_encoder is None or pipe.tokenizer is None:
            raise ValueError("Text encoder/tokenizer is not loaded")
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        embedding = pipe.text_encoder(ids, mask)
        for index, length in enumerate(mask.gt(0).sum(1).long()):
            embedding[index, length:] = 0
        return embedding


def model_fn_wan_video(
    dit: WanModel,
    vace: Optional[VaceWanModel] = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    vace_context: torch.Tensor = None,
    vace_scale: float = 1.0,
    audio_adapter=None,
    audio_tokens: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **_,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], -1).reshape(f * h * w, 1, -1).to(x.device)
    hints = None if vace_context is None else vace(
        x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload)
    for block_id, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
            block, use_gradient_checkpointing,
            use_gradient_checkpointing_offload, x, context, t_mod, freqs)
        if hints is not None and block_id in vace.vace_layers_mapping:
            x = x + hints[vace.vace_layers_mapping[block_id]] * vace_scale
        if audio_adapter is not None and audio_tokens is not None:
            x = gradient_checkpoint_forward(
                audio_adapter.blocks[block_id], use_gradient_checkpointing,
                use_gradient_checkpointing_offload, x, audio_tokens, f)
    return dit.unpatchify(dit.head(x, t), (f, h, w))
