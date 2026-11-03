"""Causal forward over the retained Wan/VACE and VidDub audio weights.

Parameter names remain compatible with the released checkpoints. The reference
is a separate prefix block; video blocks attend to themselves and earlier blocks.
Cache writes are explicit, inference-only transactions on clean history.
"""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CausalConfig:
    chunk_size: int = 3  # latent frames; first video chunk includes the VAE first frame
    num_ref: int = 1
    fps: float = 24.0

    def __post_init__(self):
        if self.chunk_size < 1 or self.num_ref not in (0, 1) or self.fps <= 0:
            raise ValueError("Require chunk_size >= 1, num_ref in {0,1}, fps > 0")

    def blocks(self, frames):
        if frames <= self.num_ref:
            raise ValueError("Sequence must contain at least one video latent")
        out = [(0, self.num_ref)] if self.num_ref else []
        out += [(i, min(i + self.chunk_size, frames))
                for i in range(self.num_ref, frames, self.chunk_size)]
        return out

    def video_blocks(self, frames):
        return [(a, b) for a, b in self.blocks(frames) if a >= self.num_ref]


@dataclass
class HistoryCache:
    """Use a NEW instance for every sample/condition/CFG branch/model revision.

    Keys are layer-specific; positions are absolute and never reset at a chunk.
    No eviction yet: this reference implementation is bounded by training length.
    """
    frames: int = 0
    entries: dict = field(default_factory=dict)
    geometry: tuple | None = None
    model_id: int | None = None
    # None preserves the correctness-reference behavior (all clean history).
    # Otherwise keep the separate reference prefix plus at most this many
    # most-recent video latent frames. Absolute RoPE is already embedded in K.
    max_video_history_frames: int | None = None
    retained_frames: int = 0

    def __post_init__(self):
        if self.max_video_history_frames is not None and self.max_video_history_frames < 0:
            raise ValueError("max_video_history_frames must be nonnegative or None")


def _sinusoidal(dim, t):
    freq = torch.pow(10000, -torch.arange(dim // 2, device=t.device,
                                       dtype=torch.float64) / (dim // 2))
    angles = t.flatten().double()[:, None] * freq[None]
    return torch.cat([angles.cos(), angles.sin()], -1).to(t.dtype)


def _rope(x, freqs, heads):
    b, n, dim = x.shape
    z = x.reshape(b, n, heads, -1, 2).double().contiguous()
    z = torch.view_as_complex(z) * freqs
    return torch.view_as_real(z).flatten(2).to(x.dtype)


def _attention(q, k, v, heads):
    # RMSNorm can promote q/k when trainable master weights are float32.
    q, k = q.to(v.dtype), k.to(v.dtype)
    if v.is_cuda and v.dtype in (torch.bfloat16, torch.float16):
        # Use the SAME fused backend as the SFT teacher. BF16 SDPA vs FA2
        # differences compound substantially through this trained checkpoint.
        from viddub.foundation.models.wan_video_dit import flash_attention
        return flash_attention(q, k, v, heads)
    def split(z):
        return z.unflatten(-1, (heads, -1)).transpose(1, 2)
    y = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.0)
    return y.transpose(1, 2).flatten(2)


class CausalWanDub(nn.Module):
    def __init__(self, dit, vace, adapter, config=CausalConfig(), causal=True):
        super().__init__()
        self.dit, self.vace, self.adapter = dit, vace, adapter
        self.config, self.causal = config, causal
        if dit.patch_size[0] != 1 or vace.vace_patch_embedding.kernel_size[0] != 1:
            raise ValueError("Only temporal patch size 1 is supported")
        if len(adapter.blocks) != len(dit.blocks):
            raise ValueError("Audio/DiT layer counts differ")

    def _self_attention(self, attn, x, freqs, blocks, tokens_per_frame,
                        start_frame, cache, pending, key, detach_history_before):
        q = _rope(attn.norm_q(attn.q(x)), freqs, attn.num_heads)
        k = _rope(attn.norm_k(attn.k(x)), freqs, attn.num_heads)
        v = attn.v(x)
        k = k.to(v.dtype)
        if cache is None and detach_history_before:
            stop = detach_history_before * tokens_per_frame
            k = torch.cat([k[:, :stop].detach(), k[:, stop:]], 1)
            v = torch.cat([v[:, :stop].detach(), v[:, stop:]], 1)
        if cache is not None:
            old = cache.entries.get(key)
            if cache.frames and old is None:
                raise ValueError(f"Missing history for {key}")
            if old is not None:
                k = torch.cat([old[0], k], 1)
                v = torch.cat([old[1], v], 1)
            if pending is not None:
                pending[key] = (k.detach(), v.detach())
        if not self.causal:
            y = _attention(q, k, v, attn.num_heads)
        elif cache is not None:
            # A cached call is exactly one complete current block. Its K/V are
            # retained clean history followed by all tokens of this block, so
            # every current query may read the entire tensor. Absolute temporal
            # positions were already applied by RoPE before K entered the cache.
            y = _attention(q, k, v, attn.num_heads)
        else:
            pieces = []
            for a, b in blocks:
                a, b = max(a, start_frame), min(b, start_frame + x.shape[1] // tokens_per_frame)
                if a >= b:
                    continue
                qa, qb = (a - start_frame) * tokens_per_frame, (b - start_frame) * tokens_per_frame
                # All spatial tokens WITHIN this block are mutually visible.
                pieces.append(_attention(q[:, qa:qb], k[:, :b * tokens_per_frame],
                                         v[:, :b * tokens_per_frame], attn.num_heads))
            y = torch.cat(pieces, 1)
        return attn.o(y)

    def _block(self, block, x, context, t_mod, freqs, blocks, spatial,
               start, cache, pending, key, detach_history_before):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(t_mod) + t_mod).unbind(2)
        h = block.norm1(x) * (1 + scale) + shift
        x = x + gate * self._self_attention(block.self_attn, h, freqs, blocks,
                                           spatial, start, cache, pending, key, detach_history_before)
        ca = block.cross_attn
        if ca.has_image_input:
            raise ValueError("Use the VACE reference prefix, not CLIP image conditioning")
        h = block.norm3(x)
        x = x + ca.o(_attention(ca.norm_q(ca.q(h)), ca.norm_k(ca.k(context)),
                                ca.v(context), ca.num_heads))
        return x + gate_mlp * block.ffn(block.norm2(x) * (1 + scale_mlp) + shift_mlp)

    def forward(self, latents, timesteps, context, vace_context, whisper_hs,
                num_frames, drop_audio=None, *, start_frame=0, cache=None,
                commit_cache=False, gradient_checkpointing=False, detach_history_before=0,
                return_hidden_layer=None, audio_tokens=None):
        """timesteps: scalar, [B], or [B,F], using t=1000*sigma.

        A cached call receives ONLY a single new block of latents/conditions.
        whisper_hs remains the full (offline) clip feature tensor, indexed with
        absolute frame offsets. Reference latents must be clean at sampling time.
        """
        b, _, frames, height, width = latents.shape
        if return_hidden_layer is not None and not 0 <= return_hidden_layer < len(self.dit.blocks):
            raise ValueError('Invalid hidden layer')
        total = (num_frames - 1) // 4 + 1 + self.config.num_ref
        if num_frames < 1 or (num_frames - 1) % 4 or start_frame + frames > total:
            raise ValueError("num_frames must be 4n+1 and cover this latent slice")
        if vace_context.shape[0] != b or vace_context.shape[2:] != latents.shape[2:]:
            raise ValueError("VACE and video latent shapes must match")
        if height % self.dit.patch_size[1] or width % self.dit.patch_size[2]:
            raise ValueError("Latent spatial dimensions must be divisible by patch size")
        blocks = self.config.blocks(total)
        read_cache = cache
        if cache is None:
            if start_frame or commit_cache:
                raise ValueError("A nonzero start/commit requires a history cache")
            pending = None
        else:
            if not self.causal or start_frame != cache.frames:
                raise ValueError("Cache must be contiguous and causal")
            if (start_frame, start_frame + frames) not in blocks:
                raise ValueError("Cached call must cover exactly one complete block")
            geometry = (b, height, width, total, self.config, latents.device, latents.dtype)
            if cache.geometry is not None and cache.geometry != geometry:
                raise ValueError("Cache geometry changed; create a new cache")
            if cache.model_id not in (None, id(self)):
                raise ValueError("Cache belongs to a different model")
            if commit_cache and (torch.is_grad_enabled() or self.training):
                raise ValueError("Cache commits require eval() and torch.no_grad()")
            pending = {} if commit_cache else None
            # Checkpoint recomputation must see the history from THIS forward,
            # even after rollout commits subsequent chunks to the live cache.
            read_cache = HistoryCache(cache.frames, cache.entries, cache.geometry, cache.model_id,
                                      cache.max_video_history_frames, cache.retained_frames)
        ts = torch.as_tensor(timesteps, device=latents.device, dtype=latents.dtype)
        if ts.numel() == 1:
            ts = ts.reshape(1, 1).expand(b, frames)
        elif ts.shape == (b,):
            ts = ts[:, None].expand(b, frames)
        elif ts.shape != (b, frames):
            raise ValueError("timesteps must be scalar, [B], or [B,F]")
        if commit_cache and torch.count_nonzero(ts).item():
            raise ValueError("Only t=0 clean history may be committed")
        if audio_tokens is None:
            tokens = self.adapter.tokens(whisper_hs, num_frames, self.config.num_ref,
                                         torch.zeros(b, device=latents.device, dtype=torch.bool)
                                         if drop_audio is None else drop_audio,
                                         fps=self.config.fps)
        else:
            if drop_audio is not None and bool(torch.as_tensor(drop_audio).any()):
                raise ValueError("Precomputed conditional audio tokens cannot serve a null-audio branch")
            expected = (num_frames - 1) // 4 + 1 + self.config.num_ref
            if audio_tokens.shape[:2] != (b, expected):
                raise ValueError("Precomputed audio-token shape does not match the sequence")
            tokens = audio_tokens
        tokens = tokens[:, start_frame:start_frame + frames]
        x = self.dit.patchify(latents)
        _, _, f, h, w = x.shape
        spatial = h * w
        x = x.flatten(2).transpose(1, 2).contiguous()
        context = self.dit.text_embedding(context)
        # Project each distinct timestep before spatial expansion. Projecting
        # thousands of repeated BF16 rows selects different GEMM kernels than
        # the original single-timestep path and needlessly amplifies rounding.
        unique_ts, inverse = torch.unique(ts, return_inverse=True)
        time_values, mod_values = [], []
        for value in unique_ts.unbind():
            emb = self.dit.time_embedding(_sinusoidal(self.dit.freq_dim, value[None]))
            time_values.append(emb)
            mod_values.append(self.dit.time_projection(emb))
        t = torch.cat(time_values)[inverse].reshape(b, f, -1).repeat_interleave(spatial, 1)
        t_mod = torch.cat(mod_values)[inverse].reshape(b, f, 6, self.dit.dim).repeat_interleave(spatial, 1)
        ff, fh, fw = self.dit.freqs
        if start_frame + f > len(ff) or h > len(fh) or w > len(fw):
            raise ValueError("Position exceeds the model RoPE table")
        freqs = torch.cat([
            ff[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            fh[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            fw[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], -1).reshape(f * h * w, 1, -1).to(x.device)

        def run(block, value, key):
            def fn(value):
                return self._block(block, value, context, t_mod, freqs, blocks,
                                   spatial, start_frame, read_cache, pending, key, detach_history_before)
            if gradient_checkpointing and torch.is_grad_enabled():
                return checkpoint(fn, value, use_reentrant=False)
            return fn(value)

        # Original VACE concatenates per-example tokens, yielding contiguous
        # [B,N,C] storage before before_proj. Preserve that GEMM input layout.
        c = self.vace.vace_patch_embedding(vace_context).flatten(2).transpose(1, 2).contiguous()
        hints = []
        for i, block in enumerate(self.vace.vace_blocks):
            if i == 0:
                c = block.before_proj(c) + x
            c = run(block, c, ("vace", i))
            hints.append(block.after_proj(c))
        selected_hidden = None
        for i, block in enumerate(self.dit.blocks):
            x = run(block, x, ("dit", i))
            if i in self.vace.vace_layers_mapping:
                x = x + hints[self.vace.vace_layers_mapping[i]]
            audio_block = self.adapter.blocks[i]
            x = checkpoint(audio_block, x, tokens, f, use_reentrant=False) if (
                gradient_checkpointing and torch.is_grad_enabled()) else audio_block(x, tokens, f)
            if i == return_hidden_layer:
                selected_hidden = x.reshape(b, f, h, w, -1)
        out = self.dit.unpatchify(self.dit.head(x, t), (f, h, w))
        if pending is not None:
            # Publish only after EVERY layer and output succeeded.
            retained_frames = start_frame + frames
            if cache.max_video_history_frames is not None:
                keep_video = min(cache.max_video_history_frames,
                                 max(0, retained_frames - self.config.num_ref))
                keep_total = self.config.num_ref + keep_video
                for key, (k, v) in pending.items():
                    ref_tokens = self.config.num_ref * spatial
                    video_tokens = keep_video * spatial
                    if ref_tokens:
                        k_ref, v_ref = k[:, :ref_tokens], v[:, :ref_tokens]
                    else:
                        k_ref, v_ref = k[:, :0], v[:, :0]
                    if video_tokens:
                        k_video, v_video = k[:, -video_tokens:], v[:, -video_tokens:]
                    else:
                        k_video, v_video = k[:, :0], v[:, :0]
                    pending[key] = (torch.cat([k_ref, k_video], 1).detach(),
                                    torch.cat([v_ref, v_video], 1).detach())
                retained_frames = keep_total
            cache.entries = pending
            cache.frames = start_frame + frames
            cache.retained_frames = retained_frames
            cache.geometry, cache.model_id = geometry, id(self)
        return (out, selected_hidden) if return_hidden_layer is not None else out

    def architecture(self):
        return {"config": asdict(self.config), "causal": self.causal,
                "dim": self.dit.dim, "blocks": len(self.dit.blocks),
                "vace_blocks": len(self.vace.vace_blocks),
                "reference_policy": ("clean_separate_prefix" if self.causal
                                     else "jointly_noised_full_context"),
                "history_policy": ("clean_t0" if self.causal else "none"),
                "audio_policy": "offline_whisper"}
