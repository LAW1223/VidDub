"""Audio conditioning for Wan-Dub: frozen Whisper encoder -> per-latent-frame tokens -> frame-local cross-attn in every DiT block."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperModel

WHISPER_HZ = 50
VIDEO_FPS = 24
TOKENS_PER_LATENT = 16  # audio frames (20ms each) around each latent frame's centre = 320ms


def stitch_hidden_segments(segments, total_frames):
    """Blend overlapping 50-Hz hidden-state segments on one global timeline.

    Each item is ``(global_start_frame, hidden_states)`` with hidden states
    shaped [B,L,T,C]. Linear edge ramps suppress segment-boundary context
    discontinuities. Exact source values are reconstructed when overlapping
    segments contain the same global features.
    """
    if total_frames < 1 or not segments:
        raise ValueError("Need nonempty segments and a positive output length")
    first = segments[0][1]
    if first.ndim != 4:
        raise ValueError("Expected [B,L,T,C] hidden states")
    values = first.new_zeros((*first.shape[:2], total_frames, first.shape[3]), dtype=torch.float32)
    weights = first.new_zeros((1, 1, total_frames, 1), dtype=torch.float32)
    for start, hidden in segments:
        if hidden.shape[:2] != first.shape[:2] or hidden.shape[3] != first.shape[3]:
            raise ValueError("Hidden-state segment shapes differ")
        if start < 0 or start >= total_frames:
            raise ValueError("Segment start is outside the global timeline")
        length = min(hidden.shape[2], total_frames-start)
        if length < 1: continue
        # Positive triangular weights; a single segment therefore remains exact.
        position = torch.arange(length, device=hidden.device, dtype=torch.float32)
        edge = torch.minimum(position+1, torch.tensor(float(length),device=hidden.device)-position)
        weight = edge.view(1,1,length,1)
        values[:,:,start:start+length] += hidden[:,:,:length].float()*weight
        weights[:,:,start:start+length] += weight
    if bool((weights == 0).any()):
        raise ValueError("Segment stitching left uncovered audio frames")
    return (values/weights).to(first.dtype)


class WhisperFeatures(nn.Module):
    """Frozen whisper encoder; returns all hidden states [B, L, T_a, C] for the first `seconds` of audio."""

    def __init__(self, path):
        super().__init__()
        self.extractor = WhisperFeatureExtractor.from_pretrained(path)
        self.encoder = WhisperModel.from_pretrained(path).encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, audios, device, dtype):
        feats = self.extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        hs = self.encoder(feats, output_hidden_states=True).hidden_states  # (L+1) x [B, 1500, C]
        n = math.ceil(max(len(a) for a in audios) / 16000 * WHISPER_HZ)
        return torch.stack([h[:, :n] for h in hs], 1)

    @torch.no_grad()
    def forward_long(self, audio, device, dtype, window_seconds=30, hop_seconds=20):
        """Encode arbitrarily long mono 16-kHz audio with overlapping windows.

        Audio no longer than one Whisper window uses ``forward`` directly and is
        bitwise identical to the existing path. Longer audio is encoded in
        overlapping windows and blended on the global 50-Hz timeline.
        """
        if getattr(audio, "ndim", 1) != 1:
            raise ValueError("forward_long expects one mono waveform")
        if not 0 < hop_seconds <= window_seconds <= 30:
            raise ValueError("Require 0 < hop <= window <= 30 seconds")
        window = int(window_seconds*16000); hop = int(hop_seconds*16000)
        if len(audio) <= window:
            return self.forward([audio], device, dtype)
        total = math.ceil(len(audio)/16000*WHISPER_HZ)
        starts = list(range(0, max(1, len(audio)-window+1), hop))
        last = max(0, len(audio)-window)
        if not starts or starts[-1] != last: starts.append(last)
        segments = []
        for sample_start in starts:
            waveform = audio[sample_start:min(len(audio),sample_start+window)]
            hidden = self.forward([waveform], device, dtype)
            frame_start = round(sample_start/16000*WHISPER_HZ)
            segments.append((frame_start, hidden))
        return stitch_hidden_segments(segments, total)


def latent_frame_centres(num_frames, fps=VIDEO_FPS):
    """Centre time (s) of each VAE latent frame: latent 0 = frame 0, latent k = frames 4k-3..4k."""
    n_lat = (num_frames - 1) // 4 + 1
    return torch.tensor([0.0] + [(4 * k - 1.5) / fps for k in range(1, n_lat)])


class AudioCrossAttn(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, audio_tokens, num_latent_frames):
        """x: [B, F*HW, C]; audio_tokens: [B, F, K, C]. Each latent frame only attends to its own audio window."""
        B, N, C = x.shape
        Fr, K = audio_tokens.shape[1], audio_tokens.shape[2]
        assert Fr == num_latent_frames, (Fr, num_latent_frames)
        h = self.num_heads
        q = self.q(self.norm(x)).view(B * Fr, N // Fr, h, C // h).transpose(1, 2)
        k, v = self.kv(audio_tokens).view(B * Fr, K, 2, h, C // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return x + self.o(out)


class DubAudioAdapter(nn.Module):
    def __init__(self, whisper_layers=13, whisper_dim=768, dim=1536, num_heads=12, num_blocks=30):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(whisper_layers))
        self.proj = nn.Sequential(nn.Linear(whisper_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Parameter(torch.zeros(TOKENS_PER_LATENT, dim))
        self.null = nn.Parameter(torch.zeros(TOKENS_PER_LATENT, dim))
        self.blocks = nn.ModuleList([AudioCrossAttn(dim, num_heads) for _ in range(num_blocks)])

    def tokens(self, whisper_hs, num_frames, num_ref_frames, drop, fps=VIDEO_FPS):
        """whisper_hs: [B, L, T_a, Cw] -> [B, num_ref + n_lat, K, dim]. drop: bool tensor [B] -> use null tokens."""
        B, _, T_a, _ = whisper_hs.shape
        feats = (whisper_hs * self.layer_logits.softmax(0).view(1, -1, 1, 1).to(whisper_hs.dtype)).sum(1)  # B, T_a, Cw
        centres = latent_frame_centres(num_frames, fps) * WHISPER_HZ
        idx = (centres[:, None] + torch.arange(TOKENS_PER_LATENT) - TOKENS_PER_LATENT / 2).round().long().clamp(0, T_a - 1)
        tok = self.proj(feats[:, idx.to(feats.device)]) + self.pos.to(feats.dtype)  # B, n_lat, K, dim
        null = self.null.to(tok.dtype).expand_as(tok)
        tok = torch.where(drop.view(B, 1, 1, 1).to(tok.device), null, tok)
        ref = self.null.to(tok.dtype).expand(B, num_ref_frames, -1, -1)
        return torch.cat([ref, tok], 1)
