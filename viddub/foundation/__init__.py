"""Minimal Wan2.1-VACE backend used by VidDub.

The retained Wan model code is derived from DiffSynth-Studio commit
7686e54d41d25c0e8ed5f1318acc23b6bb832654 under Apache-2.0. Unrelated model
families and Wan extensions are intentionally excluded from this fork.
"""

from .pipelines.wan_video import ModelConfig, WanVideoPipeline
