# Retained upstream foundations

VidDub includes a reduced `viddub/foundation/` fork containing only the Wan2.1-VACE
foundation required by this project. It is derived from DiffSynth-Studio commit
`7686e54d41d25c0e8ed5f1318acc23b6bb832654` under Apache-2.0.

The retained subset covers Wan DiT, VACE, VAE, T5/tokenization, gradient
checkpointing and the Wan flow-matching scheduler. Other DiffSynth model
families and optional Wan methods are intentionally excluded.

## Causal-Forcing attribution

The optional consistency branch is adapted from Causal-Forcing commit
`da3ddf1a590f30c9edc568e11c2f77170995c31c`. Its required flow-matching
scheduler is included as a small attributed compatibility module so users do
not need to clone the complete upstream repository. See `viddub/causal/cf/NOTICE`.
