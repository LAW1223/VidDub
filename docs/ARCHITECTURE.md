# Architecture summary

VidDub extends Wan2.1-VACE with frozen Whisper-small features and a trainable
local audio adapter. Whisper hidden states are layer-weighted and converted to
frame-local audio tokens. Each Wan block receives audio cross-attention after
the VACE hint path. The adapter output projection is zero-initialized so the
initial network preserves the base model behavior.

The repository includes a reduced DiffSynth-derived backend containing only
the Wan2.1-VACE components required by VidDub: DiT, VACE, VAE, T5, gradient
checkpointing and the Wan flow-matching scheduler. Unrelated DiffSynth methods
and optional Wan extensions are not included.

All implementation code lives under one `viddub` package. `foundation`
contains the retained Wan/VACE backend, `full_context` contains teacher
training and inference utilities, and `causal` contains shared few-step
objectives plus the optional causal ablation. The final VidDub model is
full-context; `causal` is retained as a module name for compatibility with the
research trainer and ablation checkpoints.

Training begins with full-context flow matching. Paired-audio samples retain
full-frame reconstruction, whereas replacement-audio samples mask
source-derived reconstruction in the mouth and use audio--visual
synchronization to supervise articulation. Matched trajectory regression
transfers the guided 48-step teacher trajectories into a two-step
full-context student. Distribution matching distillation then refines the
student against frozen real and online fake score models while preserving
non-mouth source content.

The application detects a face and forms one stable 512px crop, lower-face
mask, and reference frame. An 81-frame clip uses one full-context model call.
A longer video uses overlapping 81-frame windows with 64-frame stride and
17-frame overlap; window seeds advance by index, shared frames are linearly
blended, and lower-face color is harmonized before the generated region is
pasted into the original-resolution frames.
