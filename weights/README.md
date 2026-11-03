# Weights

Binary checkpoints are published separately from Git. Every model version is an
atomic folder containing matching `dit`, `vace`, and `adapter` safetensors plus
a manifest with byte sizes, SHA256 values, stage, step, configuration, and
parent lineage.

Recommended public folders:

- `sft-r2a-ema200`: final bidirectional teacher.
- `full2-ode-r3-step400`: selected full-context two-step initializer.
- `full2-fake-r3-step200`: selected full-context fake-score initializer.
- `viddub-full2-dmd-r4-step100`: selected final full-context two-step model
  (`FULL2_DMD_R4@100`).

Base Wan/VACE, Whisper and auxiliary checkpoints are dependencies and should not
be silently repackaged unless their licenses explicitly allow it.

The expected auxiliary files are:

- `${VIDDUB_LATENTSYNC_ROOT}/checkpoints/stable_syncnet.pt`
  (LatentSync StableSyncNet used by the differentiable training loss);
- `${VIDDUB_SYNCNET_PATH}` or
  `weights/LatentSync-1.5/auxiliary/syncnet_v2.model`
  (SyncNet used for preprocessing alignment);
- `weights/whisper-small/` (Whisper-small feature encoder).

Set `VIDDUB_LATENTSYNC_ROOT` and `VIDDUB_SYNCNET_PATH` explicitly when
these assets are stored elsewhere. They are frozen dependencies and are not
part of the reported 2.441B released trained-component count.
