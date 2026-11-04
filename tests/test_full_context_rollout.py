import unittest

import torch

from viddub.causal.sampling import rollout, schedule


class _FullContextConfig:
    num_ref = 1

    @staticmethod
    def video_blocks(frames):
        return [(1, frames)]


class _FullContextModel(torch.nn.Module):
    causal = False
    config = _FullContextConfig()

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        latents,
        timesteps,
        context,
        vace_context,
        whisper_hs,
        num_frames,
        drop_audio=None,
        **kwargs,
    ):
        self.calls.append((latents.shape, timesteps.shape, drop_audio))
        return torch.ones_like(latents)


class FullContextRolloutTests(unittest.TestCase):
    def setUp(self):
        self.model = _FullContextModel()
        self.reference = torch.zeros(1, 2, 4, 2, 2)
        self.conditions = {
            "context": torch.zeros(1, 1, 1),
            "vace_context": torch.zeros(1, 3, 4, 2, 2),
            "whisper_hs": torch.zeros(1, 1, 1),
            "num_frames": 13,
        }

    def test_bidirectional_model_denoises_the_complete_sequence_each_step(self):
        output = rollout(
            self.model,
            self.reference,
            self.conditions,
            schedule(2),
            noise=torch.zeros_like(self.reference),
            use_cache=False,
        )

        self.assertEqual(len(self.model.calls), 2)
        self.assertTrue(all(call[0] == self.reference.shape for call in self.model.calls))
        self.assertTrue(all(call[1] == (1, 4) for call in self.model.calls))
        self.assertTrue(torch.equal(output, -torch.ones_like(output)))

    def test_bidirectional_model_rejects_causal_cache(self):
        with self.assertRaisesRegex(ValueError, "cannot use a history cache"):
            rollout(
                self.model,
                self.reference,
                self.conditions,
                schedule(2),
                noise=torch.zeros_like(self.reference),
                use_cache=True,
            )


if __name__ == "__main__":
    unittest.main()
