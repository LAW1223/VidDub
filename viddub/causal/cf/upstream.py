"""Attributed scheduler compatibility from Causal-Forcing.

Adapted from ``utils/scheduler.py`` at commit
da3ddf1a590f30c9edc568e11c2f77170995c31c. The upstream project is
Apache-2.0 licensed. Keeping this small compatibility class locally avoids
requiring users to clone an otherwise unused complete repository.
"""
import hashlib
from pathlib import Path

import torch

COMMIT = "da3ddf1a590f30c9edc568e11c2f77170995c31c"
REPOSITORY = "https://github.com/thu-ml/Causal-Forcing"


def provenance():
    source = Path(__file__).resolve()
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "compatibility_file": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, extra_one_step=False,
                 reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0,
                      training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, count)
        if self.extra_one_step:
            self.sigmas = self.sigmas[:-1]
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            shifted = y - y.min()
            self.linear_timesteps_weights = shifted * (num_inference_steps / shifted.sum())

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        shape = (-1,) + (1,) * (sample.ndim - 1)
        sigma = self.sigmas[timestep_id].reshape(shape)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            next_sigma = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            next_sigma = self.sigmas[timestep_id + 1].reshape(shape)
        return sample + model_output * (next_sigma - sigma)

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        shape = (-1,) + (1,) * (noise.ndim - 1)
        sigma = self.sigmas[timestep_id].reshape(shape)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    @staticmethod
    def training_target(sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        return self.linear_timesteps_weights[timestep_id]
