"""Wan flow-matching scheduler retained from DiffSynth-Studio."""
import torch


class FlowMatchScheduler:
    def __init__(self, template="Wan"):
        if template != "Wan":
            raise ValueError("The VidDub fork only supports the Wan schedule")
        self.num_train_timesteps = 1000
        self.set_timesteps(100)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0,
                      training=False, shift=5.0):
        sigma_start = float(denoising_strength)
        self.sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = bool(training)
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - 500) / 1000) ** 2)
            shifted = y - y.min()
            self.linear_timesteps_weights = shifted * (1000 / shifted.sum())
            if len(self.timesteps) != 1000:
                self.linear_timesteps_weights *= len(self.timesteps) / 1000
                self.linear_timesteps_weights += self.linear_timesteps_weights[1]

    def _index(self, timestep):
        value = timestep.detach().float().cpu() if torch.is_tensor(timestep) else torch.tensor(timestep)
        return torch.argmin((self.timesteps - value).abs())

    def step(self, model_output, timestep, sample, to_final=False):
        index = self._index(timestep)
        sigma = self.sigmas[index].to(sample)
        next_sigma = sample.new_zeros(()) if to_final or index + 1 >= len(self.sigmas) else self.sigmas[index + 1].to(sample)
        return sample + model_output * (next_sigma - sigma)

    def add_noise(self, original_samples, noise, timestep):
        sigma = self.sigmas[self._index(timestep)].to(original_samples)
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        return self.linear_timesteps_weights[self._index(timestep)].to(timestep.device)
