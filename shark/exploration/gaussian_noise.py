
import torch
from .exploration_base import ExplorationBase


class GaussianNoise(ExplorationBase):
    def __init__(self, scale, initial_eps=0.2, final_eps=0.05, fixed_ratio=0.7):
        super(GaussianNoise, self).__init__(initial_eps, final_eps, fixed_ratio)
        assert scale >= 0 and "exploration noise scale should not be negative"
        self._scale = scale

    def is_continuous(self):
        return True

    def sample(self, training_ratio, shape, dtype, device=None):
        eps = self.eps(training_ratio)
        return torch.randn(*shape, dtype=dtype, device=device) * self._scale * eps

