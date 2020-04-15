# -*- coding: utf-8 -*-
from collections import deque
from abc import ABC, abstractmethod


class ReplayBufferBase(ABC):
    def __init__(self, transition, capacity, batch_size):

        # data instance, see shark/policy/(base_dpg.py/DPGTransition, dqn.py/DQNTransition)
        self.transition = transition
        self.capacity = capacity
        self.batch_size = batch_size

        self.memory = deque(maxlen=capacity)
        self.position = 0

    def append(self, item):
        """Saves a transition."""
        assert isinstance(item, self.transition)
        self.memory.append(item)
        idx = self.position
        self.position = (self.position + 1) % self.capacity
        return idx

    def __len__(self):
        return len(self.memory)

    @abstractmethod
    def sample(self, **kwargs):
        pass


class PrioritizedReplayBufferBase(ReplayBufferBase):
    def __init__(self, transition, capacity, batch_size, alpha=0.6, eps=1e-10, decay_alpha=.5):
        super(PrioritizedReplayBufferBase, self).__init__(transition, capacity, batch_size)
        assert alpha >= 0 and eps > 0 and "alpla >= 0 and eps > 0"
        self._alpha = alpha
        self._eps = eps
        self._decay_alpha = decay_alpha

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @abstractmethod
    def update_priorities(self, indices, old_priorities, priorities):
        pass
