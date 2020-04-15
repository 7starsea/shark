# -*- coding: utf-8 -*-
import numpy as np
from .replay_base import PrioritizedReplayBufferBase


class NaivePrioritizedReplayBuffer(PrioritizedReplayBufferBase):
    def __init__(self, transition, capacity, batch_size, alpha=0.6, eps=1e-10, decay_alpha=.5):
        super(NaivePrioritizedReplayBuffer, self).__init__(transition, capacity, batch_size,
                                                      alpha=alpha, eps=eps, decay_alpha=decay_alpha)

        self.priorities = np.zeros((capacity,), dtype=np.float64)  # # actually is priorities ** alpha
        self._max_priority = 1.0

    def append(self, item):
        idx = super().append(item)
        self.priorities[idx] = self._max_priority  # # already with ** alpha

    def sample(self, beta=0.4):
        total = len(self.memory)
        if total == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # probs = priorities ** self._alpha
        # probs /= (probs.sum() + self._eps)
        probs = priorities / (priorities.sum() + self._eps)

        indices = np.random.choice(total, self.batch_size, replace=False, p=probs)

        transitions = [self.memory[idx] for idx in indices]
        batch = self.transition(*zip(*transitions))

        # weights = (probs[indices]) ** (-beta)

        priorities = priorities[indices]
        weights = np.power(priorities, -beta)
        weights /= (np.max(weights) + self._eps)
        return batch, indices, weights, priorities

    def update_priorities(self, indices, old_priorities, priorities):
        np.clip(priorities, self._eps, None, out=priorities)
        np.power(priorities, self._alpha, out=priorities)

        self._max_priority = max(self._max_priority, np.max(priorities))
        self.priorities[indices] = np.maximum(priorities, self._decay_alpha * old_priorities)


