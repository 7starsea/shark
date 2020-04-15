# -*- coding: utf-8 -*-
import numpy as np
from .SharkUtil import PrioritizedExperienceBase
from .replay_base import PrioritizedReplayBufferBase


class PrioritizedReplayBuffer(PrioritizedReplayBufferBase):
    def __init__(self, transition, capacity, batch_size, alpha=0.6, eps=1e-10, decay_alpha=.5):
        super(PrioritizedReplayBuffer, self).__init__(transition, capacity, batch_size,
                                                      alpha=alpha, eps=eps, decay_alpha=decay_alpha)
        self._prior_base_c = PrioritizedExperienceBase(capacity, batch_size)

        self.data = [None for i in range(capacity)]
        self._indices = np.zeros(batch_size, dtype=np.int32)
        self._priorities = np.zeros(batch_size, dtype=np.float64)
        self._max_priority = 1.0

    def append(self, item):
        assert isinstance(item, self.transition)
        index = self._prior_base_c.add_c(self._max_priority)   # # already with ** alpha
        self.data[index] = item

    def sample(self, beta=.4):
        if self._prior_base_c.size() <= self.batch_size:
            return None, None, None, None

        self._prior_base_c.sample_c(self._indices, self._priorities)
        ind = self._indices >= 0
        if len(ind) <= 0:
            return None, None, None, None

        indices, priorities = self._indices[ind], self._priorities[ind]

        weights = np.array(priorities)
        np.power(weights, -beta, out=weights)
        weights /= (np.max(weights) + self._eps)

        transitions = [self.data[idx] for idx in indices]
        batch = self.transition(*zip(*transitions))
        return batch, indices, weights, priorities

    def update_priorities(self, indices, old_priorities, priorities):
        """
        :param indices:  np.1darray
        :param old_priorities: np.1darray
        :param priorities: np.1darray
        :return:
        """
        np.clip(priorities, self._eps, None, out=priorities)
        np.power(priorities, self._alpha, out=priorities)
        self._max_priority = max(self._max_priority, np.max(priorities))

        old_priorities = old_priorities * self._decay_alpha
        np.maximum(priorities, old_priorities, out=priorities)

        self._prior_base_c.update_priority_c(indices, priorities)

    def __len__(self):
        return len(self._prior_base_c)
