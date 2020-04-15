# -*- coding: utf-8 -*-
import random
from .replay_base import ReplayBufferBase


class SimpleReplayBuffer(ReplayBufferBase):
    def __init__(self, transition, capacity, batch_size):
        super(SimpleReplayBuffer, self).__init__(transition, capacity, batch_size)

    def sample(self):
        res = random.sample(self.memory, self.batch_size)
        batch = self.transition(*zip(*res))
        return batch

