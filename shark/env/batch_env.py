# -*- coding: utf-8 -*-
import numpy as np
import torch
from gym import Wrapper


class BatchDeviceWrapper(Wrapper):
    def __init__(self, env, device='cpu'):
        super(BatchDeviceWrapper, self).__init__(env)
        self.device = device

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.reshape(-1)
        ob, reward, done, info = self.env.step(action)
        ob = ob.to(device=self.device)

        reward = torch.from_numpy(np.stack([reward])).to(device=self.device, dtype=torch.float32)
        done = torch.from_numpy(np.stack([done])).to(device=self.device, dtype=torch.float32)
        return ob.unsqueeze(0), reward, done, info

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        ob = ob.to(device=self.device)
        return ob.unsqueeze(0)

    # def render(self, *args, **kwargs):
    #     self.env.render(*args, **kwargs)
    #
    # def seed(self, *args, **kwargs):
    #     self.env.seed(*args, **kwargs)
    #
    # def close(self, *args, **kwargs):
    #     self.env.close(*args, **kwargs)
