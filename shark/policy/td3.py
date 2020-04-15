# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_dpg import BaseDPGPolicy, DPGDualCriticModel


# def TD3DualCriticFactory(name, critic_model):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.critic = critic_model(*args, **kwargs)
#         self.critic2 = critic_model(*args, **kwargs)
#
#     def forward(self, *args, **kwargs):
#         return self.critic(*args, **kwargs), self.critic2(*args, **kwargs)
#
#     def q1(self, *args, **kwargs):
#         return self.critic(*args, **kwargs)
#
#     new_class = type(name, (nn.Module,), {"__init__": __init__, 'forward': forward, 'q1': q1})
#     return new_class


# class TD3DualCriticWrap(object):
#     def __init__(self, critic):
#         self.critic = critic
#         self.critic2 = deepcopy(critic)
#
#     def __call__(self, *args, **kwargs):
#         return self.critic(*args, **kwargs), self.critic2(*args, **kwargs)
#
#     def forward(self, *args, **kwargs):
#         return self.critic(*args, **kwargs), self.critic2(*args, **kwargs)
#
#     def eval(self):
#         self.critic.eval(), self.critic2.eval()
#
#     def train(self):
#         self.critic.train(), self.critic2.train()
#
#     def q1(self, *args, **kwargs):
#         return self.critic(*args, **kwargs)
#
#     def parameters(self):
#         return list(self.critic.parameters()) + list(self.critic2.parameters())
#
#     def __deepcopy__(self, memo={}):
#         cls = self.__class__
#         result = cls.__new__(cls)
#         memo[id(self)] = result
#
#         for k, v in self.__dict__.items():
#             setattr(result, k, deepcopy(v, memo))
#         return result


# class TD3DualCriticModel(nn.Module):
#     def __init__(self, critic_model, *args, **kwargs):
#         super(TD3DualCriticModel, self).__init__()
#         self.critic = critic_model(*args, **kwargs)
#         self.critic2 = critic_model(*args, **kwargs)
#
#     def forward(self, *args, **kwargs):
#         return self.critic(*args, **kwargs), self.critic2(*args, **kwargs)
#
#     def q1(self, *args, **kwargs):
#         return self.critic(*args, **kwargs)


class TD3Policy(BaseDPGPolicy):
    def __init__(self, actor, critic, actor_optim, critic_optim, gamma, tau=0.005,
                 eps=0.1, action_range=None,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2
                 ):
        super().__init__('TD3', actor, critic, actor_optim, critic_optim, gamma, tau=tau,
                         eps=eps, action_range=action_range)

        assert isinstance(critic, DPGDualCriticModel) and "We should use DPGDualCriticModel, see policy/base_dpg.py!"
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_freq = policy_freq
        self._count = 0

    def learn(self, batch, **kwargs):
        obs, act, reward, next_obs, done = batch.apply(torch.cat)

#        obs = torch.cat(batch.obs)
#        act = torch.cat(batch.act)
#        reward = torch.cat(batch.reward)
#        next_obs = torch.cat(batch.next_obs)
#        done = torch.cat(batch.done)

        # Compute the expected Q values (label)
        with torch.no_grad():
            next_a = self.target_actor(next_obs).detach()

            noise = torch.randn(size=next_a.shape, device=next_a.device) * self._policy_noise
            if self._noise_clip >= 0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            next_a += noise
            next_a = next_a.clamp(self._range[0], self._range[1])

            q1, q2 = self.target_critic(next_obs, next_a)
            q_next_value = torch.min(q1, q2).detach()

        q_next_value = reward + q_next_value * (1 - done) * self.gamma

        # Update the critic network
        q1, q2 = self.critic(obs, act)
        td_error = torch.abs(q1.detach() - q_next_value) + torch.abs(q2.detach() - q_next_value)

        weights = kwargs.get('weights', None)
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).to(dtype=q_next_value.dtype, device=q_next_value.device)
            critic_loss = (weights * ((q1 - q_next_value) ** 2 + (q2 - q_next_value) ** 2)).mean()
        else:
            critic_loss = F.mse_loss(q1, q_next_value) + F.mse_loss(q2, q_next_value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if 0 == self._count % self._policy_freq:
            # Update the actor network
            actor_loss = -self.critic.q1(obs, self.actor(obs))
            actor_loss = actor_loss.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_update()

        self._count += 1
        return critic_loss.item(), td_error

