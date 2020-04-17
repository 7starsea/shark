# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from .base_dpg import BaseDPGPolicy


class DDPGPolicy(BaseDPGPolicy):
    def __init__(self, actor, critic, actor_optim, critic_optim, gamma, tau=0.005,
                 action_range=None):
        super().__init__('DDPG', actor, critic, actor_optim, critic_optim, gamma, tau,
                         action_range=action_range)

    def learn(self, batch, **kwargs):
        obs, act, reward, next_obs, done = batch.apply(torch.cat)

        with torch.no_grad():
            next_a = self._target_actor(next_obs)
            next_a = next_a * self._action_scale + self._action_bias

            q_next_value = self._target_critic(next_obs, next_a).detach()

            # Compute the expected Q values
            q_next_value = reward + q_next_value * (1 - done) * self.gamma

        q_value = self._critic(obs, act)
        td_error = torch.abs(q_value.detach() - q_next_value)

        weights = kwargs.get('weights', None)
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).to(dtype=q_next_value.dtype, device=q_next_value.device)
            critic_loss = (weights * ((q_value - q_next_value) ** 2)).mean()
        else:
            critic_loss = F.mse_loss(q_value, q_next_value)

        # Update the critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update the actor network
        actor_loss = -self._critic(obs, self.actor(obs)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        return critic_loss.item(), td_error

