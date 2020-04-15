# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from .base_dpg import BaseDPGPolicy


class DDPGPolicy(BaseDPGPolicy):
    def __init__(self, actor, critic, actor_optim, critic_optim, gamma, tau=0.005,
                 eps=0.1, action_range=None):
        super().__init__('DDPG', actor, critic, actor_optim, critic_optim, gamma, tau,
                         eps=eps, action_range=action_range)

    def learn(self, batch, **kwargs):
        obs, act, reward, next_obs, done = batch.apply(torch.cat)

        with torch.no_grad():
            next_act = self._target_actor(next_obs).detach()
            q_next_value = self._target_critic(next_obs, next_act)

        # Compute the expected Q values
        q_next_value = reward + q_next_value * (1 - done) * self.gamma

        q_value = self._critic(obs, act)
        td_error = torch.abs(q_value.detach() - q_next_value)
        critic_loss = F.mse_loss(q_value, q_next_value)

        # Update the critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update the actor network
        actor_loss = -self._critic(obs, self.actor(obs))
        actor_loss = actor_loss.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        return critic_loss.item(), td_error

