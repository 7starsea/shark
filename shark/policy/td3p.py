# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from collections import namedtuple
import torch
import torch.nn.functional as F
from .base_dpg import BaseDPGPolicy
from .ddpg import DDPGPolicy


class TD3Policy(BaseDPGPolicy):
    def __init__(self, actor, critic, critic2, actor_optim, critic_optim, critic2_optim, gamma, tau=0.005,
                 eps=0.1, action_range=None,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2
                 ):
        super().__init__('TD3P', actor, critic, actor_optim, critic_optim, gamma, tau=tau,
                         eps=eps, action_range=action_range)

        self.critic2 = critic2
        self.critic2_optim = critic2_optim

        self.target_critic2 = deepcopy(critic2)
        self.target_critic2.eval()

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_freq = policy_freq
        self._count = 0

    def soft_update(self):
        super().soft_update()

        for o, n in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self.tau) + n.data * self.tau)

    def learn(self, batch, **kwargs):
        state_batch = torch.cat(batch.obs)
        next_state_batch = torch.cat(batch.next_obs)
        action_batch = torch.cat(batch.act)
        reward_batch = torch.cat(batch.reward)
        done = torch.cat(batch.done)

        # Compute the expected Q values (label)
        with torch.no_grad():
            next_a = self.target_actor(next_state_batch).detach()

            noise = torch.randn(size=next_a.shape, device=next_a.device) * self._policy_noise
            if self._noise_clip >= 0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            next_a += noise
            next_a = next_a.clamp(self._range[0], self._range[1])

            q_next_value = torch.min(self.target_critic(next_state_batch, next_a),
                                     self.target_critic2(next_state_batch, next_a))

        q_next_value = reward_batch + q_next_value * (1 - done) * self.gamma
        q_next_value = q_next_value.detach()

        # Update the critic1 network
        q_value = self.critic(state_batch, action_batch)
        td_error = torch.abs(q_value.detach() - q_next_value)

        critic_loss = F.mse_loss(q_value, q_next_value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update the critic2 network
        q_value = self.critic2(state_batch, action_batch)
        td_error += torch.abs(q_value.detach() - q_next_value)

        critic_loss2 = F.mse_loss(q_value, q_next_value)
        self.critic2_optim.zero_grad()
        critic_loss2.backward()
        self.critic2_optim.step()

        if 0 == self._count % self._policy_freq:
            # Update the actor network
            actor_loss = -self.critic(state_batch, self.actor(state_batch))
            actor_loss = actor_loss.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_update()

        self._count += 1
        return (critic_loss + critic_loss2).item(), td_error

