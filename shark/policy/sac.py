
# https://github.com/denisyarats/pytorch_sac

import numpy as np
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from .base_dpg import BaseDPGPolicy, DPGDualCriticModel


class SACPolicy(BaseDPGPolicy):
    def __init__(self, actor, critic, actor_optim, critic_optim, gamma, tau=0.005,
                 eps=0.1, action_range=None, policy_freq=2,
                 alpha=0.2, dist_fn=torch.distributions.Normal,
                 automatic_entropy_tuning=True
                 ):
        super().__init__('SAC', actor, critic, actor_optim, critic_optim, gamma, tau=tau,
                         eps=eps, action_range=action_range)

        assert isinstance(critic, DPGDualCriticModel) and "We should use DPGDualCriticModel, see policy/base_dpg.py!"

        self._alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            device = next(actor.parameters()).device

            self.target_entropy = -torch.prod(torch.Tensor(1).to(device=device)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self._log_alpha], lr=actor_optim.defaults['lr'])

        self._count = 0
        self._policy_freq = policy_freq
        self._dist_fn = dist_fn

    def actor(self, s):
        with torch.no_grad():
            mu_std = self._actor(s)

            assert isinstance(mu_std, tuple)
            dist = self._dist_fn(*mu_std)

            y = torch.tanh(dist.rsample())

            l, h = self._range
            action = (h - l) * (y + 1) / 2 + l

            # action = action.clamp(*self.action_range)
        return action

    def _evaluate(self, s):
        mu_std = self._actor(s)

        assert isinstance(mu_std, tuple)
        dist = self._dist_fn(*mu_std)

        z = dist.sample()
        y = torch.tanh(z)

        l, h = self._range
        action = (h - l) * (y + 1) / 2 + l

        # action = action.clamp(*self.action_range)

        log_prob = dist.log_prob(z) - torch.log(1 - y.pow(2) + self._eps)
        log_prob = log_prob.sum(-1)
        return action, log_prob

    def learn(self, batch, **kwargs):
        obs, act, reward, next_obs, done = batch.apply(torch.cat)

        # Compute the expected Q values (label)
        with torch.no_grad():
            next_a, log_prob = self._evaluate(next_obs)

            q1, q2 = self._target_critic(next_obs, next_a)
            q_next_value = torch.min(q1, q2).detach()
            q_next_value = q_next_value - self._alpha * log_prob
            q_next_value = reward + q_next_value * (1 - done) * self.gamma


        # Update the critic network
        q1, q2 = self._critic(obs, act)
        td_error = torch.abs(q1.detach() - q_next_value) + torch.abs(q2.detach() - q_next_value)

        critic_loss = F.mse_loss(q1, q_next_value) + F.mse_loss(q2, q_next_value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        a, log_prob = self._evaluate(obs)
        q1a, q2a = self._critic(obs, a)
        actor_loss = self._alpha * log_prob - torch.min(q1a, q2a)
        actor_loss = actor_loss.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self._alpha = self._log_alpha.exp()

        self._count += 1
        if 0 == self._count % self._policy_freq:
            self.soft_update()

        return critic_loss.item(), td_error
