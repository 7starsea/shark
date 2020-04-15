# coding=utf-8
from abc import ABC, abstractmethod
from copy import deepcopy
import torch.nn as nn
from .base import BasePolicy
from .namedarraytuple import namedarraytuple


DPGTransition = namedarraytuple('DPGTransition', ('obs', 'act', 'reward', 'next_obs', 'done'))


class DPGDualCriticModel(nn.Module):
    def __init__(self, critic_model, *args, **kwargs):
        super(DPGDualCriticModel, self).__init__()
        self.critic = critic_model(*args, **kwargs)
        self.critic2 = critic_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.critic(*args, **kwargs), self.critic2(*args, **kwargs)

    def q1(self, *args, **kwargs):
        return self.critic(*args, **kwargs)


class BaseDPGPolicy(BasePolicy):
    def __init__(self, name, actor, critic, actor_optim, critic_optim, gamma, tau,
                 eps=0.1, action_range=None):
        super().__init__(name, nn.ModuleDict({'actor': actor, 'critic': critic}))
        self.actor = actor
        self.critic = critic
        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)

        self.target_actor.eval(), self.target_critic.eval()

        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self.gamma = gamma
        self.tau = tau

        assert eps >= 0 and "noise eps should not be negative"
        self._eps = eps

        assert action_range is not None
        self._range = action_range

    def set_eps(self, eps):
        assert eps >= 0 and "noise eps should not be negative"
        self._eps = eps

    def soft_update(self):
        for o, n in zip(self.target_actor.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self.tau) + n.data * self.tau)

        for o, n in zip(self.target_critic.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self.tau) + n.data * self.tau)

    def sync_target(self, tau=1):
        for o, n in zip(self.target_actor.parameters(), self.actor.parameters()):
            o.data.copy_(n.data)

        for o, n in zip(self.target_critic.parameters(), self.critic.parameters()):
            o.data.copy_(n.data)

    def reset_model(self):
        for o, n in zip(self.target_actor.parameters(), self.actor.parameters()):
            n.data.copy_(o.data)

        for o, n in zip(self.target_critic.parameters(), self.critic.parameters()):
            n.data.copy_(o.data)

    def actor(self, s):
        with torch.no_grad():
            action = self.actor(s)
        if self._eps > 0:
            action += torch.randn(size=action.shape, device=action.device) * self._eps
        action = action.clamp(self._range[0], self._range[1])
        return action

    def collect(self, s_final, s_lst, a_lst, r_lst, done_lst):
        res = []
        n = len(s_lst)
        next_s = s_final
        for i in range(n - 1, -1, -1):
            res.append(DPGTransition(s_lst[i], a_lst[i], r_lst[i], next_s, done_lst[i]))
            next_s = s_lst[i]

        return res
        # batch = DDPGTransition(*zip(*res))
        # return batch

    def replay_transition(self):
        return DPGTransition

    @abstractmethod
    def learn(self, batch, **kwargs):
        pass


