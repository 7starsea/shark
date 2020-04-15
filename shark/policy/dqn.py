# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F
from .base import BasePGPolicy
from .namedarraytuple import namedarraytuple

DQNTransition = namedarraytuple('DQNTransition', ('obs', 'act', 'reward', 'next_obs', 'done'))


class DQNPolicy(BasePGPolicy):
    def __init__(self, policy_net, optimizer, gamma, double_q=True):
        super().__init__('DQN', policy_net, optimizer, gamma)
        self.double_q = double_q

    def actor(self, s):
        with torch.no_grad():
            q_value = self.policy_net.forward(s)
            action = q_value.max(1)[1].view(-1, 1).clone()
        return action

    def collect(self, s_final, s_lst, a_lst, r_lst, done_lst):
        res = []
        n = len(s_lst)
        next_s = s_final
        for i in range(n - 1, -1, -1):
            res.append(DQNTransition(s_lst[i], a_lst[i], r_lst[i], next_s, done_lst[i]))
            next_s = s_lst[i]

        return res
        # batch = DQNTransition(*zip(*res))
        # return batch

    def replay_transition(self):
        return DQNTransition

    def learn(self, batch, **kwargs):
        obs, act, reward, next_obs, done = batch.apply(torch.cat)
        weights = kwargs.get('weights', None)

        with torch.no_grad():
            q_next_value = self.target_net(next_obs).detach()
        if self.double_q:
            with torch.no_grad():
                q_next_1 = self.policy_net(next_obs).detach()
                best_act = torch.argmax(q_next_1, dim=1)
            q_next_value = q_next_value.gather(1, best_act.unsqueeze(1)).squeeze(1)
        else:
            q_next_value = q_next_value.max(1)[0]

        # Compute the expected Q values
        q_next_value = reward + q_next_value * (1 - done) * self.gamma

        q_value = self.policy_net(obs).gather(1, act)
        loss = q_value.squeeze(1) - q_next_value
        td_error = torch.abs(loss.detach()).detach()
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).to(dtype=q_next_value.dtype, device=q_next_value.device)
            loss = ((loss ** 2) * weights).mean()
        else:
            loss = F.smooth_l1_loss(q_value.squeeze(1), q_next_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item(), td_error


