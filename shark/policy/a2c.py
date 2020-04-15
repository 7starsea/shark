# coding=utf-8

import torch
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical
import numpy as np
from .base import BasePGPolicy
from .namedarraytuple import namedarraytuple

A2CTransition = namedarraytuple('A2CTransition', ('obs', 'act', 'v_label'))


# def compute_target2(v_final, r_lst, done_lst, gamma):
#     n = len(done_lst)
#
#     r_lst.append(v_final)
#     r = torch.stack(r_lst, dim=0)
#     mask = 1.0 - torch.stack(done_lst, dim=0)
#     mask *= gamma
#     for i in range(n-1, -1, -1):
#         r[i] += r[i + 1] * mask[i]
#     return r[:n].reshape(-1)


def compute_target(v_final, r_lst, done_lst, gamma):
    G = v_final
    td_target = list()

    for r, done in zip(r_lst[::-1], done_lst[::-1]):
        G = r + gamma * G * (1 - done)
        td_target.append(G)

    return torch.cat(td_target[::-1])


class A2CPolicy(BasePGPolicy):
    def __init__(self, policy_net, optimizer, gamma, dist_fn=Categorical,
                 vf_coef=.6, ent_coef=.01, max_grad_norm=.5):
        super().__init__('A2C', policy_net, optimizer, gamma)
        self.dist_fn = dist_fn

        self.w_vf = vf_coef
        self.w_ent = ent_coef
        self.max_grad_norm = max_grad_norm

    def actor(self, s):
        prob = self.policy_net.pi(s, softmax_dim=1)
        a = self.dist_fn(prob).sample()
        return a

    def critic(self, s):
        return self.policy_net.v(s)

    def collect(self, s_final, s_lst, a_lst, r_lst, done_lst):
        with torch.no_grad():
            v_final = self.policy_net.v(s_final).detach()
            # v_final = self.target_net.v(s_final).detach()
        v_label = compute_target(v_final, r_lst, done_lst, self.gamma)

        s_vec = torch.cat(s_lst)
        a_vec = torch.cat(a_lst)

        return A2CTransition(s_vec, a_vec, v_label)

    def replay_transition(self):
        return object

    def learn(self, batch, **kwargs):
        s_vec, a_vec, v_label = batch.obs, batch.act, batch.v_label
        v_hat = self.policy_net.v(s_vec)

        advantage = (v_label - v_hat).detach()
        td_err = torch.abs(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        pi = self.policy_net.pi(s_vec, softmax_dim=1)

        # # policy loss (Q-network) + critic loss (Critic)
        # # loss = -(torch.log(pi_a) * advantage).mean() + advantage.pow(2).mean()
        m = self.dist_fn(pi)
        loss = -(m.log_prob(a_vec) * advantage).mean() + self.w_vf * F.mse_loss(v_hat, v_label) \
               - self.w_ent * m.entropy().mean()

        # # policy loss (Q-network) + critic loss (Critic)
        # # loss = -(torch.log(pi_a) * advantage).mean() + advantage.pow(2).mean()
        # pi_a = pi.gather(1, a_vec.unsqueeze(1)).squeeze(1)
        # print(m.log_prob(a_vec).requires_grad, torch.log(pi_a).requires_grad)
        # assert (torch.sum(torch.abs(m.log_prob(a_vec) - torch.log(pi_a)))) <= 1.0e-4
        # loss = -(torch.log(pi_a) * advantage).mean() + F.smooth_l1_loss(v_hat, v_label)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), td_err

