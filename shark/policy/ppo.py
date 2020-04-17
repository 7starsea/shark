# coding=utf-8

import torch
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical
import numpy as np
from .base import BasePGPolicy
from .namedarraytuple import namedarraytuple


PPOTransition = namedarraytuple('PPOTransition', ('obs', 'act', 'v_label'))


def compute_target(v_final, r_lst, done_lst, gamma):
    G = v_final
    td_target = list()

    for r, done in zip(r_lst[::-1], done_lst[::-1]):
        G = r + gamma * G * (1 - done)
        td_target.append(G)

    return torch.cat(td_target[::-1])


class PPOPolicy(BasePGPolicy):
    def __init__(self, policy_net, optimizer, gamma, dist_fn=Categorical,
                 eps_clip=0.2, vf_coef=.8, ent_coef=.01, max_grad_norm=.5, k_epochs=3):
        super().__init__('PPO', policy_net, optimizer, gamma)
        self.dist_fn = dist_fn
        self.eps_clip = eps_clip

        self.w_vf = vf_coef
        self.w_ent = ent_coef
        self.max_grad_norm = max_grad_norm
        self.k_epochs = k_epochs

    def actor(self, s, noise=None):
        prob = self.policy_net.pi(s, softmax_dim=1)
        a = self.dist_fn(prob).sample()
        return a

    def critic(self, s):
        return self.policy_net.v(s)

    def collect(self, s_final, s_lst, a_lst, r_lst, done_lst):
        with torch.no_grad():
            v_final = self.policy_net.v(s_final).detach()
        v_label = compute_target(v_final, r_lst, done_lst, self.gamma)

        s_vec = torch.cat(s_lst)
        a_vec = torch.cat(a_lst)

        # s_vec_next = s_vec.clone()
        # s_vec_next[:-1] = s_vec[1:]
        # s_vec_next[-1] = s_final

        return PPOTransition(s_vec, a_vec, v_label)

    def replay_transition(self):
        return object

    def learn(self, batch, **kwargs):
        s_vec, a_vec, v_label = batch.obs, batch.act, batch.v_label

        with torch.no_grad():
            dist_old = self.dist_fn(self.target_net.pi(s_vec).detach())
        log_prob_old = dist_old.log_prob(a_vec)

        losses, td_errors = [], []
        for _ in range(self.k_epochs):
            v_hat = self.policy_net.v(s_vec)
            advantage = (v_label - v_hat).detach()
            td_errors.append(torch.abs(advantage))
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

            dist = self.dist_fn(self.policy_net.pi(s_vec))
            ratio = torch.exp(dist.log_prob(a_vec) - log_prob_old)
            surr1 = ratio * advantage
            surr2 = ratio.clamp(1. - self.eps_clip, 1 + self.eps_clip) * advantage
            clip_loss = -torch.min(surr1, surr2).mean()

            e_loss = dist.entropy().mean()
            vf_loss = F.mse_loss(v_hat, v_label)
            loss = clip_loss + self.w_vf * vf_loss - self.w_ent * e_loss

            # pi = self.policy_net.pi(s_vec, softmax_dim=1)
            #
            # # # policy loss (Q-network) + critic loss (Critic)
            # # # loss = -(torch.log(pi_a) * advantage).mean() + advantage.pow(2).mean()
            # m = self.dist_fn(pi)
            # loss = -(m.log_prob(a_vec) * advantage).mean() + F.smooth_l1_loss(v_hat, v_label)

            # # policy loss (Q-network) + critic loss (Critic)
            # # loss = -(torch.log(pi_a) * advantage).mean() + advantage.pow(2).mean()
            # pi_a = pi.gather(1, a_vec.unsqueeze(1)).squeeze(1)
            # loss = -(torch.log(pi_a) * advantage).mean() + F.smooth_l1_loss(v_hat, v_label)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            losses.append(loss.item())
        self.sync_target()
        return np.mean(losses), torch.mean(torch.stack(td_errors, dim=1), dim=1)

