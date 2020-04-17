# coding=utf-8
import os.path
import random
import torch
from torch.distributions import Categorical
import numpy as np
from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
from shark.replay.replay_base import ReplayBufferBase, PrioritizedReplayBufferBase
from shark.exploration.exploration_base import ExplorationBase
from .trainer_base import RLConfig, BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, train_env, test_env, policy, device, is_train=True):
        super().__init__(config, train_env, test_env, policy, device, is_train=is_train)

    def train(self, num_frames=100000, checkpoints=4):
        s = self.env.reset()
        config = self.config
        episode_r = torch.zeros(config.processes, device=self.device)
        checkpoints = int(num_frames / checkpoints) + 1

        # percentile = 75
        # buffer_size = 8
        # buffer = deque(maxlen=buffer_size)

        exploration_sub_args = ()
        if isinstance(config.exploration, ExplorationBase):
            if config.exploration.is_continuous():
                a = self.policy.actor(s)
                exploration_sub_args = a.shape, a.dtype
            else:
                exploration_sub_args = (s.shape[0], 1), torch.long

        episode_r_index = 0

        desc = '%s_%s' % (self.env.spec.id, self.policy.name)
        for frame_idx in tqdm(range(1, num_frames + 1), desc=desc, ncols=120, mininterval=.4):
            s_lst, a_lst, r_lst, done_lst = list(), list(), list(), list()
            next_s = s
            with torch.no_grad():
                episode_r_mean = 0.0
                episode_r_num = 0
                for _ in range(config.forward_step):
                    if isinstance(config.exploration, ExplorationBase):
                        x = frame_idx / (num_frames + 1.0)
                        noise = config.exploration.sample(x, *exploration_sub_args, device=self.device)
                        a = self.policy.actor(s, noise=noise)
                    else:
                        a = self.policy.actor(s)

                    next_s, r, done, info = self.env.step(a.cpu().numpy())

                    episode_r += r
                    ind = torch.where(done > .5)[0]
                    if len(ind) > 0:
                        episode_r_mean += (episode_r[ind].mean()).item()
                        episode_r_num += 1
                        episode_r[ind] = 0

                        if 1 == config.processes:
                            next_s = self.env.reset()
                        else:
                            s_obs = self.env.reset(ind)
                            next_s[ind] = s_obs[ind]

                    s_lst.append(s), a_lst.append(a), r_lst.append(r), done_lst.append(done)
                    s = next_s

            if isinstance(self.buffer, ReplayBufferBase):
                res = self.policy.collect(next_s, s_lst, a_lst, r_lst, done_lst)
                if isinstance(res, list):
                    for tran in res:
                        self.buffer.append(tran)
                else:
                    self.buffer.append(res)

                if len(self.buffer) >= config.init_learn_size:
                    if isinstance(self.buffer, PrioritizedReplayBufferBase):
                        beta = 0.4 + 0.6 * frame_idx / num_frames
                        batch, indices, weights, priorities = self.buffer.sample(beta=beta)
                        if isinstance(indices, np.ndarray):
                            loss, td_err = self.policy.learn(batch, weights=weights)
                            self.logger.add_scalar('train/loss', loss, frame_idx)
                            self.buffer.update_priorities(indices, priorities, td_err.cpu().numpy())
                    else:
                        batch = self.buffer.sample()
                        loss, td_err = self.policy.learn(batch)
                        self.logger.add_scalar('train/loss', loss, frame_idx)

            else:
                batch = self.policy.collect(next_s, s_lst, a_lst, r_lst, done_lst)
                assert isinstance(batch, tuple)
                loss, td_err = self.policy.learn(batch)

                # tmp = torch.sort(td_err)[0][int(td_err.numel() * percentile / 100)]
                # ind = torch.where(td_err >= tmp)[0]
                # buffer.append(batch[ind])
                # if len(buffer) == buffer_size:
                #     tran_cls = type(batch)
                #     data = list(zip(*list(buffer)))
                #     batch_err = tran_cls(*[torch.cat(item) for item in data])
                #     loss, td_err = self.policy.learn(batch_err)

                self.logger.add_scalar('train/loss', loss, frame_idx)

            if episode_r_num > 0:
                self.logger.add_scalar('train/reward', episode_r_mean / episode_r_num, episode_r_index)
                episode_r_index += 1

            if 0 == frame_idx % config.updates:
                self.policy.sync_target(config.updates_tau)
            if 0 == frame_idx % checkpoints:
                self.save_param(frame_idx)
        self.save_param()
