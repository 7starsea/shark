# coding=utf-8
import os.path
import torch
import numpy as np
from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter
from shark.replay import ReplayBufferBase, SimpleReplayBuffer


class RLConfig(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.init_learn_size = batch_size * 4

        # # buffer related parameter
        self.capacity = 100000
        self.buffer = SimpleReplayBuffer
        self.buffer_kwargs = dict()

        self.double_q = True
        self.exploration = None

        self.processes = 6
        self.forward_step = 10

        self.gamma = 0.99  # # discount rate
        self.updates = 200
        self.updates_tau = 1
        self.learning_rate = 0.0001


class BaseTrainer(ABC):
    def __init__(self, config, train_env, test_env, policy, device, is_train=True, log_dir='logs', param_dir='params'):
        assert isinstance(config, RLConfig)
        # assert isinstance(policy, BasePolicy)
        self.config = config
        self.env = train_env
        self.test_env = test_env
        self.policy = policy
        self.device = device

        self.buffer = None
        tran_cls = self.policy.replay_transition()
        if issubclass(tran_cls, tuple):
            assert issubclass(config.buffer, ReplayBufferBase) and "config.buffer is not a subclass of ReplayBufferBase"
            self.buffer = config.buffer(tran_cls, config.capacity, config.batch_size, **config.buffer_kwargs)

        self._param_dir = param_dir
        self.logger = None
        if is_train:
            assert os.path.isdir(self._param_dir) and "Invalid param_dir!"
            from datetime import datetime
            assert os.path.isdir(log_dir) and "Invalid log_dir!"
            tod = datetime.today().strftime('%m-%dT%H:%M')
            log_file = '%s_%s_%s' % (self.env.spec.id, self.policy.name, tod)
            self.logger = SummaryWriter(os.path.join(log_dir, log_file))

    def __del__(self):
        try:
            if self.logger:
                self.logger.close()
        except:
            pass

    def load_param(self, param_file):
        self.policy.load_param(param_file)
        self.policy.sync_target()

    def save_param(self, frame_idx=0):
        from datetime import datetime
        tod = datetime.today().strftime('%m%dT%H%M')
        param_file = '%s_%s_%s_%d.bin' % (self.env.spec.id, self.policy.name, tod, frame_idx)
        self.policy.save_param(os.path.join(self._param_dir, param_file))

    def test(self, num_episode=1):
        total_rewards = 0
        env = self.test_env
        with torch.no_grad():
            for i in range(num_episode):

                s = env.reset()
                episode_r = 0
                while True:
                    a = self.policy.actor(s)
                    next_s, r, done, info = env.step(a.cpu().numpy())
                    s = next_s

                    env.render()

                    episode_r += r[0]
                    ind = torch.where(done > .5)[0]
                    if len(ind) > 0:
                        break
                print('Episode %d Reward %.2f' % (i + 1, episode_r))
                total_rewards += episode_r
        print('Total Reward %.2f' % total_rewards)

    @abstractmethod
    def train(self, num_frames=100000, checkpoints=10):
        pass
