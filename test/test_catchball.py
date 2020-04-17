# -*- coding: utf-8 -*-
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from shark.env.vector_env import ParallelEnv
from shark.env.batch_env import BatchDeviceWrapper
from shark.policy.a2c import A2CPolicy
from shark.policy.ppo import PPOPolicy
from shark.policy.dqn import DQNPolicy
from shark.policy.ddpg import DDPGPolicy
from shark.policy.td3 import TD3Policy, DPGDualCriticModel
from shark.trainer.trainer import Trainer, RLConfig

from shark.replay import SimpleReplayBuffer, PrioritizedReplayBuffer
from shark.exploration import EpsilonGreedy, GaussianNoise

from shark.example.env.catch_ball_env import CatchBallEnv
from catch_ball_net import SharedDiscreteNet, Actor, Critic


def get_network(env, device, is_continuous=False, is_dual_critic=False):
    if is_continuous:
        action_dim = np.product(env.action_space.shape)
        actor = Actor(*env.observation_space.shape, action_dim)
        if is_dual_critic:
            critic = DPGDualCriticModel(Critic, *env.observation_space.shape, action_dim)
        else:
            critic = Critic(*env.observation_space.shape, action_dim)
        return actor.to(device), critic.to(device)
    else:
        model = SharedDiscreteNet(*env.observation_space.shape, env.action_space.n)
        return model.to(device)


def train(policy, is_train, device='cuda', param_file=None):
    is_continuous = policy in ['ddpg', 'td3', 'sac']
    policy_name = policy.upper() + 'Policy'
    assert policy_name in globals() and "Failed to find policy_name"
    local_policy = globals()[policy_name]

    device = torch.device(device)
    env_fun = lambda: CatchBallEnv(num_balls=10, action_penalty=0.005, waiting=0, is_continuous=is_continuous)
    my_test_env = BatchDeviceWrapper(env_fun(), device=device)

    config = RLConfig(batch_size=256)
    config.gamma = .99
    config.capacity = 100000
    config.buffer = SimpleReplayBuffer
    config.updates = 100
    config.learning_rate = 0.0001

    config.processes = 8 if policy in ['a2c', 'ppo'] else 1

    if 'dqn' == policy:
        config.exploration = EpsilonGreedy(my_test_env.action_space.n)
    elif policy in ['td3', 'ddpg']:
        low, high = my_test_env.action_space.low[0], my_test_env.action_space.high[0]
        config.exploration = GaussianNoise((high - low) / 2)

    if 1 == config.processes:
        config.forward_step = 1
        my_train_env = BatchDeviceWrapper(env_fun(), device=device)
    else:
        config.forward_step = 10
        my_train_env = ParallelEnv(env_fun, processes=config.processes, device=device)
        my_train_env.seed(10)

    if is_continuous:
        is_dual_critic = policy in ['td3', 'sac']
        actor, critic = get_network(my_test_env, device, is_continuous=True, is_dual_critic=is_dual_critic)
        
        actor_optim = optim.Adam(actor.parameters(), lr=config.learning_rate, weight_decay=0.001)
        critic_optim = optim.Adam(critic.parameters(), lr=config.learning_rate, weight_decay=0.001)

        kwargs = dict(action_range=(my_test_env.action_space.low[0], my_test_env.action_space.high[0]))
        if 'td3' == policy:
            kwargs.update(dict(policy_noise=0.3, noise_clip=0.51, policy_freq=2))
        print(policy, kwargs)
        my_policy = local_policy(actor, critic, actor_optim, critic_optim, config.gamma, **kwargs)
    else:
        policy_net = get_network(my_test_env, device)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate, weight_decay=0.001)

        my_policy = local_policy(policy_net, optimizer, config.gamma)

    print('Processes %d Exploration %s Policy %s' % (config.processes, type(config.exploration), policy))
    my_dqn = Trainer(config, my_train_env, my_test_env, my_policy, device)
    if is_train:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.train(num_frames=200000)
    else:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CatchBall train/simulate')
    parser.add_argument("-t", "--train",  dest="is_train", action='store_true', help="training mode", default=True)
    parser.add_argument("-s", "--simulate", dest="is_train", action='store_false', help="simulation mode")
    parser.add_argument('-p', "--policy", choices=['dqn', 'a2c', 'ppo', 'ddpg', 'td3', 'sac'], default='td3')
    parser.add_argument('--device', help="device: cpu, cuda, cuda:1", default='cuda')
    parser.add_argument('param_file', nargs='?', default=None, help="param_file")

    opt = parser.parse_args()
    train(opt.policy, opt.is_train, opt.device, opt.param_file)


# http://www.mitchellspryn.com/2017/10/28/Solving-A-Maze-With-Q-Learning.html
