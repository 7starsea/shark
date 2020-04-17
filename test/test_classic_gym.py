# -*- coding: utf-8 -*-
import os.path
import numpy as np
import torch
import torch.optim as optim

from shark.env.vector_env import ParallelEnv
from shark.env.batch_env import BatchDeviceWrapper
from shark.policy.a2c import A2CPolicy
from shark.policy.ppo import PPOPolicy
from shark.policy.dqn import DQNPolicy
from shark.policy.ddpg import DDPGPolicy
from shark.policy.td3 import TD3Policy, DPGDualCriticModel
from shark.policy.sac import SACPolicy
from shark.trainer.trainer import Trainer, RLConfig
from shark.replay import SimpleReplayBuffer, PrioritizedReplayBuffer
from shark.exploration import EpsilonGreedy, GaussianNoise

import gym
from atari_net import SharedDiscreteNet, Actor, ActorProb, Critic


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super(FrameStack, self).__init__(env)

        from collections import deque
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        obs = torch.from_numpy(np.stack(tuple(self.frames), axis=0))
        return obs, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        obs = torch.from_numpy(np.stack(tuple(self.frames), axis=0))
        return obs


class TorchStateWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        # print(observation.dtype)
        return torch.from_numpy(observation).to(dtype=torch.float32)


def get_env_fun(env_id):
    my_env = gym.make(env_id)
    print(my_env.observation_space, my_env.action_space.low, my_env.action_space.high, my_env.action_space.shape)
    return TorchStateWrapper(my_env)


def get_network(env, device, policy, is_continuous=False, is_dual_critic=False):
    if is_continuous:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        if policy == 'sac':
            actor = ActorProb(state_dim, action_dim, max_action)
        else:
            actor = Actor(state_dim, action_dim, max_action)
        if is_dual_critic:
            critic = DPGDualCriticModel(Critic, state_dim, action_dim)
        else:
            critic = Critic(state_dim, action_dim)

        # actor = Actor(*env.observation_space.shape, env.action_space.n)
        # if is_dual_critic:
        #     critic = DPGDualCriticModel(Critic, *env.observation_space.shape, env.action_space.n)
        # else:
        #     critic = Critic(*env.observation_space.shape, env.action_space.n)

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

    env_id = 'Pendulum-v0'
    # env_id = 'MountainCarContinuous-v0'
    env_fun = lambda: get_env_fun(env_id)
    my_test_env = BatchDeviceWrapper(env_fun(), device=device)

    config = RLConfig(batch_size=256)
    config.gamma = .99
    # config.buffer = PrioritizedReplayBuffer
    config.updates = 100
    config.learning_rate = 0.0001/4

    config.processes = 8 if policy in ['a2c', 'ppo'] else 1
    config.epsilon_decay = 'None'

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
        actor, critic = get_network(my_test_env, device, policy, is_continuous=True, is_dual_critic=is_dual_critic)

        actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
        critic_optim = optim.Adam(critic.parameters(), lr=config.learning_rate)

        kwargs = dict(action_range=(my_test_env.action_space.low[0], my_test_env.action_space.high[0]))
        if 'td3' == policy:
            kwargs.update(dict(policy_noise=0.01, noise_clip=0.03, policy_freq=2))

        my_policy = local_policy(actor, critic, actor_optim, critic_optim, config.gamma, **kwargs)
    else:
        policy_net = get_network(my_test_env, device, policy)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)

        my_policy = local_policy(policy_net, optimizer, config.gamma)

    print('Processes %d Exploration %s Policy %s' % (config.processes, type(config.exploration), policy))
    my_dqn = Trainer(config, my_train_env, my_test_env, my_policy, device)
    if is_train:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.train(num_frames=100000)
    else:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.test()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CatchBall train/simulate')
    parser.add_argument("-t", "--train", dest="is_train", action='store_true', help="training mode", default=True)
    parser.add_argument("-s", "--simulate", dest="is_train", action='store_false', help="simulation mode")
    parser.add_argument('-p', "--policy", choices=['dqn', 'a2c', 'ppo', 'ddpg', 'td3', 'sac'], default='td3')
    parser.add_argument('--device', help="device: cpu, cuda, cuda:1", default='cuda')
    parser.add_argument('param_file', nargs='?', default=None, help="param_file")

    opt = parser.parse_args()
    train(opt.policy, opt.is_train, opt.device, opt.param_file)

# https://github.com/floodsung/DQN-Atari-Tensorflow/blob/master/BrainDQN_Nature.py
