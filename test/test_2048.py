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
from shark.example.env.game_2048 import Game2048Env

from game_2048_net import SharedDiscreteNet, Actor, Critic


def get_network(env, device):
    model = SharedDiscreteNet(*env.observation_space.shape, env.action_space.n)
    return model.to(device)


def train(policy, is_train, device='cuda', param_file=None):
    # env = Game2048Env()
    # total_score = []
    # for i in range(10):
    #     import random
    #     env.reset()
    #     # env.render()
    #     valid_actions = [0, 1, 2, 3]
    #     epsi_reward = 0
    #     while True:
    #         act = random.choice(valid_actions)
    #
    #         # print("action %d" % act, valid_actions)
    #         _, reward, done, info = env.step(act)
    #         epsi_reward += reward
    #         valid_actions = info['valid_actions']
    #         # env.render()
    #         # print()
    #         if done:
    #             break
    #
    #     total_score.append(epsi_reward)
    #     print(i, epsi_reward)
    # print('mean:', np.mean(total_score))
    # exit(0)

    is_continuous = policy in ['ddpg', 'td3', 'sac']
    assert not is_continuous and "Game 2048 only support discrete action policy algorithms."

    policy_name = policy.upper() + 'Policy'
    assert policy_name in globals() and "Failed to find policy_name"
    local_policy = globals()[policy_name]

    device = torch.device(device)
    env_fun = Game2048Env
    my_test_env = BatchDeviceWrapper(env_fun(), device=device)

    config = RLConfig(batch_size=512)
    config.gamma = .9
    config.capacity = 100000
    config.buffer = PrioritizedReplayBuffer
    config.updates = 100
    config.learning_rate = 0.0001/4

    config.processes = 8 if policy in ['a2c', 'ppo'] else 1
    config.epsilon_decay = 'None'

    if 1 == config.processes:
        config.forward_step = 1
        if not is_continuous:
            config.epsilon_decay = 'LinearFixed'

        my_train_env = BatchDeviceWrapper(env_fun(), device=device)
    else:
        config.forward_step = 10

        my_train_env = ParallelEnv(env_fun, processes=config.processes, device=device)
        my_train_env.seed(10)

    policy_net = get_network(my_test_env, device)
    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate, weight_decay=0.001)

    my_policy = local_policy(policy_net, optimizer, config.gamma)

    print('Processes %d EpsilonDecay %s Policy %s' % (config.processes, config.epsilon_decay, policy))
    my_dqn = Trainer(config, my_train_env, my_test_env, my_policy, device)
    if is_train:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.train(num_frames=800000)
    else:
        if param_file and os.path.isfile(param_file):
            my_dqn.load_param(param_file)

        my_dqn.test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Game_2048 train/simulate')
    parser.add_argument("-t", "--train",  dest="is_train", action='store_true', help="training mode", default=True)
    parser.add_argument("-s", "--simulate", dest="is_train", action='store_false', help="simulation mode")
    parser.add_argument('-p', "--policy", choices=['dqn', 'a2c', 'ppo', 'ddpg', 'td3', 'sac'], default='td3')
    parser.add_argument('--device', help="device: cpu, cuda, cuda:1", default='cuda')
    parser.add_argument('param_file', nargs='?', default=None, help="param_file")

    opt = parser.parse_args()
    train(opt.policy, opt.is_train, opt.device, opt.param_file)


# http://www.mitchellspryn.com/2017/10/28/Solving-A-Maze-With-Q-Learning.html
