
# -*- coding: utf-8 -*-
import numpy as np
import torch
from types import SimpleNamespace
from gym import spaces, Env

from .SharkExampleEnv import Game2048


class Game2048Env(Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self):
        h, w = 4, 4

        # # see https://github.com/FelipeMarcelino/2048-DDQN-PER-Reinforcement-Learning
        # #     """Transform board to a power 2 matrix. Maximum value: 65536"""
        self.screen = np.zeros((16, h, w), dtype=np.uint8)
        self.observation_space = spaces.Space(shape=(16, h, w), dtype=np.uint8)
        self.action_space = spaces.Discrete(n=4)
        self.spec = SimpleNamespace(id='Game_2048')

        self._game = Game2048()

    def seed(self, seed_id):
        self._game.seed(seed_id)

    def _get_obs(self):
        state = np.zeros_like(self.screen)
        self._game.get_board(state)
        return torch.from_numpy(state)

    def reset(self):
        self._game.reset()
        return self._get_obs()

    def render(self, mode='human'):
        h, w = 4, 4
        screen = np.zeros((h, w), dtype=np.uint8)
        self._game.get_board(screen)
        print('Score %.0f Max_value %d' % (self._game.score(), self._game.max_value()))
        screen = screen.astype(int)
        ind = screen > 0
        screen[ind] = np.power(2, screen[ind])
        print(screen)
        import time
        time.sleep(0.2)

    def step(self, action):
        reward = self._game.step(action)
        valid_actions = self._game.legal_actions()
        done = 0 == len(valid_actions)

        return self._get_obs(), reward, done, dict({'valid_actions': valid_actions})

    def close(self):
        pass

