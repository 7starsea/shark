# -*- coding: utf-8 -*-
import numpy as np
import PIL
import torch
import torchvision.transforms as TF
from types import SimpleNamespace

from gym import spaces, Env

from .CatchBallSimulate import CatchBallSimulate

# internal_screen_h, internal_screen_w = 80, 140


class CatchBallEnvBase(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, screen=(80, 140), num_balls=10, action_penalty=.02, waiting=0, is_continuous=False):
        self.game = CatchBallSimulate(screen, (6, 6), (5, 2), (5, 15),
                                      action_penalty=action_penalty,
                                      waiting=waiting, is_continuous=is_continuous)

        self.num_balls = num_balls
        self.index = 0
        self.screen = np.zeros(self.game.screen_size + (3,), dtype=np.uint8)

        h, w = screen
        self.observation_space = spaces.Space(shape=(h, w, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(n=3)

        self.spec = SimpleNamespace(id='CatchBall_%d' % num_balls)

        self.ax = None
        self.fig = None

    def seed(self, seed):
        self.game.seed(seed)

    def close(self):
        self.ax.clear()

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        if not self.ax:
            self.fig = plt.figure(1, figsize=(8, 10))
            self.ax = plt.subplot(111)
        self.ax.clear()

        self.screen.fill(0)
        self.game.get_display(self.screen)
        self.ax.imshow(self.screen)
        plt.pause(0.02)

    def reset(self):
        self.game.reset()
        self.index = 0

        state = np.zeros_like(self.screen)
        self.game.get_display(state)
        return state

    def reset_task(self):
        pass

    def step(self, action):
        # # in discrete-setting, action should be 0, 1, 2
        is_game_over, reward = self.game.step(int(action))

        if is_game_over:
            if self.num_balls > 0:
                self.index += 1
                is_game_over = self.index >= self.num_balls
            else:
                is_game_over = reward < .5
            self.game.reset_ball()

        next_state = np.zeros_like(self.screen)
        self.game.get_display(next_state)
        return next_state, reward, is_game_over, {}


class CatchBallEnv(CatchBallEnvBase):
    def __init__(self, *args, **kwargs):
        super(CatchBallEnv, self).__init__(*args, **kwargs)

        self.kwargs = dict(dtype=torch.float32)

        h, w, _ = self.observation_space.shape
        h, w = int(h / 2), int(w / 2)

        self.observation_space = spaces.Space(shape=(1, h, w), dtype=np.float32)

        self.composer = TF.Compose([TF.Grayscale(), TF.Resize((h, w)), TF.ToTensor()])

    def _preprocess(self, image):
        x = PIL.Image.fromarray(image)
        image = self.composer(x)
        image = image.to(**self.kwargs)
        return image

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.reshape(-1)[0])
        state, r, done, info = super().step(action)
        return self._preprocess(state), r, done, info

    def reset(self):
        state = super().reset()
        return self._preprocess(state)
