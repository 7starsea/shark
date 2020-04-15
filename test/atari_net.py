# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDiscreteNet(nn.Module):
    def __init__(self, in_channel, h, w, outputs):
        super(SharedDiscreteNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.layer = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.conv2,
#            self.bn2,
            nn.ReLU(),
            self.conv3, self.bn3, nn.ReLU()
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return int((size - (kernel_size - 1) - 1) / stride + 1)

        convw = (conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4), kernel_size=4))
        convh = (conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4), kernel_size=4))

        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 16
        output_size = 256
        self.advantage = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 1)
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32) / 255.0

        x = self.layer(x)
        x = x.view(x.size(0), -1)

        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

    def pi(self, x, softmax_dim=1):
        x = x.to(dtype=torch.float32) / 255.0

        x = self.layer(x)
        x = x.view(x.size(0), -1)

        x = self.advantage(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = x.to(dtype=torch.float32) / 255.0

        x = self.layer(x)
        x = x.view(x.size(0), -1)

        v = self.value(x)
        return v.squeeze(1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 30)
        self.l3 = nn.Linear(30, action_dim)

        self._max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self._max_action * torch.tanh(self.l3(x))
        return x


class ActorProb(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-20, log_std_max=2):
        super(ActorProb, self).__init__()

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 30)
        self.mu = nn.Linear(30, action_dim)
        self.sigma = nn.Linear(30, action_dim)

        self._max_action = max_action
        self._log_std_min, self._log_std_max = log_std_min, log_std_max

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self._max_action * torch.tanh(self.mu(x))
        sigma = torch.exp(torch.clamp(self.sigma(x), self._log_std_min, self._log_std_max))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 30)
        self.l3 = nn.Linear(30, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze(1)
