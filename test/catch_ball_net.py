# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDiscreteNet(nn.Module):
    def __init__(self, in_channel, h, w, num_action):
        super(SharedDiscreteNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return int((size + 2 * 2 - (kernel_size - 1) - 1) / stride + 1)

        convw = (conv2d_size_out(conv2d_size_out(w)))
        convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16

        # self.fc3 = nn.Linear(linear_input_size, outputs)

        # print('LinearInputSize %d Action %d' % (linear_input_size, outputs))
        output_size = 16
        self.advantage = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, num_action)
        )

        self.value = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 1)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = self.advantage(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        v = self.value(x)
        return v.squeeze(1)


class Actor(nn.Module):
    def __init__(self, in_channel, h, w, num_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return int((size + 2 * 2 - (kernel_size - 1) - 1) / stride + 1)

        convw = (conv2d_size_out(conv2d_size_out(w)))
        convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        output_size = 16
        self.advantage = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, num_action),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = self.advantage(x)
        return x * 20.0


class Critic(nn.Module):
    def __init__(self, in_channel, h, w, num_action):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return int((size + 2 * 2 - (kernel_size - 1) - 1) / stride + 1)

        convw = (conv2d_size_out(conv2d_size_out(w)))
        convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 16 + num_action
        output_size = 16
        self.value = nn.Sequential(
            nn.Linear(linear_input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 1)
        )

    def forward(self, x, a):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        v = self.value(torch.cat([x, a], dim=1))
        return v.squeeze(1)
