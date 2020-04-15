# coding=utf-8
import os.path
from abc import ABC, abstractmethod
from copy import deepcopy
import torch


class BasePolicy(ABC):
    def __init__(self, name, module_net):
        super().__init__()
        self.name = name
        self._module_net = module_net

    @abstractmethod
    def collect(self, obs_final, obs_lst, action_lst, reward_lst, done_lst):
        pass

    @abstractmethod
    def learn(self, batch, **kwargs):
        pass

    @abstractmethod
    def replay_transition(self):
        pass

    @abstractmethod
    def sync_target(self, tau=1):
        pass
 
    @abstractmethod
    def reset_model(self):
        pass

    def load_param(self, param):
        if isinstance(param, str):
            if os.path.isfile(param):
                param = torch.load(param)
        self._module_net.load_state_dict(param)

    def save_param(self, param_file):
        torch.save(self._module_net.state_dict(), param_file)


class BasePGPolicy(BasePolicy):
    def __init__(self, name, policy_net, optimizer, gamma):
        super().__init__(name, policy_net)
        self.policy_net = policy_net
        self.target_net = deepcopy(policy_net)
        self.target_net.eval()
        self.optimizer = optimizer

        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self.gamma = gamma

    def sync_target(self, tau=1):
        for o, n in zip(self.target_net.parameters(), self.policy_net.parameters()):
            o.data.copy_(o.data * (1 - tau) + n.data * tau)

    def reset_model(self):
        self.policy_net.load_state_dict(self.target_net.state_dict())

    @abstractmethod
    def collect(self, obs_final, obs_lst, action_lst, reward_lst, done_lst):
        pass

    @abstractmethod
    def learn(self, batch, **kwargs):
        pass

    @abstractmethod
    def replay_transition(self):
        pass

