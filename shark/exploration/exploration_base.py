

from abc import ABC, abstractmethod


class ExplorationBase(ABC):
    def __init__(self, initial_eps=1.0, final_eps=0.1, fixed_ratio=.7):
        assert 0 <= final_eps < initial_eps <= 1 and 0 < fixed_ratio < 1 and "Invalid parameters."
        self._beg_eps = initial_eps
        self._end_eps = final_eps
        self._fixed_ratio = fixed_ratio

    def eps(self, x):
        """
        :param x: training_ratio
        :return:
        """
        if x <= self._fixed_ratio:
            eps = (1.0 - x / self._fixed_ratio) * (self._beg_eps - self._end_eps) + self._end_eps
        else:
            eps = self._end_eps
        return eps

    @abstractmethod
    def is_continuous(self):
        pass

    @abstractmethod
    def sample(self, training_ratio, shape, dtype, device=None):
        """
        :param training_ratio:  a value between (0, 1) indicating the training progress
        :param shape:
        :param dtype:
        :param device:
        :return:
        """
        pass
