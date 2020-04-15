# -*- coding: utf-8 -*-
import numpy as np


def fetch_epsilon_decay(num_frames=10000, eps_beg=.99, eps_end=.1, eps_decay=1000, decay_policy='LinearFixed'):
    if decay_policy == 'Linear':
        epsilon_s = eps_end + (eps_beg - eps_end) * (1 - np.arange(num_frames + 1) / num_frames)
    elif decay_policy == 'LinearExp':
        eps_mid = eps_beg / 2
        half_num_frames = int(num_frames / 2)
        epsilon_s_1 = eps_mid + (eps_beg - eps_mid) * (1 - np.arange(half_num_frames + 1) / half_num_frames)
        epsilon_s_2 = eps_end + (eps_mid - eps_end) * np.exp(-np.arange(num_frames - half_num_frames + 1) / eps_decay)
        epsilon_s = np.concatenate([epsilon_s_1, epsilon_s_2])
    elif decay_policy == 'LinearFixed':
        epsilon_s_1 = eps_end + (eps_beg - eps_end) * (1 - np.arange(eps_decay + 1) / eps_decay)
        epsilon_s_2 = eps_end * np.ones(num_frames - eps_decay)
        epsilon_s = np.concatenate([epsilon_s_1, epsilon_s_2])
    else:
        epsilon_s = eps_end + (eps_beg - eps_end) * np.exp(-np.arange(num_frames + 1) / eps_decay)

    assert len(epsilon_s) >= num_frames + 1
    return epsilon_s
