# coding=utf-8
import torch
import torch.multiprocessing as mp
import numpy as np


def worker(env_fun, worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = env_fun()
    env.seed(worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'seed':
            env.seed(data)
            worker_end.send(None)
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ParallelEnv(object):
    def __init__(self, env_fun, processes=6, device='cuda'):
        self.processes = processes
        self.waiting = False
        self.closed = False
        self.workers = list()
        self.device = torch.device(device) if isinstance(device, str) else device

        tmp_env = env_fun()

        import copy
        self.observation_space = copy.deepcopy(tmp_env.observation_space)
        self.action_space = copy.deepcopy(tmp_env.action_space)
        self.spec = copy.deepcopy(tmp_env.spec)

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.processes)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(env_fun, worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', int(action)))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return torch.stack(obs).to(device=self.device), \
               torch.from_numpy(np.stack(rews)).to(device=self.device, dtype=torch.float32), \
               torch.from_numpy(np.stack(dones)).to(device=self.device, dtype=torch.float32),\
               infos

    def reset(self, ids=None):
        if ids is None:
            for master_end in self.master_ends:
                master_end.send(('reset', None))
            obs = torch.stack([master_end.recv() for master_end in self.master_ends])
        else:
            for i in ids:
                self.master_ends[i].send(['reset', None])
            obs2 = [self.master_ends[i].recv() for i in ids]
            obs = torch.zeros(self.processes, *obs2[0].shape, dtype=obs2[0].dtype)
            obs[ids] = torch.stack(obs2)
        return obs.to(device=self.device)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

    def render(self, **kwargs):
        pass

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.processes)]
        elif seed is None:
            seed = [seed] * self.processes
        for p, s in zip(self.master_ends, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.master_ends]

