# Shark - Reinforcement Learning (rl) Project with Pytorch

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Shark is inspired by several rl-platform https://github.com/thu-ml/tianshou and https://github.com/astooke/rlpyt and https://github.com/openai/baselines/ and many independent rl-algorithm implementations (e.g. https://github.com/fanchenyou/RL-study/) in github.

## Implemented Algorithms
**Replay Buffers** supporting DQN, uniform replay, prioritized experience replay

**Policy Gradient** A2C, PPO.

**Q-Function Policy Gradient** DDPG, TD3, SAC.

## Features
* Provide a fast (cpp-version) implementation of prioritized experience replay buffer. (see also https://github.com/7starsea/Prioritized-Experience-Replay)
* Provide envs: CatchBall and Game2048, see shark/example/env


## Requirements
* python3, (tested on 3.6, 3.7)
* pytorch 1.2
* gym 0.15.0
* tqdm
* scikit-build, (tested on 0.10.0)
* pybind11, (tested on 2.4.3)
* cmake3, (tested on 3.14)

(we use ananconda enviroment for testing)

## Compile [tested on ubuntu]
on linux, clone a copy by `git clone https://github.com/7starsea/shark.git`, and install with
```bash
cd shark
python setup.py install
```
you can also install a local copy (mainly for building cpp) using `./compile.sh`.

## test
```bash
cd test
mkdir -p logs params  ## for local install, also need to add a symbolic link: ln -s ../shark shark
python test_catchball -p dqn
python test_catchball -p td3
python test_atari.py -p a2c
```

## Contributing
Shark is still in infancy, we welcome contributions from everywhere to make Shark better.
 
