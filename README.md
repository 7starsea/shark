# Shark
## Reinforcement Learning (rl) Project with Pytorch
================
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Shark is inspired by several rl-platform https://github.com/thu-ml/tianshou and https://github.com/astooke/rlpyt and https://github.com/openai/baselines/ and many independent rl-algorithm implementations in github.

## Implemented Algorithms
**Replay Buffers** supporting DQN, uniform replay, prioritized experience replay

**Policy Gradient** A2C, PPO.

**Q-Function Policy Gradient** DDPG, TD3, SAC(todo).

## Features
*Provide a fast (cpp-version) implementation of prioritized experience replay buffer. (see also https://github.com/7starsea/Prioritized-Experience-Replay)
*Provide a CatchBall env, see shark/example/env


## Requirement
*python3, (tested on 3.6, 3.7)
**pytorch 1.2
**gym 0.15.0
**tqdm
**scikit-build, (tested on 0.10.0)
*pybind11, (tested on 2.4.3)
*cmake3, (tested on 3.14)

(we use ananconda enviroment for testing)

## Compile [tested on ubuntu]
clone the repo by `git clone https://github.com/7starsea/shark.git`, and on linux, standary python install is like:
`
cd shark
python setup.py install
`
if your python env already satisfy requirements, you could just do local build:
`
cd shark 
./compile.sh
`

## test
`
cd test
 # # Note for local build, you should copy these *py files to parent directory and run.
mkdir -p logs params  ## 
python test_catchball -p dqn
python test_atari.py -p a2c
`

## Contributing
Shark is still in infancy, we welcome contributions from everywhere to help make Shark better.
 
