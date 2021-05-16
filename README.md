# energy-py

- implementation in Tensorflow 2.0
- test episodes
- checkpoints & restarts
- logging in Tensorboard
- tested on Pendulum and LunarLanderContinuous


## Setup

```bash
make setup
```


## Train an agent from the shell

```bash
$ energypy benchmarks/battery.json
```


`Pendulum-v0` - [source](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)
`LunarLanderContinuous-v2` - [source](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py)

```bash
$ sac benchmarks/pendulum.json
```

![](assets/pendulum.png)


```bash
$ sac benchmarks/lunar.json
```

![](assets/lunar.png)



## Play gym environments

$ energypy benchmarks/lunar.json

Will load the best actor checkpoint based on average test rollouts rewards:

```bash
$ sac/play.py experiments/lunar/run-1
```


## References

[Open AI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)

[Haarnoja et. al (2018) Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) - [pdf](https://arxiv.org/pdf/1801.01290.pdf)


## Other Implementations

Open AI Spinning Up - TF 1

- [sac.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/sac.py)
- [core.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/core.py)

[Stable Baselines - TF 1](https://stable-baselines.readthedocs.io/en/master/modules/sac.html)

SLM-Lab

- [sac.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/sac.py)
- [lunar lander benchmark hyperparameters](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json)
