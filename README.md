# energypy

energy-py is a framework for running reinforcement learning experiments on energy environments.

The library is focused on electric battery storage, and offers a implementation of a many batteries operating in parallel.

energy-py includes an implementation of the Soft Actor-Critic reinforcement learning agent, implementated in Tensorflow 2.

- test & train episodes based on historical Australian electricity price data,
- checkpoints & restarts,
- logging in Tensorboard.

energy-py is built and maintained by Adam Green - adam.green@adgefficiency.com.


## Setup

```bash
$ make setup
```


## Running experiments

`energypy` has a high level API to run a specific run of an experiment from a `JSON` config file - examples are included in [benchmarks]():

```bash
$ energypy benchmarks/battery.json
```

Results are saved into `./experiments/{env_name}/{run_name}`:

```bash
$ tree -L 3 experiments
experiments/
└── battery
    ├── eight
    │   ├── checkpoints
    │   ├── hyperparameters.json
    │   ├── logs
    │   └── tensorboard
    └── random.pkl

```

Also wrappers around two `gym` environments:

```bash
$ energypy benchmarks/battery.json
```
