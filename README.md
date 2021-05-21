# energy-py

`energypy` is a framework for running reinforcement learning experiments on energy environments, with a focus on electric battery storage.


- Soft Actor-Critic,
- battery environment parallelized efficiently in numpy,
- implemented in Tensorflow 2.0,
- test & train episodes based on historical Australian electricity price data,
- checkpoints & restarts,
- logging in Tensorboard.

`energypy` is built and maintained by Adam Green - adam.green@adgefficiency.com.


## Setup

```bash
$ make setup


## Running experiments

`energypy` has a high level API to run a specific run of an experiment from a `JSON` config file - examples are included in [benchmarks]().

Results are saved into `experiments/{env_name}/{run_name}`


## Run an experiment from the shell

```bash
$ energypy benchmarks/battery.json
```

The most interesting experiment is


Also wrappers around two `gym` environments:


## Run an experiment from Python

```python
from energypy import experiment
```


## Low level API
