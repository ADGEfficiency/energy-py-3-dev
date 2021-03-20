from energypy.envs.battery import Battery
from energypy.envs.gym_wrappers import GymWrapper

from energypy.datasets import *


registry = {
    'lunar': GymWrapper,
    'pendulum': GymWrapper,

    'battery': Battery,

    'random-dataset': RandomDataset,
    'nem-dataset': NEMDataset,
}


def make(name=None, *args, **kwargs):
    if name is None:
        name = kwargs['name']
    return registry[name](*args, **kwargs)
