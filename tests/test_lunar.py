from energypy.agent.random_policy import make as make_random_policy
from energypy.envs.gym_wrappers import GymWrapper


def test_envs():
    env = GymWrapper('pendulum')
    policy = make_random_policy(env)

    env = GymWrapper('lunar')
    policy = make_random_policy(env)
