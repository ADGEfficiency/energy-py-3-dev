"""
give it test & train data
check we only get one episode etc
"""

from energypy.registry import make

#  def test_nem_dataset_train_test():
env = make(
    'battery',
    episode_length=64,
    dataset={'name': 'nem-dataset'},
    n_batteries=1,
)


from collections import defaultdict

env.reset(mode='train')
results = defaultdict(list)
done = False
while not done:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
