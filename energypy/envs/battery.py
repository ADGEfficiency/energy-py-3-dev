from collections import namedtuple
import numpy as np

from energypy import registry


def battery_energy_balance(initial_charge, final_charge, import_energy, export_energy, losses):
    delta_charge = final_charge - initial_charge
    balance = import_energy - (export_energy + losses + delta_charge)
    np.testing.assert_almost_equal(balance, 0)


def calculate_losses(delta_charge, efficiency):
    #  delta_charge = battery delta_charge charge
    delta_charge = np.array(delta_charge)
    efficiency = np.array(efficiency)
    #  account for losses / the round trip efficiency
    #  we lose electricity when we discharge
    losses = delta_charge * (1 - efficiency)
    losses = np.array(losses)
    losses[delta_charge > 0] = 0
    return np.abs(losses)


def set_battery_config(value, n_batteries):
    if isinstance(value, list):
        return np.array(value).reshape(1, n_batteries, 1)
    else:
        return np.full((n_batteries, 1), value).reshape(1, n_batteries, 1)


class BatteryObservationSpace:
    def __init__(self, dataset):
        self.shape = dataset.dataset['features'].shape[1:]


class BatteryActionSpace:
    def __init__(self, n_batteries=2):
        self.n_batteries = n_batteries
        self.shape = (self.n_batteries, 1)

        self.low = -1
        self.high = 1

    def sample(self):
        return np.random.uniform(-1, 1, self.n_batteries).reshape(1, *self.shape)

    def contains(self, action):
        assert (action <= 1.0).all()
        assert (action >= -1.0).all()
        return True


class Battery:
    """
    data = (n_battery, timesteps, features)
    """
    def __init__(
        self,
        n_batteries=2,
        power=2.0,
        capacity=4.0,
        efficiency=0.9,
        initial_charge=0.0,
        episode_length=288,
        dataset={'name': 'random-dataset'}
    ):
        self.n_batteries = n_batteries

        self.power = set_battery_config(power, n_batteries)
        self.capacity = set_battery_config(capacity, n_batteries)
        self.efficiency = set_battery_config(efficiency, n_batteries)
        self.initial_charge = set_battery_config(initial_charge, n_batteries)

        self.episode_length = int(episode_length)

        dataset['n_batteries'] = n_batteries
        self.dataset = registry.make(**dataset)

        self.observation_space = BatteryObservationSpace(self.dataset)
        self.action_space = BatteryActionSpace(n_batteries)

        self.elements = (
            ('observation', self.observation_space.shape, 'float32'),
            ('action', self.action_space.shape, 'float32'),
            ('reward', (self.n_batteries, ), 'float32'),
            ('next_observation', self.observation_space.shape, 'float32'),
            ('done', (1, ), 'bool')
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])

    def reset(self, mode='train'):
        len_dataset = 1000
        self.cursor = 0
        self.charge = self.initial_charge

        self.dataset.reset(mode)
        data = self.dataset.get_data(self.cursor)
        self.cursor += 1
        return data['features']

    def step(self, action):
        action = action.reshape(1, self.n_batteries, 1)

        #  expect a scaled action here
        #  -1 = discharge max, 1 = charge max
        action = np.clip(action, -1, 1)
        action = action * self.power

        #  convert from power to energy, kW -> kWh
        action = action / 12

        #  charge at the start of the 5 min interval, kWh
        initial_charge = self.charge

        #  charge at end of the interval
        #  clipped at battery capacity, kWh
        final_charge = np.clip(initial_charge + action, 0, self.capacity)

        #  accumulation in battery, kWh
        #  delta_charge can also be thought of as gross_power
        delta_charge = final_charge - initial_charge

        #  losses are applied when we discharge, kWh
        losses = calculate_losses(delta_charge, self.efficiency)

        #  net of losses, kWh
        #  add losses here because in delta_charge, export is negative
        #  to reduce export, we add a positive losses
        net_energy = delta_charge + losses

        import_energy = np.zeros_like(net_energy)
        import_energy[net_energy > 0] = net_energy[net_energy > 0]

        export_energy = np.zeros_like(net_energy)
        export_energy[net_energy < 0] = np.abs(net_energy[net_energy < 0])

        #  set charge for next timestep
        self.charge = initial_charge + delta_charge

        battery_energy_balance(
            initial_charge,
            final_charge,
            import_energy,
            export_energy,
            losses
        )

        price = self.dataset.get_data(self.cursor)['prices'].reshape(1, self.n_batteries,  -1)
        price = np.array(price).reshape(1, self.n_batteries, 1)
        reward = export_energy * price - import_energy * price

        self.cursor += 1
        next_obs = self.dataset.get_data(self.cursor)['features'].reshape(1, self.n_batteries,  -1)
        done = np.array(self.cursor == (self.episode_length + 1))

        info = {
            'cursor': self.cursor,
            'episode_length': self.episode_length,
            'done': done,
            'charge': self.charge
        }

        return next_obs, reward, done, info
