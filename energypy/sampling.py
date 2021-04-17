import numpy as np

from energypy import random_policy


def episode(
    env,
    buffer,
    actor,
    hyp,
    counters,
    rewards,
    mode,
    logger=None
):
    obs = env.reset(mode=mode)
    done = False

    reward_scale = hyp['reward-scale']
    episode_rewards = []

    while not done:
        act, _, deterministic_action = actor(obs)

        if mode == 'test':
            act = deterministic_action

        next_obs, reward, done, _ = env.step(np.array(act))
        episode_rewards.append(np.mean(reward))

        for o, a, r, no in zip(obs, act, reward, next_obs):
            buffer.append(env.Transition(o, a, r/reward_scale, no, done))

        # if logger:
        #     logger.debug(
        #         f'{obs}, {act}, {reward}, {next_obs}, {done}, {mode}'
        #     )

        counters['env-steps'] += 1
        obs = next_obs

    counters['episodes'] += 1
    counters[f'{mode}-episodes'] += 1

    return episode_rewards


def run_episode(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    mode,
    logger=None
):
    episode_rewards = episode(
        env,
        buffer,
        actor,
        hyp,
        counters,
        rewards,
        mode,
        logger=logger
    )
    episode_reward = float(np.sum(episode_rewards))

    rewards['episode-reward'].append(episode_reward)
    rewards[f'{mode}-reward'].append(episode_reward)

    writers[mode].scalar(
        episode_reward,
        f'{mode}-episode-reward',
        f'{mode}-episodes',
        verbose=True
    )
    writers['episodes'].scalar(
        episode_reward,
        'episode-reward',
        'episodes'
    )
    return episode_rewards


def sample_random(
    env,
    buffer,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'random'
    print(f"filling buffer with {buffer.size} samples")
    policy = random_policy.make(env)

    while not buffer.full:
        run_episode(
            env,
            buffer,
            policy,
            hyp,
            writers,
            counters,
            rewards,
            mode,
            logger=logger
        )

    assert len(buffer) == buffer.size
    print(f"buffer filled with {len(buffer)} samples\n")
    return buffer


def sample_test(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'test'

    env.setup_test()

    test_results = []
    test_done = False
    while not test_done:

        test_rewards = run_episode(
            env,
            buffer,
            actor,
            hyp,
            writers,
            counters,
            rewards,
            mode,
            logger=logger
        )
        test_results.append(sum(test_rewards))
        test_done = env.test_done

    return test_results


def sample_train(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'train'
    return run_episode(
        env,
        buffer,
        actor,
        hyp,
        writers,
        counters,
        rewards,
        mode,
        logger=logger
    )
