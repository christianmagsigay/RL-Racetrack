import gym
import numpy as np


def sample_episode(env, policy, render=False):
    """ Follow policy through an episode and return arrays of visited actions, states, and returns """
    choices_ridxs = np.arange(int(np.prod(env.action_space.nvec)))
    state_ridxs = []
    action_ridxs = []
    rewards = []

    done = False
    state = env.reset()
    if render:
        env.render()

    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        state_ridxs += [state_ridx]

        # Sample action from the policy
        action_ridx = np.random.choice(choices_ridxs, p=policy[state_ridx])
        action = np.array(np.unravel_index(action_ridx, env.action_space.nvec))
        action_ridxs += [action_ridx]

        # Step the environment forward and take the sampled action
        state, reward, done, info = env.step(action)
        rewards += [reward]

        if render:
            env.render()

    # Returns without discounting
    returns = np.cumsum(rewards[::-1])[::-1]

    assert len(state_ridxs) == len(action_ridxs) == len(returns)
    return state_ridxs, action_ridxs, returns


def monte_carlo_control_eps_soft(env, num_episodes, eps=0.10, alpha=0.05):
    """ Every-visit Monte Carlo algorithm for eps-soft policies per Chapter 5.4 """

    assert type(env.action_space) == gym.spaces.MultiDiscrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    n_action_ridx = np.ravel_multi_index(env.action_space.nvec - 1, env.action_space.nvec) + 1
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1

    q = np.ones([n_state_ridx, n_action_ridx], dtype=float)
    policy = np.ones([n_state_ridx, n_action_ridx], dtype=float) / n_action_ridx

    # store returns
    returns_log = []

    for episode in range(num_episodes):
        state_ridxs, action_ridxs, returns = sample_episode(env, policy)

        # log total return (episode performance)
        returns_log.append(returns[0])

        # Q update
        for state_ridx, action_ridx, retrn in zip(state_ridxs, action_ridxs, returns):
            q[state_ridx, action_ridx] += alpha * (retrn - q[state_ridx, action_ridx])

        # Policy update
        visited_states = np.unique(state_ridxs)
        for s in visited_states:
            greedy_a = np.argmax(q[s])
            policy[s, :] = eps / n_action_ridx
            policy[s, greedy_a] = 1 - eps + eps / n_action_ridx

        assert np.allclose(np.sum(policy, axis=1), 1)

        if episode % 10_000 == 0:
            print(f"Episode {episode}/{num_episodes}: return={returns[0]}")

    # Final greedy policy
    greedy_action_ridxs = np.argmax(q, axis=1)
    policy[:, :] = 0
    policy[np.arange(n_state_ridx), greedy_action_ridxs] = 1

    # RETURN learning curve too
    return q, policy, returns_log

