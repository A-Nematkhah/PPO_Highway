"""
buffer.py

RolloutBuffer: stores one rollout (N_STEPS x N_ENVS transitions) and computes
advantages/returns via Generalized Advantage Estimation (GAE).

Storage layout is (n_steps, n_envs, ...) so that "flattening" for minibatch
sampling later is a simple reshape.
"""

import numpy as np

from config import GAE_LAMBDA, GAMMA, N_ENVS, N_STEPS


class RolloutBuffer:
    def __init__(self, n_steps: int = N_STEPS, n_envs: int = N_ENVS, obs_dim: int = 75):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim

        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """
        Adds one timestep of data (across all n_envs) at the current pointer.

        obs:      (n_envs, obs_dim)
        action:   (n_envs,)
        log_prob: (n_envs,)
        reward:   (n_envs,)
        done:     (n_envs,)   -- done flag from *before* this step (i.e. was
                                 the env reset at the start of this transition)
        value:    (n_envs,)
        """
        assert self.ptr < self.n_steps, "RolloutBuffer overflow - call reset() first"

        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray,
                     gamma: float = GAMMA, lam: float = GAE_LAMBDA):
        """
        Computes advantages and returns via GAE, working backwards through
        the rollout.

        last_values: (n_envs,) - V(s_T) for the observation *after* the last
                                  stored step (used to bootstrap).
        last_dones:  (n_envs,) - done flags for that same final observation.

        Returns:
            advantages: (n_steps, n_envs)
            returns:    (n_steps, n_envs)   = advantages + values
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def get_flat(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Flattens (n_steps, n_envs, ...) tensors into (n_steps * n_envs, ...)
        for minibatch sampling during the PPO update.
        """
        b_obs = self.obs.reshape(-1, self.obs_dim)
        b_actions = self.actions.reshape(-1)
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values


if __name__ == "__main__":
    # quick sanity check with random data
    buf = RolloutBuffer(n_steps=8, n_envs=2, obs_dim=75)

    for t in range(8):
        buf.add(
            obs=np.random.randn(2, 75).astype(np.float32),
            action=np.random.randint(0, 5, size=2),
            log_prob=np.random.randn(2).astype(np.float32),
            reward=np.random.randn(2).astype(np.float32),
            done=np.zeros(2, dtype=np.float32),
            value=np.random.randn(2).astype(np.float32),
        )

    last_values = np.random.randn(2).astype(np.float32)
    last_dones = np.zeros(2, dtype=np.float32)

    advantages, returns = buf.compute_gae(last_values, last_dones)
    print("advantages.shape:", advantages.shape)
    print("returns.shape:", returns.shape)

    b_obs, b_actions, b_log_probs, b_adv, b_ret, b_val = buf.get_flat(advantages, returns)
    print("flattened obs.shape:", b_obs.shape)
    print("flattened actions.shape:", b_actions.shape)
