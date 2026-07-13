"""
ppo.py

The core PPO algorithm: given a rollout (already flattened + GAE computed),
runs N_EPOCHS of minibatch gradient updates using the clipped surrogate
objective.

This file has NO knowledge of the environment or rollout collection - it
only takes tensors in and returns loss statistics out. Keeping it isolated
like this makes the algorithm itself easy to test/reason about separately
from env plumbing.
"""

import numpy as np
import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    ENT_COEF,
    LR,
    MAX_GRAD_NORM,
    N_EPOCHS,
    VF_COEF,
)


class PPOAgent:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def set_lr(self, lr: float):
        """
        Updates the optimizer's learning rate in place. Used for linear LR
        annealing over training (helps the policy settle into a stable
        optimum in the later stages instead of continuing to take large,
        potentially destabilizing steps at a constant high LR).
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values):
        """
        Runs N_EPOCHS of minibatch updates over the flattened rollout data.

        All b_* arguments are numpy arrays of shape (batch_size_total, ...)
        coming from RolloutBuffer.get_flat().

        Returns a dict of diagnostic statistics (useful for logging /
        detecting training pathologies - e.g. exploding clipfrac or
        collapsing entropy).
        """
        # normalize advantages (per full rollout, standard PPO trick -
        # reduces variance of the policy gradient estimate)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=self.device)
        b_actions_t = torch.as_tensor(b_actions, dtype=torch.int64, device=self.device)
        b_log_probs_t = torch.as_tensor(b_log_probs, dtype=torch.float32, device=self.device)
        b_advantages_t = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)
        b_returns_t = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)

        batch_size_total = b_obs.shape[0]
        indices = np.arange(batch_size_total)

        pg_losses, v_losses, entropy_losses, clipfracs, approx_kls = [], [], [], [], []

        for epoch in range(N_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, batch_size_total, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = indices[start:end]

                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                    b_obs_t[mb_idx], b_actions_t[mb_idx]
                )

                log_ratio = new_log_prob - b_log_probs_t[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # approx_kl: cheap approximation of KL(old || new),
                    # useful for spotting a policy update that moved too far
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()

                mb_advantages = b_advantages_t[mb_idx]

                # clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (plain MSE - no value clipping, kept simple)
                v_loss = 0.5 * ((new_value - b_returns_t[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                clipfracs.append(clipfrac)
                approx_kls.append(approx_kl)

        explained_var = _explained_variance(b_values, b_returns)

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "clipfrac": float(np.mean(clipfracs)),
            "approx_kl": float(np.mean(approx_kls)),
            "explained_variance": explained_var,
        }


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """
    How much of the return variance is explained by the value function.
    1.0 = perfect predictions, 0.0 = no better than predicting the mean,
    negative = worse than predicting the mean (a red flag for the critic).
    """
    var_returns = np.var(returns)
    if var_returns == 0:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_returns)


if __name__ == "__main__":
    # quick sanity check: one update step on random data should not crash
    # and should produce finite losses
    from networks import ActorCritic

    device = torch.device("cpu")
    model = ActorCritic(obs_dim=75, n_actions=5).to(device)
    agent = PPOAgent(model, device)

    batch_size_total = 128
    b_obs = np.random.randn(batch_size_total, 75).astype(np.float32)
    b_actions = np.random.randint(0, 5, size=batch_size_total)
    b_log_probs = np.random.randn(batch_size_total).astype(np.float32)
    b_advantages = np.random.randn(batch_size_total).astype(np.float32)
    b_returns = np.random.randn(batch_size_total).astype(np.float32)
    b_values = np.random.randn(batch_size_total).astype(np.float32)

    stats = agent.update(b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values)
    print(stats)
