"""
networks.py

ActorCritic network for discrete-action PPO on HighwayEnv.

Design choice: policy_net and value_net are fully independent MLPs (no
shared trunk). This avoids gradient interference between the policy and
value losses, which tends to make PPO training more stable at the cost of
a few more parameters.

Input: flattened Kinematics observation, shape (15, 5) -> obs_dim = 75
Output: 5 discrete actions (DiscreteMetaAction)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import NET_ARCH


def build_mlp(input_dim: int, output_dim: int, hidden_sizes: list[int]) -> nn.Sequential:
    """
    Builds a simple feedforward MLP:
        Linear -> Tanh -> Linear -> Tanh -> ... -> Linear (no activation on output)

    Tanh is used (rather than ReLU) because it is the standard choice for
    PPO policy/value networks - it keeps activations bounded, which helps
    with the stability of the clipped objective.
    """
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes if hidden_sizes is not None else NET_ARCH

        # independent networks: no shared parameters between actor and critic
        self.policy_net = build_mlp(obs_dim, n_actions, hidden_sizes)
        self.value_net = build_mlp(obs_dim, 1, hidden_sizes)

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization (standard for PPO, used by the original
        PPO paper and SB3's default init). The policy's final layer gets a
        small gain so the initial action distribution starts close to
        uniform rather than saturated.
        """
        for module in self.policy_net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.4142135623730951)  # sqrt(2)
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_net[-1].bias, 0.0)

        for module in self.value_net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.4142135623730951)
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)
        nn.init.constant_(self.value_net[-1].bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch_size, obs_dim) - already flattened.

        Returns:
            logits: (batch_size, n_actions) - raw policy scores
            value:  (batch_size,)           - V(s) estimate
        """
        logits = self.policy_net(obs)
        value = self.value_net(obs).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        obs: (batch_size, obs_dim)
        action: optional (batch_size,) - if provided, evaluate log_prob/entropy
                for this given action instead of sampling a new one (used
                during the PPO update, where we re-evaluate actions taken
                during rollout collection).

        Returns:
            action:    (batch_size,)
            log_prob:  (batch_size,)
            entropy:   (batch_size,)
            value:     (batch_size,)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


def flatten_obs(obs: torch.Tensor) -> torch.Tensor:
    """
    Flattens a batch of (15, 5) Kinematics observations into (batch, 75).
    Expects obs of shape (batch_size, 15, 5).
    """
    return obs.reshape(obs.shape[0], -1)


if __name__ == "__main__":
    # quick sanity check with dummy data matching HighwayEnv's obs shape
    obs_dim = 15 * 5
    n_actions = 5

    model = ActorCritic(obs_dim, n_actions)

    dummy_obs = torch.randn(4, 15, 5)  # batch of 4
    dummy_obs_flat = flatten_obs(dummy_obs)

    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs_flat)

    print("action:", action.shape, action)
    print("log_prob:", log_prob.shape, log_prob)
    print("entropy:", entropy.shape, entropy)
    print("value:", value.shape, value)
