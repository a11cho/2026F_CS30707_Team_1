"""독립 행동을 선택하는 공유 Actor와 관측별 가치를 추정하는 공유 Critic."""

import torch
from torch import nn
from torch.distributions import Categorical


def _mlp(input_size, hidden_size, output_size, output_gain):
    layers = nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.Tanh(),
        nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        nn.Linear(hidden_size, output_size),
    )
    linear = [layer for layer in layers if isinstance(layer, nn.Linear)]
    for layer in linear:
        nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linear[-1].weight, gain=output_gain)
    return layers


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, action_count):
        super().__init__()
        self.policy = _mlp(input_size, hidden_size, action_count, 0.01)

    def forward(self, features):
        return self.policy(features)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.value = _mlp(input_size, hidden_size, 1, 1)

    def forward(self, features):
        return self.value(features).squeeze(-1)


def masked_distribution(logits, mask):
    return Categorical(logits=logits.masked_fill(~mask, float("-inf")))


def sample_actions(actor, features, mask, deterministic=False):
    dist = masked_distribution(actor(features), mask)
    actions = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
    return actions, dist.log_prob(actions)


def make_networks(env, cfg, device):
    hidden = cfg.model["hidden_size"]
    return (Actor(env.input_size, hidden, env.action_count).to(device),
            Critic(env.input_size, hidden).to(device))
