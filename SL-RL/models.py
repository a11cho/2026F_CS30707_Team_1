"""소유권 인코더와 개체 수에 독립적인 팀 정책입니다."""
import torch
from torch import nn
from torch.distributions import Categorical
from territory_rl.models import Action


class MapEncoder(nn.Module):
    def __init__(self, cells, hidden, latent):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2*cells, hidden), nn.ReLU(), nn.Linear(hidden, latent))

    def forward(self, board, mask=None):
        if mask is None:
            mask = torch.zeros_like(board, dtype=torch.bool)
        return self.layers(torch.cat([board.masked_fill(mask, 0), mask.float()], dim=-1))


class TeamPolicy(nn.Module):
    def __init__(self, latent, config):
        super().__init__()
        e, h = config["entity_hidden"], config["hidden"]
        self.entity = nn.Sequential(nn.Linear(6, e), nn.Tanh())
        width = latent + 6 + e + 1
        self.actor = nn.Sequential(nn.Linear(width, h), nn.Tanh(), nn.Linear(h, len(Action)))
        self.critic = nn.Sequential(nn.Linear(width, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, latent, own, entities, mask):
        # 1vs1에서도 값이 있는 전체 개체 수 하나를 사용합니다.
        counts = own.new_full((len(own), 1), entities.shape[1])
        features = torch.cat([latent, own, self.entity(entities).mean(1), counts], dim=1)
        logits = self.actor(features).masked_fill(~mask, torch.finfo(latent.dtype).min)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)
