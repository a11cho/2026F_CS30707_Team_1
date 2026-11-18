from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


class StateBatch(NamedTuple):
    board: Tensor
    features: Tensor


def encode_states(observations):
    """Encode the public v1 observation in the requesting team's slot order."""
    owner = np.stack([obs["owner"] for obs in observations])
    batch_size, height, width = owner.shape
    board = np.zeros((batch_size, 8, height, width), dtype=np.float32)
    board[:, 0] = owner == 1
    board[:, 1] = owner == -1
    alive = np.stack([obs["agent_alive"] for obs in observations])
    ready = np.stack([obs["agent_attack_ready"] for obs in observations])
    wait = np.stack([obs["agent_respawn_wait"] for obs in observations]).astype(np.float32)
    positions = np.stack([obs["agent_positions"] for obs in observations])
    samples, slots = np.nonzero(alive)
    rows, columns = positions[samples, slots].T
    board[samples, slots + 2, rows, columns] = 1.0
    identity = np.eye(3, dtype=np.float32)[[obs["self_index"] for obs in observations]]
    time = np.array([obs["remaining_ticks"] / obs["max_ticks"] for obs in observations])
    features = np.column_stack((alive, ready, wait / (1 + wait), identity, time)).astype(np.float32)
    return StateBatch(torch.from_numpy(board), torch.from_numpy(features))


def encode_proposals(observations, hide_teammates=False):
    """Return slots and their visibility mask, with the self slot kept in the ablation."""
    features = torch.from_numpy(np.stack([obs["team_action_features"] for obs in observations]))
    known = np.stack([obs["team_actions_known"] for obs in observations]).astype(bool)
    if hide_teammates:
        own_slots = np.eye(3, dtype=bool)[[obs["self_index"] for obs in observations]]
        known &= own_slots
    known = torch.from_numpy(known)
    return features.masked_fill(~known.unsqueeze(-1), 0), known


def action_masks(observations):
    return torch.from_numpy(np.stack([obs["action_mask"] for obs in observations])).bool()


class CommunicationPolicy(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.canvas_shape = (height, width)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(16 * height * width + 22, 128), nn.Tanh())
        self.proposal_head = nn.Linear(128, 9)
        self.message_encoder = nn.Sequential(nn.Linear(9, 16), nn.Tanh())
        self.action_head = nn.Sequential(nn.Linear(128 + 3 * 16, 128), nn.Tanh(), nn.Linear(128, 9))
        self.value_head = nn.Linear(128, 1)
        for head in (self.proposal_head, self.action_head[-1]):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def encode(self, state):
        board = state.board
        height, width = board.shape[-2:]
        extra_rows = self.canvas_shape[0] - height
        extra_columns = self.canvas_shape[1] - width
        if extra_rows < 0 or extra_columns < 0:
            raise ValueError(f"map {(height, width)} exceeds the policy canvas {self.canvas_shape}")
        if extra_rows or extra_columns:
            # Both ownership channels are zero outside the physical board.
            top, left = extra_rows // 2, extra_columns // 2
            board = F.pad(board, (left, extra_columns - left, top, extra_rows - top))
        return self.state_encoder(torch.cat((self.board_encoder(board), state.features), dim=-1))

    def proposal(self, state, mask):
        return Categorical(logits=self.proposal_head(state).masked_fill(~mask, -torch.inf))

    def action(self, state, proposals, known, mask):
        messages = self.message_encoder(proposals) * known.unsqueeze(-1)
        logits = self.action_head(torch.cat((state, messages.flatten(start_dim=-2)), dim=-1))
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def value(self, state):
        return self.value_head(state).squeeze(-1)


def policy_from_checkpoint(checkpoint):
    game = checkpoint["game"]
    canvas = checkpoint.get("canvas_shape", (game["height"], game["width"]))
    policy = CommunicationPolicy(*canvas)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval()
