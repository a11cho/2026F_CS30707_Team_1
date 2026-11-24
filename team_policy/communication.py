from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from .policy import TeamPolicy
from .slot_encoder import SLOTS, SlotQwen
from .state import ACTION_COUNT, CANVAS, JOINT_ACTIONS, JOINT_ATTACKS, JOINT_MOVES, STEPS


class Intent(IntEnum):
    SILENT = 0
    ENTER = 1
    CAPTURE = 2
    COVER = 3
    FOCUS = 4


DIRECTIONS = STEPS[1:5]
INTENT_COUNT = 1 + 4 * (len(Intent) - 1)
CELL_COUNT = CANVAS[0] * CANVAS[1]


def intent_candidates(state):
    neighbors = state.positions[:SLOTS, None] + DIRECTIONS
    rows = neighbors[:, :, 0].clip(0, state.board.shape[0] - 1)
    cols = neighbors[:, :, 1].clip(0, state.board.shape[1] - 1)
    tiles = state.board[rows, cols]
    hostile = (tiles == 1) | (tiles >= 5)
    attack = state.action_masks[:, 5:9]
    mask = np.concatenate((np.ones((SLOTS, 1), dtype=bool), state.action_masks[:, 1:5],
                           attack & hostile, attack & ((tiles >= 2) & (tiles <= 4)), attack & hostile), axis=1)
    cells = rows * CANVAS[1] + cols
    packets = np.concatenate((np.zeros((SLOTS, 1), dtype=np.int64),
                              *(1 + (kind - 1) * CELL_COUNT + cells for kind in tuple(Intent)[1:])), axis=1)
    return mask, packets


def teacher_intents(state, joint):
    actions = JOINT_ACTIONS[joint]
    targets = state.targets[joint]
    same_tile = (targets[:, None] == targets[None, :]).all(axis=-1)
    attacks = (same_tile & JOINT_ATTACKS[joint]).sum(axis=1)
    output = np.zeros(SLOTS, dtype=np.int64)
    for slot in np.flatnonzero(JOINT_MOVES[joint] | JOINT_ATTACKS[joint]):
        row, col = targets[slot]
        code = state.board[row, col]
        hostile = code == 1 or code >= 5
        direction = (actions[slot] - 1) % 4
        if JOINT_MOVES[joint, slot] and hostile and attacks[slot]:
            kind = Intent.ENTER
        elif JOINT_ATTACKS[joint, slot] and hostile and attacks[slot] >= 2:
            kind = Intent.FOCUS
        elif JOINT_ATTACKS[joint, slot] and hostile and (same_tile[slot] & JOINT_MOVES[joint]).any():
            kind = Intent.CAPTURE
        elif JOINT_ATTACKS[joint, slot] and 2 <= code <= 4:
            enemy_distance = np.abs(state.positions[SLOTS:] - targets[slot]).sum(axis=1)
            if ((enemy_distance == 1) & state.ready[SLOTS:] & (state.positions[SLOTS:, 0] >= 0)).any():
                kind = Intent.COVER
            else:
                continue
        else:
            continue
        output[slot] = 1 + 4 * (kind - 1) + direction
    return output


@dataclass(frozen=True)
class SlotInput:
    board: np.ndarray
    status: np.ndarray
    joint_mask: np.ndarray
    intent_mask: np.ndarray
    packets: np.ndarray
    first_slot: int
    tokens: list[int] | None


@dataclass
class SlotDecision:
    intent: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    log_probability: torch.Tensor
    values: torch.Tensor
    prefixes: list[torch.Tensor] | None

    @property
    def joint(self):
        return (self.actions[:, 0] * ACTION_COUNT + self.actions[:, 1]) * ACTION_COUNT + self.actions[:, 2]


def masked(logits, mask):
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def draw(distribution, generator):
    if generator is None:
        return distribution.logits.argmax(dim=-1)
    flat = distribution.probs.reshape(-1, distribution.probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(distribution.batch_shape)


class AgentHeads(nn.Module):
    def __init__(self, source, slot, features):
        super().__init__()
        self.text_projection = deepcopy(source.text_projection) if source.text_encoder is not None else None
        self.intent = nn.Linear(features, INTENT_COUNT)
        self.action = nn.Linear(features, ACTION_COUNT)
        message_width = 16
        self.intent_embedding = nn.Embedding(len(Intent), message_width, padding_idx=0)
        self.target_embedding = nn.Embedding(CELL_COUNT + 1, message_width, padding_idx=0)
        self.response = nn.Sequential(nn.Linear(features + SLOTS * message_width, 64), nn.Tanh(),
                                      nn.Linear(64, ACTION_COUNT))
        block = slice(slot * ACTION_COUNT, (slot + 1) * ACTION_COUNT)
        with torch.no_grad():
            self.action.weight.copy_(source.action_head.unary.weight[block])
            self.action.bias.copy_(source.action_head.unary.bias[block])
            nn.init.zeros_(self.response[-1].weight)
            nn.init.zeros_(self.response[-1].bias)

    def action_scores(self, features, packets):
        offset = (packets - 1).clamp_min(0)
        types = torch.where(packets > 0, offset // CELL_COUNT + 1, 0)
        targets = torch.where(packets > 0, offset % CELL_COUNT + 1, 0)
        messages = self.intent_embedding(types) + self.target_embedding(targets)
        return self.action(features) + self.response(torch.cat((features, messages.flatten(1)), dim=1))


class SlotTeamPolicy(nn.Module):
    def __init__(self, backbone, device, *, base=None, initialize=None, communication=True):
        super().__init__()
        source = TeamPolicy(backbone, device, base=base, rules_prompt=True, rules_prompt_version=4)
        if initialize is not None:
            source.load_trainable_weights(initialize)
        self.device = source.device
        self.backbone = backbone
        self.additive_only = False
        self.communication = communication
        self.mute_received = False
        self.tokenizer = source.tokenizer
        self.spatial = source.spatial
        self.spatial_projection = source.spatial_projection
        self.value = nn.Linear(128, 1)
        with torch.no_grad():
            self.value.weight.copy_(source.value_head.weight[:, -128:])
            self.value.bias.copy_(source.value_head.bias)
        self.agents = nn.ModuleList(AgentHeads(source, slot, source.action_head.unary.in_features)
                                    for slot in range(SLOTS))
        self.text_encoder = SlotQwen(source.text_encoder) if source.text_encoder is not None else None
        self.register_buffer("joint_actions", torch.from_numpy(JOINT_ACTIONS), persistent=False)
        self.to(self.device)

    def prepare(self, views):
        if self.tokenizer is None:
            tokens = [None] * len(views)
        else:
            texts = []
            for state, history, _ in views:
                prompt = state.text_input(history, rules_prompt=True, rules_prompt_version=4)
                prompt = prompt.replace("Choose one joint action for a,b,c.", "Each ally selects its own action.", 1)
                texts.append(prompt.removesuffix("Team action:") + "Ally action:")
            tokens = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [SlotInput(*state.spatial_input(), mask, *intent_candidates(state), state.tick % SLOTS, ids)
                for (state, _, mask), ids in zip(views, tokens, strict=True)]

    def spatial_features(self, samples):
        boards = torch.from_numpy(np.stack([item.board for item in samples])).to(self.device)
        status = torch.from_numpy(np.stack([item.status for item in samples])).to(self.device)
        return self.spatial_projection(torch.cat((self.spatial(boards), status), dim=1))

    def hidden(self, samples, prefixes=None, *, cache=False):
        if self.text_encoder is None:
            return None, None
        if prefixes is None:
            inputs = self.tokenizer.pad({"input_ids": [item.tokens for item in samples]},
                                        return_tensors="pt").to(self.device)
            attention = inputs.attention_mask
            prefix = self.text_encoder.prefix(inputs.input_ids, attention)
        else:
            prefix = pad_sequence(prefixes, batch_first=True).to(self.device)
            lengths = torch.tensor([len(x) for x in prefixes], device=self.device)
            attention = (torch.arange(prefix.shape[1], device=self.device) < lengths[:, None]).long()
        cached = None
        if cache:
            cpu_prefix = prefix.cpu()
            cached = [row[:len(item.tokens)] for row, item in zip(cpu_prefix, samples, strict=True)]
        return self.text_encoder(prefix, attention), cached

    def features(self, spatial, hidden):
        if hidden is None:
            return [spatial] * SLOTS
        return [torch.cat((agent.text_projection(hidden[:, slot]), spatial), dim=1)
                for slot, agent in enumerate(self.agents)]

    def intent_distribution(self, samples, features):
        scores = torch.stack([agent.intent(features[slot]) for slot, agent in enumerate(self.agents)], dim=1)
        mask = torch.from_numpy(np.stack([sample.intent_mask for sample in samples])).to(self.device)
        if not self.communication:
            mask[:, :, 1:] = False
        return masked(scores, mask)

    def scores(self, samples, features, intents):
        table = torch.from_numpy(np.stack([sample.packets for sample in samples])).to(self.device)
        packets = table.gather(2, intents.unsqueeze(2)).squeeze(2)
        return torch.stack([agent.action_scores(features[slot], self.inbox(packets, slot))
                            for slot, agent in enumerate(self.agents)], dim=1)

    def inbox(self, packets, slot):
        if self.mute_received:
            return packets * (torch.arange(SLOTS, device=self.device) == slot)
        return packets

    def commit(self, samples, scores=None, generator=None, *, prescribed=None):
        remaining = torch.from_numpy(np.stack([sample.joint_mask for sample in samples])).to(self.device)
        first = torch.tensor([sample.first_slot for sample in samples], device=self.device)
        rows = torch.arange(len(samples), device=self.device)
        masks = torch.empty((len(samples), SLOTS, ACTION_COUNT), dtype=torch.bool, device=self.device)
        actions = torch.empty((len(samples), SLOTS), dtype=torch.long, device=self.device)
        choices = torch.arange(ACTION_COUNT, device=self.device)
        for offset in range(SLOTS):
            slots = (first + offset) % SLOTS
            options = self.joint_actions[:, slots].T
            valid = (remaining[:, :, None] & (options[:, :, None] == choices)).any(dim=1)
            chosen = prescribed[rows, slots] if prescribed is not None else draw(masked(scores[rows, slots], valid), generator)
            masks[rows, slots] = valid
            actions[rows, slots] = chosen
            remaining &= options == chosen[:, None]
        if prescribed is not None and not remaining.any(dim=1).all():
            raise ValueError("teacher action is outside the joint policy mask")
        return actions, masks

    def decide(self, samples, *, message_rng=None, action_rng=None, cache=False):
        spatial = self.spatial_features(samples)
        hidden, prefixes = self.hidden(samples, cache=cache)
        features = self.features(spatial, hidden)
        intentions = self.intent_distribution(samples, features)
        messages = draw(intentions, message_rng)
        scores = self.scores(samples, features, messages)
        actions, masks = self.commit(samples, scores, action_rng)
        log_probability = (intentions.log_prob(messages) + masked(scores, masks).log_prob(actions)).sum(dim=1)
        return SlotDecision(messages, actions, masks, log_probability, self.value(spatial).squeeze(1), prefixes)

    def distributions(self, samples, intents, masks, *, prefixes=None, hidden=None):
        spatial = self.spatial_features(samples)
        if hidden is None:
            hidden, _ = self.hidden(samples, prefixes)
        features = self.features(spatial, hidden)
        return (self.intent_distribution(samples, features), masked(self.scores(samples, features, intents), masks),
                self.value(spatial).squeeze(1))

    def trainable_weights(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters() if value.requires_grad}

    @torch.no_grad()
    def load_trainable_weights(self, weights):
        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if parameters.keys() != weights.keys():
            raise ValueError("checkpoint parameters do not match the slot policy")
        for name, value in parameters.items():
            value.copy_(weights[name])
