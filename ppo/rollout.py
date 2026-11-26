from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.controller import FrontierController, RandomController
from territory_rl.models import Action
from territory_rl.observation import rotate_action_mask
from territory_rl.parallel_env import TerritoryParallelEnv

from .policy import StateBatch, action_masks, encode_proposals, encode_states


OPPONENTS = ("random", "expand", "aggressive", "defend")


class FeedbackMatch:
    def __init__(self, game: GameConfig, experiment: ExperimentConfig):
        window_size = max(experiment.observation_window_size, 2 * max(game.height, game.width) - 1)
        experiment = replace(experiment, observation_window_size=window_size)
        self.env = TerritoryParallelEnv(game, experiment)

    def reset(self, seed, team, opponent, *, options=None):
        self.team = team
        observations, _ = self.env.reset(seed=seed, options=options)
        self.learners = [agent for agent in self.env.agents if agent.startswith(team + "_")]
        bots = [agent for agent in self.env.agents if agent not in self.learners]
        bot_seeds = np.random.SeedSequence(seed, spawn_key=(1,)).generate_state(len(bots))
        self.bots = {
            agent: RandomController(int(bot_seed)) if opponent == "random"
            else FrontierController(int(bot_seed), style=opponent, tile_encoding=self.env.core.experiment.tile_encoding)
            for agent, bot_seed in zip(bots, bot_seeds)
        }
        self._record_observations(observations)

    def _record_observations(self, observations):
        self._bot_observations = {agent: observations[agent] for agent in self.bots}
        encoding = self.env.core.experiment.tile_encoding
        rotation = self.env.core.observation_rotation
        window = np.rot90(observations[self.learners[0]]["tilemap"], rotation)
        rows, columns = np.nonzero(window != encoding.outside)
        board = window[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        self._owner = np.where(np.isin(board, encoding.friendly_tiles), 1, -1).astype(np.int8)
        self._positions = np.full((6, 2), -1, dtype=np.int32)
        for slot, tile in enumerate((*encoding.friendly_agents, *encoding.enemy_agents)):
            position = np.argwhere(board == tile)
            if len(position):
                self._positions[slot] = position[0]
        self.observations = {agent: self._policy_view(observations[agent]) for agent in self.learners}

    def _policy_view(self, observation):
        # All learners see the same board in map coordinates. The frozen proposal
        # round reuses that board and refreshes the per-slot action information.
        auxiliary = observation["auxiliary"].copy()
        rotation = self.env.core.observation_rotation
        if rotation:
            auxiliary["action_mask"] = rotate_action_mask(auxiliary["action_mask"], -rotation)
            auxiliary["team_action_features"] = np.stack([
                rotate_action_mask(proposal, -rotation) for proposal in auxiliary["team_action_features"]
            ])
        return {**auxiliary, "owner": self._owner, "agent_positions": self._positions}

    def _session(self, agent):
        team, number = agent.rsplit("_", 1)
        return self.env.agent_session(team, int(number))

    def _submit(self, learner_actions):
        for agent, index in zip(self.learners, learner_actions, strict=True):
            action = self.env.core.action_to_observation(Action.from_index(int(index)))
            self._session(agent).submit_action(action)
        for agent, action in self.bot_actions.items():
            self._session(agent).submit_action(action)

    def begin(self, proposals):
        self.env.begin_decision()
        # Fixed opponents make one decision per physical tick in every condition.
        self.bot_actions = {agent: bot.act(self._bot_observations[agent]) for agent, bot in self.bots.items()}
        self._submit(proposals)
        if self.env.finish_decision_round() is not None:
            raise RuntimeError("the proposal round must not advance the environment")
        return [self._policy_view(self._session(agent).observe()) for agent in self.learners]

    def finish(self, actions):
        self._submit(actions)
        observations, rewards, terminated, truncated, _ = self.env.finish_decision_round()
        self._record_observations(observations)
        if any(truncated.values()):
            raise RuntimeError("the v1 collector expects external cuts at rollout boundaries")
        return [rewards[agent] for agent in self.learners], all(terminated.values())


@dataclass
class TickBatch:
    state: StateBatch
    proposals: Tensor
    known: Tensor
    proposal_mask: Tensor
    action_mask: Tensor
    proposal: Tensor
    action: Tensor
    log_probability: Tensor
    value: Tensor
    reward: Tensor
    terminated: Tensor
    listening: Tensor | None


def current_states(matches):
    return [match.observations[agent] for match in matches for agent in match.learners]


@torch.no_grad()
def collect_tick(policy, matches, *, hide_teammates=False, greedy=False, measure_listening=False):
    first = current_states(matches)
    state = encode_states(first)
    encoded = policy.encode(state)
    proposal_mask = action_masks(first)
    proposal_distribution = policy.proposal(encoded, proposal_mask)
    proposal = proposal_distribution.probs.argmax(dim=-1) if greedy else proposal_distribution.sample()

    second = []
    for match, indices in zip(matches, proposal.reshape(-1, 3).tolist(), strict=True):
        second.extend(match.begin(indices))
    proposals, known = encode_proposals(second, hide_teammates)
    action_mask = action_masks(second)
    action_distribution = policy.action(encoded, proposals, known, action_mask)
    action = action_distribution.probs.argmax(dim=-1) if greedy else action_distribution.sample()

    results = [
        match.finish(indices)
        for match, indices in zip(matches, action.reshape(-1, 3).tolist(), strict=True)
    ]
    rewards, terminated = zip(*results)
    listening = None
    if measure_listening:
        hidden, hidden_known = encode_proposals(second, hide_teammates=True)
        without_peers = policy.action(encoded, hidden, hidden_known, action_mask)
        listening = 0.5 * (action_distribution.probs - without_peers.probs).abs().sum(dim=-1)
    return TickBatch(
        state, proposals, known, proposal_mask, action_mask, proposal, action,
        proposal_distribution.log_prob(proposal) + action_distribution.log_prob(action),
        policy.value(encoded), torch.tensor(rewards).flatten(), torch.tensor(terminated), listening,
    )


def generalized_advantage(rewards, values, terminated, bootstrap, gamma, gae_lambda):
    advantages = torch.empty_like(values)
    future_advantage = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for tick in reversed(range(len(rewards))):
        continuation = ~terminated[tick]
        residual = rewards[tick] + gamma * next_value * continuation - values[tick]
        future_advantage = residual + gamma * gae_lambda * future_advantage * continuation
        advantages[tick] = future_advantage
        next_value = values[tick]
    return advantages
