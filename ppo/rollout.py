from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.controller import FrontierController, RandomController
from territory_rl.models import Action
from territory_rl.parallel_env import TerritoryParallelEnv

from .policy import StateBatch, action_masks, encode_proposals, encode_states


OPPONENTS = ("random", "expand", "aggressive", "defend")


class FeedbackMatch:
    def __init__(self, game: GameConfig, experiment: ExperimentConfig):
        self.env = TerritoryParallelEnv(game, experiment)

    def reset(self, seed, team, opponent, *, options=None):
        self.team = team
        self.observations, _ = self.env.reset(seed=seed, options=options)
        self.learners = [agent for agent in self.env.agents if agent.startswith(team + "_")]
        bots = [agent for agent in self.env.agents if agent not in self.learners]
        bot_seeds = np.random.SeedSequence(seed, spawn_key=(1,)).generate_state(len(bots))
        self.bots = {
            agent: RandomController(int(bot_seed)) if opponent == "random"
            else FrontierController(int(bot_seed), style=opponent)
            for agent, bot_seed in zip(bots, bot_seeds)
        }

    def _session(self, agent):
        team, number = agent.rsplit("_", 1)
        return self.env.agent_session(team, int(number))

    def _submit(self, learner_actions):
        for agent, index in zip(self.learners, learner_actions, strict=True):
            self._session(agent).submit_action(Action.from_index(int(index)))
        for agent, action in self.bot_actions.items():
            self._session(agent).submit_action(action)

    def begin(self, proposals):
        self.env.begin_decision()
        # Fixed opponents make one decision per physical tick in every condition.
        self.bot_actions = {agent: bot.act(self.observations[agent]) for agent, bot in self.bots.items()}
        self._submit(proposals)
        if self.env.finish_decision_round() is not None:
            raise RuntimeError("the proposal round must not advance the environment")
        return [self._session(agent).observe() for agent in self.learners]

    def finish(self, actions):
        self._submit(actions)
        self.observations, rewards, terminated, truncated, _ = self.env.finish_decision_round()
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
