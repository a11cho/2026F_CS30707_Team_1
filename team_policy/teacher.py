from collections import deque

import numpy as np

from territory_rl.attack import resolve_attacks
from territory_rl.controller import FrontierController, RandomController
from territory_rl.models import Action
from territory_rl.movement import resolve_movements

from .cooperation_scenes import neighbors
from .state import JOINT_ATTACKS, JOINT_MOVES


def frontier_distances(owner, team, adjacent):
    distances = {cell: 0 for cell, edges in adjacent.items()
                 if owner[cell] == team and any(owner[edge] != team for edge in edges)}
    queue = deque(distances)
    while queue:
        cell = queue.popleft()
        for edge in adjacent[cell]:
            if owner[edge] == team and edge not in distances:
                distances[edge] = distances[cell] + 1
                queue.append(edge)
    return distances


def mean_distance(positions, distances, scale):
    return float(np.mean([min(1.0, distances.get(cell, scale) / scale) for cell in positions])) if positions else 1.0


def cooperative_scores(state, candidates, *, defensive=False):
    targets = state.targets[candidates]
    starts = state.positions[:3]
    attacks, moves = JOINT_ATTACKS[candidates], JOINT_MOVES[candidates]
    shared_targets = (targets[:, :, None] == targets[:, None, :]).all(axis=-1)
    attack_counts = (shared_targets & attacks[:, None, :]).sum(axis=2)
    earlier = np.tri(3, 3, k=-1, dtype=bool)
    first_attack = attacks & ~(shared_targets & attacks[:, None, :] & earlier).any(axis=2)

    adjacent = neighbors(*state.board.shape)
    threats = np.zeros_like(state.board)
    for cell in state.positions[3:][state.ready[3:]]:
        for target in adjacent[tuple(cell)]:
            threats[target] += 1
    rows = targets[:, :, 0].clip(0, state.board.shape[0] - 1)
    cols = targets[:, :, 1].clip(0, state.board.shape[1] - 1)
    tiles = state.board[rows, cols]
    enemy = (tiles == 1) | (tiles >= 5)
    captures = enemy & (attack_counts > threats[rows, cols])
    cover = (2 <= tiles) & (tiles <= 4) & (threats[rows, cols] > 0) & (attack_counts >= threats[rows, cols])
    occupied_by = np.where((2 <= tiles) & (tiles <= 4), tiles - 2, -1)
    vacant = (occupied_by < 0) | np.take_along_axis(moves, occupied_by.clip(0), axis=1)

    own_tiles = (state.board == 0) | ((2 <= state.board) & (state.board <= 4))
    distances = np.full(state.board.shape, sum(state.board.shape), dtype=np.int64)
    for cell, distance in frontier_distances(~own_tiles, 0, adjacent).items():
        distances[cell] = distance
    closer = distances[rows, cols] < distances[tuple(starts.T)]
    advance = moves & vacant & ((closer & ~enemy) | captures)
    wasted = moves & ((enemy & ~captures) | ~vacant)
    return ((first_attack & captures).sum(axis=1) + 1.2 * (first_attack & captures & (tiles >= 5)).sum(axis=1)
            + (1.8 if defensive else 0.75) * (first_attack & cover).sum(axis=1)
            + 0.24 * advance.sum(axis=1) - 0.2 * wasted.sum(axis=1)
            + 0.02 * (attacks & enemy).sum(axis=1) - 0.01 * moves.sum(axis=1))


def cooperative_joint(state, rng, *, defensive):
    candidates = np.flatnonzero(state.choices.mask(True))
    score = cooperative_scores(state, candidates, defensive=defensive)
    best = candidates[np.isclose(score, score.max())]
    return int(rng.choice(best))


def teacher_candidates(state, mask, rng, limit):
    allowed = np.flatnonzero(mask)
    if len(allowed) <= limit:
        return allowed
    rows = state.targets[allowed, :, 0].clip(0, state.board.shape[0] - 1)
    columns = state.targets[allowed, :, 1].clip(0, state.board.shape[1] - 1)
    tiles = state.board[rows, columns]
    enemy = (tiles == 1) | (tiles >= 5)
    attacks, moves = JOINT_ATTACKS[allowed], JOINT_MOVES[allowed]
    own = (state.board == 0) | ((2 <= state.board) & (state.board <= 4))
    distances = frontier_distances(~own, 0, neighbors(*state.board.shape))
    distance_board = np.full(state.board.shape, sum(state.board.shape), dtype=np.int64)
    for cell, distance in distances.items():
        distance_board[cell] = distance
    closer = distance_board[rows, columns] < distance_board[tuple(state.positions[:3].T)]
    priority = ((attacks & enemy).sum(axis=1) + 2 * (attacks & (tiles >= 5)).sum(axis=1)
                + (moves & closer).sum(axis=1) - (moves & enemy).sum(axis=1)
                + 4 * state.choices.certified[allowed])
    ranked = np.argsort(priority + 0.001 * rng.random(len(allowed)))
    guided = ranked[-limit // 2:]
    uniform = rng.choice(ranked[:-len(guided)], limit - len(guided), replace=False)
    return allowed[np.concatenate((guided, uniform))]


class RuleTeacher:
    def __init__(self, seed, *, candidates=64, temperature=0.04):
        self.rng = np.random.default_rng(seed)
        self.candidate_limit = candidates
        self.temperature = temperature
        self.replays = 0

    def _opponent_predictions(self, match):
        predictions = []
        for controller in (RandomController, RandomController, FrontierController, FrontierController):
            predictions.append({aid: controller(int(self.rng.integers(2**32))).act(match.observations[aid])
                                for aid in match.enemies})
        return predictions

    def distribution(self, match):
        """Score the same public-state hypotheses for every candidate, before any respawn."""
        state, _, mask = match.view()
        candidates = teacher_candidates(state, mask, self.rng, self.candidate_limit)
        enemies = self._opponent_predictions(match)
        scorer = TacticalScorer(match.env.core.snapshot(), match.team_number, match.allies)
        scores = []
        for index in candidates:
            ally_actions = world_actions(match, index)
            net = sum(scorer(ally_actions | enemy_actions) for enemy_actions in enemies)
            scores.append(net / len(enemies))
        self.replays += len(candidates) * len(enemies)
        scores = np.array(scores)
        weights = np.exp((scores - scores.max()) / self.temperature)
        target = np.zeros_like(mask, dtype=np.float32)
        target[candidates] = weights / weights.sum()
        return target


def world_actions(match, index):
    return {aid: Action.from_index(int(action)) for aid, action in zip(match.allies, match.world_indices(index), strict=True)}


class TacticalScorer:
    """Rank legal joint decisions on one snapshot with the engine's attack and movement rules."""

    def __init__(self, snapshot, team, allies):
        self.snapshot, self.team, self.allies = snapshot, team, allies
        self.agents = {agent.agent_id: agent for agent in snapshot.agents if agent.alive}
        self.starts = {aid: agent.position for aid, agent in self.agents.items()}
        self.edges = neighbors(*snapshot.owner.shape)
        self.scale = sum(snapshot.owner.shape)
        distances = frontier_distances(snapshot.owner, team, self.edges)
        self.before_distance = mean_distance([self.starts[aid] for aid in allies if aid in self.starts],
                                             distances, self.scale)
        self.distance_cache = {snapshot.owner.tobytes(): distances}

    def __call__(self, actions):
        actions = {aid: action for aid, action in actions.items() if aid in self.agents}
        next_board = self.snapshot.owner.copy()
        for event in resolve_attacks(self.snapshot.owner, self.agents, actions):
            if event.kind == "capture":
                next_board[event.position] = event.new_owner
        survivors = {aid: cell for aid, cell in self.starts.items() if next_board[cell] == self.agents[aid].team}
        destinations, _ = resolve_movements(survivors, actions, self.edges.__contains__)
        alive = {aid: cell for aid, cell in destinations.items() if next_board[cell] == self.agents[aid].team}
        death_balance = sum(1 if self.agents[aid].team != self.team else -1 for aid in self.starts.keys() - alive.keys())
        own_tiles = np.count_nonzero(next_board == self.team)
        gap_change = 2 * (own_tiles - self.snapshot.areas[self.team]) / next_board.size
        key = next_board.tobytes()
        if key not in self.distance_cache:
            self.distance_cache[key] = frontier_distances(next_board, self.team, self.edges)
        ally_positions = [alive[aid] for aid in self.allies if aid in alive]
        distance = mean_distance(ally_positions, self.distance_cache[key], self.scale)
        return gap_change + 0.03 * death_balance + 0.01 * (self.before_distance - distance)
