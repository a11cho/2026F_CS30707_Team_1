from dataclasses import dataclass, replace

import numpy as np

from territory_rl.models import AGENT_IDS, Action, AgentSnapshot, StateSnapshot

from .state import JOINT_ACTIONS, JOINT_ATTACKS, REFLECT_ACTION, STEPS, TeamState
from .teacher import TacticalScorer, cooperative_scores


@dataclass(frozen=True)
class SearchBudget:
    candidates: int
    replies: int
    exploration: float
    worst_reply_weight: float
    lookahead_candidates: int = 0


FINE_SEARCH_BUDGETS = {
    f"search_explore_{percent}": SearchBudget(8, 1, percent / 100, 0.0)
    for percent in (67, 64, 61, 58)
}

FINE_TACTICAL_BUDGETS = {
    "search_tactical_50": SearchBudget(8, 1, 0.50, 0.0),
    "search_tactical_45": SearchBudget(10, 2, 0.45, 0.1),
    "search_tactical_40": SearchBudget(10, 2, 0.40, 0.1),
}

TACTICAL_LEAF_BUDGETS = {
    f"search_tactical_leaf_{2 * candidates}": SearchBudget(candidates, 2, 0.50, 0.1)
    for candidates in (4, 6, 8, 10, 12)
}

TACTICAL_EXPLORATION_BUDGETS = {
    f"search_tactical_random_{percent}": replace(TACTICAL_LEAF_BUDGETS["search_tactical_leaf_8"],
                                                exploration=percent / 100)
    for percent in range(95, 50, -5)
}

SEARCH_BUDGETS = {
    "search_opening": SearchBudget(8, 1, 0.90, 0.0),
    "search_intro": SearchBudget(8, 1, 0.70, 0.0),
    "search_intermediate": SearchBudget(8, 1, 0.55, 0.0),
    "search_tactical": SearchBudget(12, 3, 0.35, 0.2),
    "search_tactical_0": SearchBudget(12, 3, 0.0, 0.2),
    "search_strong": SearchBudget(20, 5, 0.0, 0.4),
    "search_lookahead": SearchBudget(12, 3, 0.0, 0.2, lookahead_candidates=4),
    "search_lookahead_wide": SearchBudget(20, 5, 0.0, 0.2, lookahead_candidates=5),
    **FINE_SEARCH_BUDGETS,
    **FINE_TACTICAL_BUDGETS,
    **TACTICAL_LEAF_BUDGETS,
    **TACTICAL_EXPLORATION_BUDGETS,
}
SEARCH_TEACHERS = ("search_strong", "search_lookahead", "search_lookahead_wide")


def opponent_view(state):
    """Reconstruct the opposing view and its masks from the same public full board."""
    order = np.array([3, 4, 5, 0, 1, 2])
    board = np.array([1, 0, 5, 6, 7, 2, 3, 4], dtype=np.int8)[state.board[:, ::-1]]
    positions = state.positions[order].copy()
    alive = positions[:, 0] >= 0
    positions[alive, 1] = board.shape[1] - 1 - positions[alive, 1]
    targets = positions[:3, None] + STEPS[None, :]
    masks = alive[:3, None] & ((targets >= 0) & (targets < board.shape)).all(axis=-1)
    masks[:, Action.ATTACK_UP.index:] &= state.ready[order[:3], None]
    masks[:, Action.NOOP.index] = True
    return TeamState(board, positions, state.ready[order], state.waits[order], masks,
                     state.tick, state.max_ticks, state.areas[::-1])


def public_snapshot(state):
    owner = ((state.board == 1) | (state.board >= 5)).astype(np.int8)
    agents = tuple(AgentSnapshot(agent_id=aid, team=i // 3, position=tuple(map(int, state.positions[i])),
                                 last_attack_tick=state.tick - int(state.ready[i]), respawn_at=None,
                                 last_action=Action.NOOP, action_valid=False, alive=bool(state.positions[i, 0] >= 0))
                   for i, aid in enumerate(AGENT_IDS))
    return StateSnapshot(owner, agents, state.tick, state.max_ticks, state.areas, False)


def shortlist(state, mask, limit, rng):
    candidates = np.flatnonzero(mask)
    if len(candidates) <= limit:
        return candidates
    scores = cooperative_scores(state, candidates) + 0.001 * rng.random(len(candidates))
    ranked = candidates[np.argsort(-scores)]
    # Keep distinct attack plans near the top, then fill with the best remaining moves.
    attack_plans = np.where(JOINT_ATTACKS[ranked], JOINT_ACTIONS[ranked], 0)
    _, first = np.unique(attack_plans, axis=0, return_index=True)
    varied = ranked[np.sort(first)[:(limit + 1) // 2]]
    return np.concatenate((varied, ranked[~np.isin(ranked, varied)][:limit - len(varied)]))


def commands(indices, ids):
    return dict(zip(ids, map(Action.from_index, indices), strict=True))


def search_matrix(state, rng, budget, mask):
    mine = shortlist(state, mask, budget.candidates, rng)
    if budget.replies == 1:
        # The opening opponent assumes a passive reply.
        replies = [Action.NOOP.index]
    else:
        opponent = opponent_view(state)
        replies = shortlist(opponent, opponent.choices.legal, budget.replies, rng)
    enemy_commands = [commands(REFLECT_ACTION[JOINT_ACTIONS[index]], AGENT_IDS[3:]) for index in replies]
    score = TacticalScorer(public_snapshot(state), 0, AGENT_IDS[:3])
    values = np.array([[score(ally | enemy) for enemy in enemy_commands]
                       for ally in (commands(JOINT_ACTIONS[index], AGENT_IDS[:3]) for index in mine)])
    return mine, replies, values


def reply_scores(values, budget):
    return (1 - budget.worst_reply_weight) * values.mean(axis=1) + budget.worst_reply_weight * values.min(axis=1)


def choose_search_action(state, rng, budget, *, force_entry=True, respawn_delay=0):
    """Rank simultaneous joint actions from public state under the chosen budget."""
    mask = state.choices.mask(force_entry)
    allowed = np.flatnonzero(mask)
    if len(allowed) == 1 or rng.random() < budget.exploration:
        return int(rng.choice(allowed))

    if budget.lookahead_candidates:
        from .search_lookahead import choose_lookahead_action

        return choose_lookahead_action(state, rng, budget, force_entry, respawn_delay)
    mine, _, values = search_matrix(state, rng, budget, mask)
    scores = reply_scores(values, budget)
    return int(rng.choice(mine[np.isclose(scores, scores.max(), rtol=0, atol=1e-9)]))
