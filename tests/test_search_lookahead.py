import numpy as np
import pytest

from team_policy.match import TeamMatch
from team_policy.search import SEARCH_BUDGETS, choose_search_action, opponent_view
from team_policy.search_lookahead import terminal_payoffs, terminal_plan
from team_policy.state import JOINT_ACTIONS, REFLECT_ACTION
from territory_rl.agent import AgentState
from territory_rl.config import GameConfig
from territory_rl.models import AGENT_IDS


def _scenario():
    board = np.broadcast_to(np.arange(6) >= 3, (6, 6)).astype(np.int8)
    positions = [(2, 2), (0, 1), (4, 1), (2, 3), (0, 4), (4, 4)]
    agents = [
        AgentState(aid, i // 3, cell, alive=i % 3 == 0, respawn_at=None if i % 3 == 0 else 3)
        for i, (aid, cell) in enumerate(zip(AGENT_IDS, positions, strict=True))
    ]
    return {"owner": board, "agents": agents}


@pytest.mark.parametrize("team", ("red", "blue"))
def test_last_tick_payoffs_agree_with_engine_for_every_opponent_plan(team):
    game = GameConfig(height=6, width=6, max_ticks=1)
    scenario = _scenario()
    match = TeamMatch(game, 7, team, "noop", scenario=scenario, force_entry=False)
    actions, values, gaps = terminal_payoffs(match.state, False)
    replies = np.flatnonzero(opponent_view(match.state).choices.legal)
    for action, expected in zip(actions, gaps, strict=True):
        observed = set()
        for reply in replies:
            replay = TeamMatch(game, 7, team, "noop", scenario=scenario, force_entry=False)
            indices = JOINT_ACTIONS[reply]
            if team == "red":
                indices = REFLECT_ACTION[indices]
            replay.step(action, opponent_actions=dict(zip(replay.enemies, map(int, indices), strict=True)))
            assert replay.done
            observed.add(int(replay.state.areas[0] - replay.state.areas[1]))
        assert observed == set(expected)
    np.testing.assert_array_equal(values, np.sign(gaps))
    _, probabilities, lower, upper = terminal_plan(match.state, False)
    assert lower == upper == 0.0
    assert probabilities @ values.min(axis=1) == lower


def test_two_tick_teacher_returns_a_legal_action_with_pending_respawns():
    game = GameConfig(height=6, width=6, max_ticks=2, respawn_delay=2)
    match = TeamMatch(game, 7, "blue", "search_tactical_0", scenario=_scenario())
    budget = SEARCH_BUDGETS["search_lookahead"]
    state, _, mask = match.view()
    action = choose_search_action(state, np.random.default_rng(1), budget, respawn_delay=game.respawn_delay)
    assert mask[action]
    assert action == choose_search_action(state, np.random.default_rng(1), budget, respawn_delay=game.respawn_delay)
    assert match.state.tick == 0
