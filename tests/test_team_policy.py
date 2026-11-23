from itertools import product

import numpy as np
import pytest

from territory_rl.agent import AgentState
from territory_rl.config import GameConfig
from territory_rl.models import AGENT_IDS, Action

from team_policy.match import DISCOUNT, TeamMatch, entry_scenario
from team_policy.state import JOINT_ACTIONS, REFLECT_ACTION, joint_index


def indices(*actions):
    return joint_index(tuple(action.index for action in actions))


def test_joint_indices_are_bijective():
    assert [joint_index(actions) for actions in JOINT_ACTIONS] == list(range(729))
    np.testing.assert_array_equal(REFLECT_ACTION[REFLECT_ACTION], np.arange(9))


def test_reflection_gives_the_same_public_state_and_joint_mask():
    game = GameConfig()
    red, blue = [TeamMatch(game, 7, team, "random", scenario=entry_scenario(game, team))
                 for team in ("red", "blue")]
    for field in ("board", "positions", "action_masks", "ready"):
        np.testing.assert_array_equal(getattr(red.state, field), getattr(blue.state, field))
    np.testing.assert_array_equal(red.view()[2], blue.view()[2])
    joint = int(np.flatnonzero(red.view()[2])[0])
    np.testing.assert_array_equal(blue.world_indices(joint), REFLECT_ACTION[red.world_indices(joint)])


@pytest.mark.parametrize("team", ["red", "blue"])
def test_certified_entry_against_all_legal_opponent_actions(team):
    game = GameConfig()
    scenario = entry_scenario(game, team)
    match = TeamMatch(game, 7, team, "random", scenario=scenario)
    env = match.env
    mover, capturer, defender = match.allies
    opponent = match.enemies[0]
    joint = indices(Action.RIGHT, Action.ATTACK_DOWN, Action.ATTACK_RIGHT)
    assert match.state.choices.certified[joint]
    target = (3, 4 if team == "red" else 3)
    assert match.state.choices.entry_movers[joint].tolist() == [True, False, False]
    ally_actions = dict(zip(match.allies, match.world_indices(joint), strict=True))
    enemy_masks = [np.flatnonzero(match.observations[aid]["auxiliary"]["action_mask"]) for aid in match.enemies]
    enemy_actions = list(product(*enemy_masks))
    assert len(enemy_actions) == 729

    def succeeds(actions):
        env.reset(seed=7, options=scenario)
        env.step(actions)
        events = env.last_result.events
        moved = any(event.agent_id == mover and event.kind == "move" and event.reason == "success"
                    and event.position == target for event in events)
        died = any(event.agent_id == mover and event.kind == "death" for event in events)
        return moved and not died

    for actions in enemy_actions:
        applied = ally_actions | dict(zip(match.enemies, actions, strict=True))
        assert succeeds(applied)
        assert not succeeds(applied | {capturer: Action.NOOP.index})
        if applied[opponent] == Action.ATTACK_UP.index:
            assert not succeeds(applied | {defender: Action.NOOP.index})


def test_joint_mask_blocks_collisions_and_allows_chains():
    positions = [(3, 1), (3, 2), (2, 3), (0, 6), (1, 6), (6, 6)]
    agents = [AgentState(aid, i // 3, cell) for i, (aid, cell) in enumerate(zip(AGENT_IDS, positions, strict=True))]
    game = GameConfig()
    owner = np.zeros((8, 8), dtype=np.int8)
    owner[:, 4:] = 1
    match = TeamMatch(game, 7, "red", "random", scenario={"agents": agents, "owner": owner}, force_entry=False)
    legal = match.view()[2]
    assert not legal[indices(Action.RIGHT, Action.LEFT, Action.NOOP)]
    assert not legal[indices(Action.NOOP, Action.RIGHT, Action.DOWN)]
    chain = indices(Action.RIGHT, Action.RIGHT, Action.NOOP)
    assert legal[chain]
    match.step(chain)
    assert match.state.positions[:3].tolist() == [[3, 2], [3, 3], [2, 3]]


def test_dead_slot_cooldown_and_terminal_potential():
    game = GameConfig(height=3, width=10, max_ticks=4, respawn_delay=5)
    match = TeamMatch(game, 7, "blue", "random", force_entry=False)
    before = match.env.core.snapshot()
    agents = [AgentState(**vars(agent)) for agent in before.agents]
    agents[3].alive, agents[3].respawn_at = False, 5
    agents[4].last_attack_tick = 0
    match = TeamMatch(game, 7, "blue", "random", force_entry=False, scenario={"agents": agents, "owner": before.owner})
    board, _ = match.state.spatial_input()
    assert board.shape == (9, 8, 10)
    assert board[8].sum() == 30
    legal = JOINT_ACTIONS[match.view()[2]]
    assert (legal[:, 0] == Action.NOOP.index).all()
    assert (legal[:, 1] < Action.ATTACK_UP.index).all()
    start_gap = match.gap
    rewards = []
    while not match.done:
        rewards.append(match.step(0)[0])
    discounted = sum(DISCOUNT**i * reward for i, reward in enumerate(rewards))
    assert discounted == pytest.approx(DISCOUNT**(len(rewards) - 1) * np.sign(match.gap) - 0.25 * start_gap)
