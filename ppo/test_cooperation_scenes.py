import numpy as np
import pytest

from territory_rl.agent import AgentState
from territory_rl.environment import TerritoryEnvironment
from territory_rl.models import AGENT_IDS, Action

from .cooperation_scenes import SceneTick, SceneTracker, summarize_scenes


RED = AGENT_IDS[:3]


def scenario(red, blue, *, changed=None):
    env = TerritoryEnvironment()
    owner = np.zeros((8, 8), dtype=np.int8)
    owner[:, 4:] = 1
    for cell, team in (changed or {}).items():
        owner[cell] = team
    agents = [AgentState(aid, slot // 3, position, respawn_at=128 if position is None else None)
              for slot, (aid, position) in enumerate(zip(AGENT_IDS, red + blue, strict=True))]
    env.reset(seed=7, options={"owner": owner, "agents": agents})
    return env, SceneTracker()


def step(game, chosen, *, proposals=None):
    env, tracker = game
    actions = {aid: chosen.get(aid, Action.NOOP) for aid in AGENT_IDS}
    before = env.snapshot()
    result = env.step(actions)
    assert all(event.kind != "rejected" for event in result.events)
    proposed = [action.index for action in (proposals or [actions[aid] for aid in RED])]
    tracker.update(SceneTick(before, result, 0, RED, proposed, actions))
    return result


def counts(game, key):
    result = summarize_scenes(game[1].counts)[key]
    return result["count"], result["opportunities"]


@pytest.mark.parametrize("attack_together", [True, False])
def test_breakthrough_and_entry_depend_on_both_attackers(attack_together):
    game = scenario([(3, 3), (2, 4), (4, 4)], [(3, 5), (0, 6), (6, 6)],
                    changed={(2, 4): 0, (4, 4): 0})
    actions = {"red_0": Action.ATTACK_RIGHT, "red_2": Action.UP, "blue_0": Action.ATTACK_LEFT}
    if attack_together:
        actions["red_1"] = Action.ATTACK_DOWN
    step(game, actions)
    assert counts(game, "breakthrough") == (int(attack_together), 1)
    successes, opportunities = counts(game, "capture_entry")
    assert opportunities > 0 and successes == int(attack_together)


@pytest.mark.parametrize("defend", [True, False])
def test_cover_distinguishes_survival_from_immediate_respawn(defend):
    game = scenario([(3, 3), (2, 3), (5, 1)], [(3, 4), (0, 6), (6, 6)])
    actions = {"blue_0": Action.ATTACK_LEFT}
    if defend:
        actions["red_1"] = Action.ATTACK_DOWN
    result = step(game, actions)
    assert counts(game, "cover") == (int(defend), 1)
    if not defend:
        assert any(event.kind == "respawn" and event.agent_id == "red_0" for event in result.events)


@pytest.mark.parametrize("third_ally", [(6, 1), None])
def test_chain_requires_the_leader_to_vacate_its_tile(third_ally):
    game = scenario([(3, 3), (3, 2), third_ally], [(2, 5), (3, 5), (6, 6)])
    step(game, {"red_0": Action.UP, "red_1": Action.RIGHT})
    assert counts(game, "yield_move") == (1, 1)
    assert counts(game, "chain_move") == (1, 1)


@pytest.mark.parametrize("takes_detour", [True, False])
def test_proposal_conflict_needs_both_final_moves_to_succeed(takes_detour):
    game = scenario([(3, 3), (4, 2), (6, 1)], [(2, 5), (3, 5), (6, 6)])
    step(game, {"red_0": Action.DOWN, "red_1": Action.UP if takes_detour else Action.NOOP},
         proposals=[Action.DOWN, Action.RIGHT, Action.NOOP])
    assert counts(game, "conflicting_proposals") == (int(takes_detour), 1)
    assert counts(game, "yield_move") == (int(takes_detour), 1)


@pytest.mark.parametrize("splits", [True, False])
def test_two_distinct_targets_gain_one_tile_over_an_unnecessary_duplicate(splits):
    game = scenario([(3, 3), (2, 4), (5, 1)], [(3, 6), (2, 6), (6, 6)], changed={(2, 4): 0})
    step(game, {"red_0": Action.ATTACK_RIGHT, "red_1": Action.ATTACK_RIGHT if splits else Action.ATTACK_DOWN})
    assert counts(game, "split_targets") == (int(splits), 1)


@pytest.mark.parametrize("continues", [True, False])
def test_cooldown_rotation_tracks_the_same_front_next_tick(continues):
    game = scenario([(3, 3), (4, 3), (5, 1)], [(1, 6), (3, 6), (6, 6)])
    step(game, {"red_0": Action.ATTACK_RIGHT})
    assert counts(game, "cooldown_rotation") == (0, 0)
    step(game, {"red_0": Action.UP, "red_1": Action.ATTACK_RIGHT if continues else Action.NOOP})
    assert counts(game, "cooldown_rotation") == (int(continues), 1)


@pytest.mark.parametrize("hands_over_defense", [True, False])
def test_role_switch_tracks_two_functional_assignments(hands_over_defense):
    game = scenario([(3, 3), (2, 3), (4, 3)], [(2, 4), (5, 3), (6, 6)],
                    changed={(1, 3): 1, (5, 3): 1})
    step(game, {"blue_0": Action.ATTACK_LEFT, "red_0": Action.ATTACK_UP,
                "red_1": Action.ATTACK_UP, "red_2": Action.ATTACK_RIGHT})
    assert counts(game, "role_split") == (1, 1)
    step(game, {"red_0": Action.DOWN, "red_2": Action.RIGHT})
    step(game, {"blue_1": Action.ATTACK_UP, "red_0": Action.ATTACK_DOWN, "red_1": Action.ATTACK_RIGHT,
                "red_2": Action.ATTACK_LEFT if hands_over_defense else Action.NOOP})
    assert counts(game, "role_switch") == (int(hands_over_defense), 1)
