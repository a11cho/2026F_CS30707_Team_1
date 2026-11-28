import pytest
import torch

from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.models import Action

from .policy import CommunicationPolicy
from .tactics import Scenario, choose_actions, outcome, prepare


def execute(kind, team, roles, actions_by_role):
    scenario = Scenario(kind, 3, team, "inset", roles, 64)
    case = prepare(scenario, GameConfig(), ExperimentConfig(decision_strategy="feedback"), seed=7)
    indices = [actions_by_role[role].index for role in roles]
    case.match.begin(indices)
    case.match.finish(indices)
    return outcome(case)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_collision_uses_the_environment_movement_result(team):
    both = execute("collision", team, (2, 0, 1), (Action.DOWN, Action.UP, Action.NOOP))
    one = execute("collision", team, (2, 0, 1), (Action.DOWN, Action.NOOP, Action.NOOP))
    assert both == {"collision": True, "reached_target": False}
    assert one == {"collision": False, "reached_target": True}


@pytest.mark.parametrize("team", ["red", "blue"])
def test_cover_requires_counterattack_on_the_protected_tile(team):
    exposed = execute("cover", team, (2, 0, 1), (Action.NOOP, Action.NOOP, Action.NOOP))
    covered = execute("cover", team, (2, 0, 1), (Action.NOOP, Action.ATTACK_DOWN, Action.NOOP))
    assert exposed == {"defended": False, "held_tile": False, "ally_counterattacked": False}
    assert covered == {"defended": True, "held_tile": True, "ally_counterattacked": True}


def test_forced_proposals_reach_both_ablations_with_the_same_action_codes():
    torch.set_num_threads(1)
    game = GameConfig()
    experiment = ExperimentConfig(decision_strategy="feedback")
    policy = CommunicationPolicy(game.height, game.width).eval()
    scenarios = [Scenario(kind, 3, "blue", "flat", (2, 0, 1), 64) for kind in ("collision", "cover")]
    batches = [[prepare(scenario, game, experiment, seed=7) for scenario in scenarios] for _ in range(2)]
    on, _ = choose_actions(policy, batches[0], hide_teammates=False, force_proposals=True)
    off, _ = choose_actions(policy, batches[1], hide_teammates=True, force_proposals=True)
    assert on == off
    assert on[0][1:] == [Action.DOWN.index, Action.UP.index]
    assert (on[1][0], on[1][2]) == (Action.ATTACK_UP.index, Action.ATTACK_DOWN.index)
