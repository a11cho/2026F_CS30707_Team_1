import torch

from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.models import Action
from territory_rl.parallel_env import TerritoryParallelEnv

from .cooperation import cooperation_counts
from .policy import CommunicationPolicy, game_config_from_metadata, policy_from_checkpoint
from .rollout import FeedbackMatch, collect_tick
from .tactics import Scenario, prepare


def test_saved_canvas_executes_on_both_physical_maps():
    torch.set_num_threads(1)
    policy = CommunicationPolicy(8, 10)
    checkpoint = {
        "game": {"height": 3, "width": 10, "rules_version": "territory_v1"},
        "canvas_shape": policy.canvas_shape,
        "model": policy.state_dict(),
    }
    policy = policy_from_checkpoint(checkpoint)
    trained_game = game_config_from_metadata(checkpoint["game"])
    assert trained_game == GameConfig(height=3, width=10)
    experiment = ExperimentConfig(decision_strategy="feedback")
    for height, width in ((3, 10), (8, 8)):
        match = FeedbackMatch(GameConfig(height=height, width=width), experiment)
        match.reset(7, "red", "random")
        assert match.env.core.experiment.observation_window_size == (19 if width == 10 else 15)
        batch = collect_tick(policy, [match], greedy=True)
        assert batch.state.board.shape == (3, 8, height, width)
        assert batch.action_mask.gather(1, batch.action[:, None]).all()
        assert match.env.last_result.tick == 1


def test_full_horizon_respawn_delay_keeps_dead_agents_out():
    game = GameConfig(height=3, width=10, max_ticks=8, respawn_delay=8)
    env = TerritoryParallelEnv(game, ExperimentConfig(observation_window_size=19))
    env.reset(seed=7)
    actions = dict.fromkeys(env.agents, Action.NOOP.index)
    actions["blue_0"] = Action.ATTACK_LEFT.index
    observations, _, _, _, _ = env.step(actions)
    assert not observations["red_0"]["auxiliary"]["agent_alive"][0]
    actions["blue_0"] = Action.NOOP.index
    for _ in range(7):
        observations, _, terminated, _, _ = env.step(actions)
        assert not observations["red_0"]["auxiliary"]["agent_alive"][0]
        assert observations["red_0"]["auxiliary"]["action_mask"].sum() == 1
        assert all(event.kind != "respawn" for event in env.last_result.events)
    assert terminated["red_0"]


def test_cooperation_marks_a_yield_and_a_counterattack():
    game = GameConfig()
    experiment = ExperimentConfig(decision_strategy="feedback")
    collision = prepare(Scenario("collision", 3, "red", "flat", (0, 1, 2), 1), game, experiment, seed=7).match
    before = collision.observations["red_0"]
    proposals = [Action.DOWN.index, Action.UP.index, Action.NOOP.index]
    collision.begin(proposals)
    collision.finish([Action.DOWN.index, Action.NOOP.index, Action.NOOP.index])
    counts = cooperation_counts(before, collision.env.last_result.events, collision.learners, proposals)
    assert (counts["proposal_conflicts"], counts["resolved_proposal_conflicts"], counts["allied_collision_moves"]) == (1, 1, 0)

    cover = prepare(Scenario("cover", 3, "blue", "flat", (0, 1, 2), 1), game, experiment, seed=7).match
    before = cover.observations["blue_0"]
    actions = [Action.NOOP.index, Action.ATTACK_DOWN.index, Action.NOOP.index]
    cover.begin(actions)
    cover.finish(actions)
    counts = cooperation_counts(before, cover.env.last_result.events, cover.learners, actions)
    assert (counts["cover_opportunities"], counts["protected_tiles"]) == (1, 1)
