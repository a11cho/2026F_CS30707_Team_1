import sys

import pytest
import torch

from territory_rl.config import ExperimentConfig, GameConfig, TileEncoding
from territory_rl.models import Action

from . import __main__ as cli
from .policy import CommunicationPolicy, encode_proposals, encode_states
from .rollout import FeedbackMatch, collect_tick, current_states, generalized_advantage
from .trainer import TrainingConfig, optimize, prepare_update


@pytest.mark.parametrize("rotation", range(4))
def test_feedback_rounds_freeze_the_board_and_hide_opponent_actions(rotation):
    encoding = TileEncoding(outside=-10, friendly=11, enemy=12,
                            friendly_agents=(13, 14, 15), enemy_agents=(16, 17, 18))
    match = FeedbackMatch(GameConfig(max_ticks=1),
                          ExperimentConfig(decision_strategy="feedback", tile_encoding=encoding,
                                           randomize_observation_rotation=bool(rotation)))
    match.reset(7, "blue", "expand", options={"observation_rotation": rotation})
    before = encode_states(current_states([match]))
    assert before.board[:, 0, :, 4:].all() and not before.board[:, 0, :, :4].any()
    assert before.board[:, 2:, :, :].sum() == 18
    action = Action.ATTACK_LEFT
    observations = match.begin([action.index] * 3)
    assert all(match.env.core.action_to_world(move) == Action.ATTACK_RIGHT for move in match.bot_actions.values())
    assert all(obs["action_mask"][action.index] for obs in observations)
    proposals, known = encode_proposals(observations)
    assert match.env.last_result is None
    assert torch.equal(encode_states(observations).board, before.board)
    assert all(obs["tick"] == 0 and obs["action_source"] == "first_pass_proposal" for obs in observations)
    assert known.all()
    assert proposals.argmax(dim=-1).eq(action.index).all()
    masked, known = encode_proposals(observations, hide_teammates=True)
    assert torch.equal(known, torch.eye(3, dtype=torch.bool))
    assert torch.equal(masked[:, :, action.index], known.float())
    _, done = match.finish([Action.NOOP.index] * 3)
    assert done
    assert match.env.last_result.tick == 1


@pytest.mark.parametrize("flags, mask, weight", [
    ([], (0, 0, 1), 1.0),
    (["--reward-mask", "0", "1", "1", "--gap-weight", "0.0001220703125"], (0, 1, 1), 1 / 8192),
])
def test_train_cli_passes_reward_settings_to_ppo(monkeypatch, tmp_path, flags, mask, weight):
    experiments = []

    def record_experiment(game, experiment, config, output, *, canvas_shape):
        experiments.append(experiment)

    monkeypatch.setattr(cli, "train", record_experiment)
    monkeypatch.setattr(sys, "argv", ["ppo", "train", "--output", str(tmp_path / "run"), *flags])
    cli.main()

    experiment, = experiments
    assert experiment.resolved_reward_mask == mask
    assert experiment.gap_weight == weight
    assert experiment.loss_reward == -1.0


def test_both_sampled_choices_receive_a_ppo_update_with_rollout_masks():
    game = GameConfig(max_ticks=5)
    match = FeedbackMatch(game, ExperimentConfig(decision_strategy="feedback"))
    match.reset(7, "red", "expand")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    policy = CommunicationPolicy(game.height, game.width)
    rollout = [collect_tick(policy, [match]) for _ in range(3)]
    cfg = TrainingConfig(epochs=1)
    with torch.no_grad():
        bootstrap = policy.value(policy.encode(encode_states(current_states([match]))))
    batch = prepare_update(rollout, bootstrap, cfg)
    assert batch.proposal_mask.gather(1, batch.proposal[:, None]).all()
    assert batch.action_mask.gather(1, batch.action[:, None]).all()
    before = [head.weight.detach().clone() for head in (policy.proposal_head, policy.action_head[-1])]
    stats = optimize(policy, torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate), batch, cfg)
    assert abs(stats["approx_kl"]) < 1e-6
    assert not torch.equal(policy.proposal_head.weight, before[0])
    assert not torch.equal(policy.action_head[-1].weight, before[1])


def test_gae_distinguishes_game_termination_from_a_rollout_cut():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.zeros_like(rewards)
    terminal = torch.tensor([[True, False], [False, False]])
    bootstrap = torch.tensor([6.0, 6.0])
    advantage = generalized_advantage(rewards, values, terminal, bootstrap, gamma=0.5, gae_lambda=1.0)
    assert torch.equal(advantage, torch.tensor([[1.0, 3.5], [5.0, 5.0]]))
