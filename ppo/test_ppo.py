import torch

from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.models import Action

from .policy import CommunicationPolicy, encode_proposals, encode_states
from .rollout import FeedbackMatch, collect_tick, current_states, generalized_advantage
from .trainer import TrainingConfig, optimize, prepare_update


def test_feedback_rounds_freeze_the_board_and_hide_opponent_actions():
    match = FeedbackMatch(GameConfig(max_ticks=1), ExperimentConfig(decision_strategy="feedback"))
    match.reset(7, "blue", "random")
    before = encode_states(current_states([match]))
    observations = match.begin([Action.NOOP.index] * 3)
    proposals, known = encode_proposals(observations)
    assert match.env.last_result is None
    assert torch.equal(encode_states(observations).board, before.board)
    assert all(obs["tick"] == 0 and obs["action_source"] == "first_pass_proposal" for obs in observations)
    assert known.all()
    assert proposals.argmax(dim=-1).eq(Action.NOOP.index).all()
    masked, known = encode_proposals(observations, hide_teammates=True)
    assert torch.equal(known, torch.eye(3, dtype=torch.bool))
    assert torch.equal(masked[:, :, Action.NOOP.index], known.float())
    _, done = match.finish([Action.NOOP.index] * 3)
    assert done
    assert match.env.last_result.tick == 1


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
