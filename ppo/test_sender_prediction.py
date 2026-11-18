from types import SimpleNamespace

import torch
from torch.nn import functional as F

from territory_rl.models import Action

from .policy import CommunicationPolicy, StateBatch
from .sender import SenderActionPredictor, attach_teammate_gradients


def test_sender_prediction_sees_the_hard_symbol_and_trains_its_sender():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    policy = CommunicationPolicy(3, 10)
    predictor = SenderActionPredictor()
    sent = torch.tensor([Action.DOWN.index, Action.ATTACK_RIGHT.index])
    executed = torch.tensor([Action.UP.index, Action.NOOP.index])
    mask = torch.ones((2, len(Action)), dtype=torch.bool)
    probabilities = policy.proposal(torch.randn(2, 128), mask).probs
    loss = predictor.loss(probabilities, sent, executed, mask)
    actual_messages = F.one_hot(sent, num_classes=len(Action)).float()
    expected = F.cross_entropy(predictor(actual_messages), executed)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(policy.proposal_head.weight.grad) > 0
    assert torch.count_nonzero(predictor.weight.grad) > 0
    assert policy.action_head[-1].weight.grad is None


def test_sender_loss_and_accuracy_skip_forced_actions():
    predictor = SenderActionPredictor()
    proposals = torch.tensor([Action.NOOP.index])
    mask = torch.zeros((1, len(Action)), dtype=torch.bool)
    mask[0, Action.NOOP.index] = True
    probabilities = mask.float().requires_grad_()
    loss = predictor.loss(probabilities, proposals, proposals, mask)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(probabilities.grad) == 0
    correct, eligible = predictor.correct(proposals, proposals, mask)
    assert not correct.any() and not eligible.any()


def test_receiver_policy_gradient_reaches_only_its_visible_teammates():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    policy = CommunicationPolicy(3, 10)
    # Two different matches. Select one receiver from each, in shuffled order.
    rows = torch.tensor([4, 0])
    known = torch.ones((6, 3), dtype=torch.bool)
    known[4] = torch.tensor([True, True, False])
    known[0] = torch.tensor([True, False, True])
    symbols = F.one_hot(torch.arange(6).reshape(2, 3), num_classes=len(Action)).float()
    sent = symbols.repeat_interleave(3, dim=0).masked_fill(~known.unsqueeze(-1), 0)
    inputs = torch.randn((6, 22), requires_grad=True)
    legal = torch.ones((6, len(Action)), dtype=torch.bool)
    batch = SimpleNamespace(
        state=StateBatch(torch.zeros((6, 8, 3, 10)), inputs),
        proposals=sent,
        known=known,
        proposal_mask=legal,
    )
    messages = attach_teammate_gradients(policy, batch, rows)
    assert torch.equal(messages, sent[rows])
    receiver_state = torch.zeros((2, 128))
    actions = torch.tensor([Action.UP.index, Action.DOWN.index])
    reference = policy.action(receiver_state, sent[rows], known[rows], legal[rows]).log_prob(actions)
    linked = policy.action(receiver_state, messages, known[rows], legal[rows]).log_prob(actions)
    assert torch.equal(reference, linked)
    # This is the policy-ratio path while PPO is inside its clipping range.
    loss = -(linked - reference.detach()).exp().mean()
    loss.backward()
    # Neither receiver row nor either masked teammate contributed gradients.
    assert torch.equal(inputs.grad.abs().sum(dim=1) > 0, torch.tensor([False, False, True, True, False, False]))
    assert torch.count_nonzero(policy.proposal_head.weight.grad) > 0


def test_receiver_with_only_own_message_adds_no_gradient_path():
    policy = CommunicationPolicy(3, 10)
    sent = torch.zeros((3, 3, len(Action)))
    batch = SimpleNamespace(proposals=sent, known=torch.eye(3, dtype=torch.bool))
    messages = attach_teammate_gradients(policy, batch, torch.tensor([2, 0]))
    assert torch.equal(messages, sent[[2, 0]])
    assert not messages.requires_grad
