import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from team_policy.communication import Intent, SlotTeamPolicy, teacher_intents
from team_policy.match import TeamMatch, entry_scenario
from team_policy.slot_encoder import SlotLoRA
from team_policy.state import joint_index
from territory_rl.config import GameConfig
from territory_rl.models import Action


@pytest.mark.parametrize("team", ["red", "blue"])
def test_slot_teacher_and_commit_share_the_certified_entry(team):
    game = GameConfig()
    match = TeamMatch(game, 7, team, "noop", scenario=entry_scenario(game, team))
    policy = SlotTeamPolicy("cnn", "cpu")
    sample, = policy.prepare([match.view()])
    joint = joint_index(tuple(action.index for action in (Action.RIGHT, Action.ATTACK_DOWN, Action.ATTACK_RIGHT)))
    labels = teacher_intents(match.state, joint)

    assert match.state.choices.certified[joint]
    assert [Intent((i - 1) // 4 + 1) for i in labels] == [Intent.ENTER, Intent.CAPTURE, Intent.COVER]
    assert sample.intent_mask[np.arange(3), labels].all()
    with torch.no_grad():
        chosen = policy.decide([sample])
        _, reconstructed = policy.commit([sample], prescribed=chosen.actions)
    assert match.state.choices.certified[chosen.joint.item()]
    assert torch.equal(chosen.action_masks, reconstructed)
    match.step(chosen.joint.item())
    assert match.entry_successes == 1


@pytest.mark.parametrize("communication", [False, True])
def test_replay_uses_the_recorded_message_and_action_probabilities(communication):
    game = GameConfig()
    matches = [TeamMatch(game, seed, "red", "noop") for seed in (11, 12)]
    matches[1].step(0)
    policy = SlotTeamPolicy("cnn", "cpu", communication=communication)
    samples = policy.prepare([match.view() for match in matches])
    action_rng = torch.Generator().manual_seed(47)
    message_rng = torch.Generator().manual_seed(48)
    with torch.no_grad():
        choice = policy.decide(samples, message_rng=message_rng, action_rng=action_rng)
        send, act, _ = policy.distributions(samples, choice.intent, choice.action_masks)
        replay = (send.log_prob(choice.intent) + act.log_prob(choice.actions)).sum(dim=1)
    torch.testing.assert_close(replay, choice.log_probability)
    if not communication:
        assert choice.intent.count_nonzero() == 0


def test_slot_adapter_gradients_stay_with_the_selecting_slot():
    class Original:
        scaling = {"default": 2.0}

        def __init__(self):
            self.original = nn.Linear(4, 6, bias=False).requires_grad_(False)
            self.lora_A = {"default": nn.Linear(4, 2, bias=False)}
            self.lora_B = {"default": nn.Linear(2, 6, bias=False)}

        def get_base_layer(self):
            return self.original

    original = Original()
    encoder = SlotLoRA(original)
    features = torch.randn(6, 2, 4)
    expected = original.original(features) + 2 * original.lora_B["default"](original.lora_A["default"](features))
    output = encoder(features)
    torch.testing.assert_close(output, expected)
    output[2:4].square().sum().backward()
    assert encoder.a.grad[1].count_nonzero() > 0 and encoder.b.grad[1].count_nonzero() > 0
    assert encoder.a.grad[[0, 2]].count_nonzero() == 0 and encoder.b.grad[[0, 2]].count_nonzero() == 0
    assert encoder.base.weight.grad is None
