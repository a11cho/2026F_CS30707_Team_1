import torch
from torch import nn
from torch.nn import functional as F

from territory_rl.models import Action

from .policy import StateBatch


def straight_through_message(hard, probabilities):
    """Keep the transmitted symbol in the forward pass and differentiate the probabilities."""
    return hard + (probabilities - probabilities.detach())


def attach_teammate_gradients(policy, batch, receiver_rows):
    messages = batch.proposals[receiver_rows]
    team_size = messages.shape[1]
    slots = torch.arange(team_size, device=receiver_rows.device)
    own_slot = receiver_rows.remainder(team_size).unsqueeze(1)
    links = batch.known[receiver_rows] & (slots != own_slot)
    recipients, sender_slots = links.nonzero(as_tuple=True)
    if not recipients.numel():
        return messages

    # Rollout rows are grouped by tick, match, and then team slot. The needed
    # sender need not itself be one of the selected receivers in this minibatch.
    team_starts = receiver_rows[recipients] - own_slot[recipients, 0]
    sender_rows, inverse = torch.unique(team_starts + sender_slots, return_inverse=True)
    sender_state = StateBatch(batch.state.board[sender_rows], batch.state.features[sender_rows])
    sender_latent = policy.encode(sender_state)
    probabilities = policy.proposal(sender_latent, batch.proposal_mask[sender_rows]).probs
    linked = straight_through_message(messages[recipients, sender_slots], probabilities[inverse])
    return messages.index_put((recipients, sender_slots), linked)


class SenderActionPredictor(nn.Linear):
    """Predict the sender's final action from its transmitted proposal alone."""

    def __init__(self):
        super().__init__(len(Action), len(Action))

    def loss(self, proposal_probabilities, sent, executed, choice_mask):
        hard = F.one_hot(sent, num_classes=self.in_features).to(dtype=proposal_probabilities.dtype)
        # The decoder sees the actual message. The soft path provides a
        # straight-through gradient to the sender's current proposal logits.
        message = straight_through_message(hard, proposal_probabilities)
        per_agent = F.cross_entropy(self(message), executed, reduction="none")
        eligible = choice_mask.sum(dim=-1) > 1
        return (per_agent * eligible).sum() / eligible.sum().clamp_min(1)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        state = checkpoint.get("sender_action_predictor")
        if state is None:
            return None
        predictor = cls()
        predictor.load_state_dict(state)
        return predictor.eval()

    @torch.no_grad()
    def correct(self, sent, executed, choice_mask):
        message = F.one_hot(sent, num_classes=self.in_features).to(dtype=self.weight.dtype)
        eligible = choice_mask.sum(dim=-1) > 1
        return (self(message).argmax(dim=-1) == executed) & eligible, eligible
