from dataclasses import asdict, dataclass
import json
import math

import numpy as np
import torch
from torch import Tensor

from .evaluation import evaluate
from .policy import CommunicationPolicy, StateBatch, encode_states
from .rollout import FeedbackMatch, TickBatch, collect_tick, current_states, generalized_advantage
from .sender import SenderActionPredictor, attach_teammate_gradients


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    num_envs: int = 8
    rollout_ticks: int = 64
    updates: int = 196
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch_size: int = 256
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    eval_every: int = 20
    validation_seed: int = 100_000
    validation_pairs: int = 2
    hide_teammates: bool = False
    opponents: tuple[str, ...] = ("random", "expand")
    sender_prediction_coef: float = 0.0
    backpropagate_messages: bool = False

    def __post_init__(self):
        if not math.isfinite(self.sender_prediction_coef) or self.sender_prediction_coef < 0:
            raise ValueError("sender_prediction_coef must be finite and non-negative")


@dataclass
class PPOBatch:
    state: StateBatch
    proposals: Tensor
    known: Tensor
    proposal_mask: Tensor
    action_mask: Tensor
    proposal: Tensor
    action: Tensor
    old_log_probability: Tensor
    advantage: Tensor
    value_target: Tensor


def prepare_update(rollout: list[TickBatch], bootstrap, cfg):
    rewards = torch.stack([tick.reward for tick in rollout])
    values = torch.stack([tick.value for tick in rollout])
    terminated = torch.stack([tick.terminated.repeat_interleave(3) for tick in rollout])
    advantage = generalized_advantage(rewards, values, terminated, bootstrap, cfg.gamma, cfg.gae_lambda).flatten()

    def concatenate(name):
        return torch.cat([getattr(tick, name) for tick in rollout])

    # Keep three consecutive teammates together so the sender for any receiver
    # can be located even when PPO shuffles individual rows into minibatches.
    return PPOBatch(
        state=StateBatch(torch.cat([tick.state.board for tick in rollout]),
                         torch.cat([tick.state.features for tick in rollout])),
        proposals=concatenate("proposals"),
        known=concatenate("known"),
        proposal_mask=concatenate("proposal_mask"),
        action_mask=concatenate("action_mask"),
        proposal=concatenate("proposal"),
        action=concatenate("action"),
        old_log_probability=concatenate("log_probability"),
        advantage=(advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8),
        value_target=advantage + values.flatten(),
    )


def optimize(policy, optimizer, batch, cfg, *, sender_predictor=None):
    diagnostics = []
    prediction_losses = []
    parameters = list(policy.parameters())
    if sender_predictor is not None:
        parameters.extend(sender_predictor.parameters())
    for _ in range(cfg.epochs):
        for idx in torch.randperm(len(batch.advantage)).split(cfg.minibatch_size):
            latent = policy.encode(StateBatch(batch.state.board[idx], batch.state.features[idx]))
            proposals = policy.proposal(latent, batch.proposal_mask[idx])
            messages = attach_teammate_gradients(policy, batch, idx) if cfg.backpropagate_messages else batch.proposals[idx]
            actions = policy.action(latent, messages, batch.known[idx], batch.action_mask[idx])
            log_probability = proposals.log_prob(batch.proposal[idx]) + actions.log_prob(batch.action[idx])
            log_ratio = log_probability - batch.old_log_probability[idx]
            ratio = log_ratio.exp()
            advantage = batch.advantage[idx]
            clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage
            policy_loss = -torch.minimum(ratio * advantage, clipped).mean()
            value_loss = (policy.value(latent) - batch.value_target[idx]).square().mean()
            entropy = (proposals.entropy() + actions.entropy()).mean() / 2
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            if sender_predictor is not None:
                prediction = sender_predictor.loss(
                    proposals.probs, batch.proposal[idx], batch.action[idx], batch.action_mask[idx]
                )
                loss = loss + cfg.sender_prediction_coef * prediction
                prediction_losses.append(prediction.item())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                diagnostics.append((policy_loss.item(), value_loss.item(), entropy.item(),
                                    (ratio - 1 - log_ratio).mean().item()))
    result = dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl"),
                      np.mean(diagnostics, axis=0).tolist()))
    if prediction_losses:
        result["sender_prediction_loss"] = float(np.mean(prediction_losses))
    return result


def save_checkpoint(path, policy, game, experiment, cfg, update, *, sender_predictor=None):
    temporary = path.with_suffix(".tmp")
    checkpoint = {
        "model": policy.state_dict(),
        "canvas_shape": policy.canvas_shape,
        "game": asdict(game),
        "experiment": asdict(experiment),
        "training": asdict(cfg),
        "update": update,
    }
    if sender_predictor is not None:
        checkpoint["sender_action_predictor"] = sender_predictor.state_dict()
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(game, experiment, cfg, output, *, canvas_shape=None):
    canvas_shape = canvas_shape or (game.height, game.width)
    if canvas_shape[0] < game.height or canvas_shape[1] < game.width:
        raise ValueError("policy canvas must contain the training map")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg.seed)
    policy = CommunicationPolicy(*canvas_shape)
    sender_predictor = None
    parameters = list(policy.parameters())
    if cfg.sender_prediction_coef:
        # Auxiliary initialization must not shift the rollout RNG for this seed.
        with torch.random.fork_rng(devices=[]):
            sender_predictor = SenderActionPredictor()
        parameters.extend(sender_predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, eps=1e-5)
    rng = np.random.default_rng(cfg.seed)
    matches = [FeedbackMatch(game, experiment) for _ in range(cfg.num_envs)]
    experiment = matches[0].env.core.experiment

    def reset_match(match, team):
        match.reset(int(rng.integers(0, 2**32)), team, str(rng.choice(cfg.opponents)))

    for index, match in enumerate(matches):
        reset_match(match, "red" if index % 2 == 0 else "blue")

    metadata = {
        "game": asdict(game), "canvas_shape": policy.canvas_shape,
        "experiment": asdict(experiment), "training": asdict(cfg),
        "torch": str(torch.__version__), "numpy": np.__version__,
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best = (-float("inf"), -float("inf"))
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for update in range(cfg.updates + 1):
            row = {"update": update, "env_ticks": update * cfg.num_envs * cfg.rollout_ticks}
            if update:
                rollout = []
                for _ in range(cfg.rollout_ticks):
                    tick = collect_tick(policy, matches, hide_teammates=cfg.hide_teammates)
                    rollout.append(tick)
                    for match, done in zip(matches, tick.terminated.tolist(), strict=True):
                        if done:
                            reset_match(match, "blue" if match.team == "red" else "red")
                with torch.no_grad():
                    bootstrap = policy.value(policy.encode(encode_states(current_states(matches))))
                row.update(optimize(policy, optimizer, prepare_update(rollout, bootstrap, cfg), cfg,
                                    sender_predictor=sender_predictor))
            if update % cfg.eval_every == 0 or update == cfg.updates:
                row["validation"] = evaluate(policy, game, experiment, cfg.opponents, cfg.validation_seed,
                                             cfg.validation_pairs, cfg.hide_teammates)
                overall = row["validation"]["overall"]
                rank = (overall["game_score"], overall["mean_area_gap"])
                if rank > best:
                    best = rank
                    save_checkpoint(output / "best.pt", policy, game, experiment, cfg, update,
                                    sender_predictor=sender_predictor)
                save_checkpoint(output / "latest.pt", policy, game, experiment, cfg, update,
                                sender_predictor=sender_predictor)
                print(json.dumps(row), flush=True)
            if update == 0:
                save_checkpoint(output / "initial.pt", policy, game, experiment, cfg, update,
                                sender_predictor=sender_predictor)
            log.write(json.dumps(row) + "\n")
            log.flush()
