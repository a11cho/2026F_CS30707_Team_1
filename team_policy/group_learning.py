from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch

from .match import DISCOUNT, TeamMatch
from .optimization import batches, clipped_objective, optimize
from .policy import DecisionInput


MIN_RETURN_STD = 1e-6


@dataclass
class Episode:
    match: TeamMatch
    generator: torch.Generator
    inputs: list[DecisionInput] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    discounted_return: float = 0.0
    discount: float = 1.0

    def record(self, sample, action, log_prob):
        self.inputs.append(sample)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        reward, _ = self.match.step(action)
        self.discounted_return += self.discount * reward
        self.discount *= DISCOUNT


@dataclass
class GroupRollout:
    inputs: list[DecisionInput]
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor


def relative_advantages(returns):
    returns = np.asarray(returns, dtype=np.float64)
    std = float(returns.std())
    advantages = (returns - returns.mean()) / std if std > MIN_RETURN_STD else np.zeros_like(returns)
    return advantages, std


@torch.no_grad()
def play_groups(policy, groups, inference_batch):
    active = [episode for group in groups for episode in group]
    while active:
        for group in batches(active, inference_batch):
            samples = policy.prepare([episode.match.view() for episode in group])
            distribution = policy.actor(samples)
            actions = torch.cat([torch.multinomial(prob, 1, generator=ep.generator)
                                 for ep, prob in zip(group, distribution.probs, strict=True)])
            log_probs = distribution.log_prob(actions).tolist()
            for ep, sample, action, lp in zip(group, samples, actions.tolist(), log_probs, strict=True):
                ep.record(sample, action, lp)
        active = [episode for episode in active if not episode.match.done]


def collect_groups(policy, source, args, available_ticks, log, prior_ticks):
    """Collect complete groups while reserving the maximum length before each group starts."""
    max_group_ticks = source.game.max_ticks * args.group_size
    target = max(1, args.envs * args.rollout_steps // max_group_ticks)
    concurrency = args.envs // args.group_size
    stats = Counter()
    inputs, actions, log_probs, all_advantages, all_weights, summaries = [], [], [], [], [], []
    start = perf_counter()
    while stats["groups_kept"] < target and available_ticks - stats["ticks"] >= max_group_ticks:
        wave = min(target - stats["groups_kept"], concurrency, (available_ticks - stats["ticks"]) // max_group_ticks)
        groups = []
        for _ in range(wave):
            fixtures = source.next_group(args.group_size)
            group_id = source.started
            seeds = np.random.SeedSequence(args.seed, spawn_key=(group_id,)).generate_state(args.group_size, dtype=np.uint64)
            groups.append([Episode(match, torch.Generator(device=policy.device).manual_seed(int(seed)))
                           for match, seed in zip(fixtures, seeds, strict=True)])
        play_groups(policy, groups, args.inference_batch)
        for group in groups:
            returns = [ep.discounted_return for ep in group]
            advantages, std = relative_advantages(returns)
            gaps = [ep.match.gap for ep in group]
            lengths = [ep.match.ticks for ep in group]
            varied = std > MIN_RETURN_STD
            kept = args.algorithm == "grpo" or varied
            stats["ticks"] += sum(lengths)
            stats["groups_collected"] += 1
            stats["groups_kept"] += int(kept)
            stats["groups_with_return_signal"] += int(varied)
            stats["groups_with_mixed_outcomes"] += int(len(set(np.sign(gaps))) > 1)
            stats["entry_opportunities"] += sum(ep.match.entry_opportunities for ep in group)
            stats["entry_successes"] += sum(ep.match.entry_successes for ep in group)
            first = group[0].match
            summaries.append({"seed": first.seed, "team": first.team, "opponent": first.opponent,
                              "returns": returns, "return_std": std, "gaps": gaps, "lengths": lengths, "kept": kept})
            if not kept:
                stats["discarded_ticks"] += sum(lengths)
                continue
            for ep, advantage in zip(group, advantages, strict=True):
                valid = [i for i, sample in enumerate(ep.inputs) if sample.mask.sum() > 1]
                if not valid:
                    continue
                weight = 1.0 / len(valid) if args.algorithm == "grpo" else 1.0
                for i in valid:
                    inputs.append(ep.inputs[i])
                    actions.append(ep.actions[i])
                    log_probs.append(ep.log_probs[i])
                    all_advantages.append(advantage)
                    all_weights.append(weight)
                stats["signal_ticks"] += len(valid) if varied else 0
        log.write("group_progress", ticks=prior_ticks + stats["ticks"], groups=stats["groups_collected"],
                  groups_kept=stats["groups_kept"], seconds=perf_counter() - start)
    weights = torch.tensor(all_weights, dtype=torch.float32)
    if len(weights):
        weights /= weights.mean()
    rollout = GroupRollout(inputs, torch.tensor(actions, dtype=torch.long), torch.tensor(log_probs, dtype=torch.float32),
                           torch.tensor(all_advantages, dtype=torch.float32), weights)
    seconds = perf_counter() - start
    return rollout, {**stats, "actionable_ticks": len(inputs), "groups": summaries, "seconds": seconds,
                     "ticks_per_second": stats["ticks"] / seconds}


@torch.no_grad()
def reference_log_probs(policy, rollout, weights, batch_size):
    with policy.using_trainable_weights(weights):
        return torch.cat([policy.actor([rollout.inputs[i] for i in indices])
                          .log_prob(rollout.actions[indices].to(policy.device)).cpu()
                          for indices in batches(range(len(rollout.inputs)), batch_size)])


def grouped_update(policy, optimizer, rollout, args, rng, log, reference):
    count = len(rollout.inputs)
    if not count:
        return {"optimizer_steps": 0, "seconds": 0.0, "actionable_ticks": 0}
    upper = 1.28 if args.algorithm == "dapo" else 1.2

    def backward(indices, microbatch):
        stats = Counter()
        for mini in batches(indices, microbatch):
            distribution = policy.actor([rollout.inputs[i] for i in mini])
            current = distribution.log_prob(rollout.actions[mini].to(policy.device))
            ratio_log = current - rollout.log_probs[mini].to(policy.device)
            ratio = ratio_log.exp()
            weights = rollout.weights[mini].to(policy.device)
            advantage = rollout.advantages[mini].to(policy.device)
            actor = -(weights * clipped_objective(ratio, advantage, upper)).sum()
            loss = actor
            if reference is not None:
                log_ratio = reference[mini].to(policy.device) - current
                kl = (weights * (log_ratio.expm1() - log_ratio)).sum()
                loss = loss + 0.01 * kl
                stats["reference_kl"] += kl.detach().item()
            (loss / len(indices)).backward()
            stats["actor"] += actor.detach().item()
            stats["approx_kl"] += (ratio_log.detach().expm1() - ratio_log.detach()).sum().item()
            stats["clips"] += ((ratio.detach() < 0.8) | (ratio.detach() > upper)).sum().item()
        return stats

    sums, metrics = optimize(policy, optimizer, count, args, rng, log, backward, args.algorithm)
    decisions = count * args.epochs
    return {**metrics, "actionable_ticks": count, "actor_loss": sums["actor"] / decisions,
            "approx_kl": sums["approx_kl"] / decisions, "clip_fraction": sums["clips"] / decisions,
            "reference_kl": sums["reference_kl"] / decisions if reference is not None else None}
