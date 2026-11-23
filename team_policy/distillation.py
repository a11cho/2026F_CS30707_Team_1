from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch

from territory_rl.models import Action

from .optimization import batches, optimize
from .state import JOINT_ACTIONS
from .teacher import TacticalScorer, teacher_candidates, world_actions


@dataclass(frozen=True)
class Feedback:
    text: str
    preferred: tuple[int, ...]


def command_text(index):
    return ",".join(map(str, JOINT_ACTIONS[index]))


def tactical_feedback(match, before, state, mask, chosen, rng):
    """Describe a real outcome and replay legal improvements against the realized opponent actions."""
    if mask.sum() <= 1:
        return None
    scorer = TacticalScorer(before, match.team_number, match.allies)
    enemy = {aid: Action.from_index(match.last_actions[aid]) for aid in match.enemies}
    actual = scorer(world_actions(match, chosen) | enemy)
    one_change = (JOINT_ACTIONS != JOINT_ACTIONS[chosen]).sum(axis=1) == 1
    candidates = np.union1d(np.flatnonzero(mask & one_change), teacher_candidates(state, mask, rng, 24))
    scores = np.array([scorer(world_actions(match, index) | enemy) for index in candidates])
    best = scores.max()
    if best - actual < 0.01:
        return None
    preferred = tuple(map(int, candidates[np.isclose(scores, best, atol=1e-8)][:4]))
    events = match.env.last_result.events
    losses = [f"{match.allies.index(e.agent_id)}:{e.reason}" for e in events
              if e.kind == "death" and e.agent_id in match.allies]
    blocked = [f"{match.allies.index(e.agent_id)}:{e.reason}" for e in events
               if e.kind == "move" and e.agent_id in match.allies and e.reason != "success"]
    tied = [str(match.allies.index(e.agent_id)) for e in events
            if e.kind == "attack" and e.agent_id in match.allies and e.reason == "tie"]
    area_change = match.state.areas[0] - state.areas[0]
    alternatives = " / ".join(command_text(index) for index in preferred)
    return Feedback(
        f"\nOur tried actions [{command_text(chosen)}]. The engine reported "
        f"{area_change:+d} own tiles, ally losses {','.join(losses) or 'none'}, "
        f"blocked moves {','.join(blocked) or 'none'}, tied attacks {','.join(tied) or 'none'}. "
        f"Tested better joint actions against these same opponent moves: [{alternatives}]. "
        "Choose coordinated actions from the original board.\nTeam action:", preferred,
    )


def jensen_shannon(student_logs, teacher_probs):
    student_probs = student_logs.exp()
    finite_logs = student_logs.masked_fill(student_probs == 0, 0.0)
    tiny = torch.finfo(student_probs.dtype).tiny
    mixture_logs = ((student_probs + teacher_probs) * 0.5).clamp_min(tiny).log()
    teacher_logs = teacher_probs.clamp_min(tiny).log()
    return 0.5 * (student_probs * (finite_logs - mixture_logs)
                  + teacher_probs * (teacher_logs - mixture_logs)).sum(dim=-1)


@torch.no_grad()
def teacher_targets(policy, inputs, feedback, weights, batch_size):
    start = perf_counter()
    eligible = [(sample, item) for sample, item in zip(inputs, feedback, strict=True) if item is not None]
    samples, targets, gains = [], [], []
    with policy.using_trainable_weights(weights):
        for batch in batches(eligible, batch_size):
            original, items = zip(*batch)
            suffixes = policy.tokenizer([item.text for item in items], add_special_tokens=False)["input_ids"]
            conditioned = [replace(sample, tokens=sample.tokens + suffix)
                           for sample, suffix in zip(original, suffixes, strict=True)]
            base = policy.actor(original).probs.cpu()
            teacher = policy.actor(conditioned).probs.cpu()
            for sample, item, old, target in zip(original, items, base, teacher, strict=True):
                gain = (target[list(item.preferred)].sum() - old[list(item.preferred)].sum()).item()
                gains.append(gain)
                if gain > 1e-5:
                    samples.append(sample)
                    targets.append(target)
    metrics = {"feedback_ticks": len(eligible), "teacher_forward_ticks": 2 * len(eligible),
               "accepted_feedback_ticks": len(samples), "mean_preferred_mass_gain": float(np.mean(gains)) if gains else 0.0,
               "teacher_seconds": perf_counter() - start}
    return samples, torch.stack(targets) if targets else None, metrics


def distillation_update(policy, optimizer, inputs, targets, args, rng, log):
    if not inputs:
        return {"optimizer_steps": 0, "seconds": 0.0, "actionable_ticks": 0}

    def backward(indices, microbatch):
        sums = Counter()
        for mini in batches(indices, microbatch):
            distribution = policy.actor([inputs[index] for index in mini])
            divergence = jensen_shannon(distribution.logits, targets[mini].to(policy.device)).sum()
            (divergence / len(indices)).backward()
            sums["jsd"] += divergence.detach().item()
        return sums

    sums, metrics = optimize(policy, optimizer, len(inputs), args, rng, log, backward, "sdpo")
    return {**metrics, "actionable_ticks": len(inputs), "mean_jsd": sums["jsd"] / (len(inputs) * args.epochs)}


@torch.no_grad()
def update_ema(weights, policy, decay=0.9):
    for name, current in policy.trainable_weights().items():
        weights[name].lerp_(current, 1.0 - decay)
