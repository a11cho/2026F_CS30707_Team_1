"""Train and evaluate three communicating actors against the fixed territory fixtures."""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from territory_rl.config import GameConfig

from .communication import Intent, SlotTeamPolicy, masked, teacher_intents
from .experiment import MatchSource, RunLog, checkpoint, configure_device, evaluate, evaluate_exercises
from .experiment import load_checkpoint, match_result, optimizer_for
from .match import DISCOUNT
from .optimization import batches, clipped_objective, generalized_advantage, optimize
from .policy import BASE_REVISION
from .search import SEARCH_BUDGETS, choose_search_action
from .state import JOINT_ACTIONS


TRAINING_OPPONENTS = ("search_explore_67", "search_explore_64", "search_explore_61", "search_explore_58")
OPPONENT_INTERVAL = 4096
TEACHER = "search_strong"


@dataclass
class SlotRollout:
    samples: list
    prefixes: list | None
    messages: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    old_log_probability: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    teacher_joints: torch.Tensor
    teacher_messages: torch.Tensor


def teacher_action(match, rng):
    joint = choose_search_action(match.state, rng, SEARCH_BUDGETS[TEACHER], force_entry=match.force_entry,
                                 respawn_delay=match.env.core.config.respawn_delay)
    return joint, teacher_intents(match.state, joint)


def message_counts(intents):
    kinds = np.where(intents > 0, (intents - 1) // 4 + 1, 0)
    return {Intent(i).name.lower(): int(count) for i, count in enumerate(np.bincount(kinds.flatten(), minlength=len(Intent)))}


def warm_up(policy, game, args, log):
    rng = np.random.default_rng(args.seed + 1)
    teacher_rng = np.random.default_rng(args.seed + 2)
    source = MatchSource(game, rng, (TRAINING_OPPONENTS[0], "search_intro"), 10, True,
                         weights=(0.8, 0.2), terminal_margin_weight=args.terminal_margin_weight)
    started = perf_counter()
    views, actions, messages = [], [], []
    match = source.next()
    for _ in range(args.warmup_ticks):
        joint, message = teacher_action(match, teacher_rng)
        views.append(match.view())
        actions.append(JOINT_ACTIONS[joint])
        messages.append(message)
        match.step(joint)
        if match.done:
            match = source.next()
    samples = policy.prepare(views)
    teacher_actions = torch.from_numpy(np.stack(actions))
    intents = torch.from_numpy(np.stack(messages))
    if not policy.communication:
        intents.zero_()
    log.write("warmup_labels", ticks=len(samples), messages=message_counts(intents.numpy()))
    hidden = None
    if policy.text_encoder is not None:
        with torch.no_grad():
            hidden = torch.cat([policy.hidden(group)[0].cpu() for group in batches(samples, args.inference_batch)])
    optimizer = torch.optim.AdamW([p for name, p in policy.named_parameters()
                                    if p.requires_grad and not name.startswith("text_encoder.")], lr=args.warmup_lr)
    warm_args = argparse.Namespace(**{**vars(args), "epochs": args.warmup_epochs})

    def backward(indices, microbatch):
        losses = Counter()
        for mini in batches(indices, microbatch):
            group = [samples[i] for i in mini]
            selected = teacher_actions[mini].to(policy.device)
            intentions = intents[mini].to(policy.device)
            _, masks = policy.commit(group, prescribed=selected)
            features = None if hidden is None else hidden[mini].to(policy.device)
            send, act, _ = policy.distributions(group, intentions, masks, hidden=features)
            loss = -(send.log_prob(intentions) + act.log_prob(selected)).sum()
            (loss / len(indices)).backward()
            losses["imitation"] += loss.item()
        return losses

    _, stats = optimize(policy, optimizer, len(samples), warm_args, rng, log, backward, "warmup")
    log.write("warmup_complete", ticks=len(samples), optimizer_steps=stats["optimizer_steps"],
              seconds=perf_counter() - started)


@torch.no_grad()
def collect(policy, matches, source, steps, args, message_rng, action_rng, teacher_rng):
    samples, prefixes, intentions, actions, masks = [], [], [], [], []
    old, values, rewards, terminals, targets, target_messages = [], [], [], [], [], []
    finished = []
    opponent_ticks = Counter()
    start = perf_counter()
    for _ in range(steps):
        batch = policy.prepare([match.view() for match in matches])
        selections = [policy.decide(group, message_rng=message_rng, action_rng=action_rng, cache=True)
                      for group in batches(batch, args.inference_batch)]
        selected = torch.cat([choice.joint for choice in selections]).cpu().tolist()
        samples.extend(batch)
        for choice in selections:
            intentions.append(choice.intent.cpu())
            actions.append(choice.actions.cpu())
            masks.append(choice.action_masks.cpu())
            old.append(choice.log_probability.cpu())
            if choice.prefixes is not None:
                prefixes.extend(choice.prefixes)
        values.append(torch.cat([choice.values for choice in selections]).cpu())
        outcomes = []
        for i, (match, joint) in enumerate(zip(matches, selected, strict=True)):
            opponent_ticks[match.opponent] += 1
            target, message = -1, np.zeros(3, dtype=np.int64)
            if args.search_imitation_weight and batch[i].joint_mask.sum() > 1 and teacher_rng.random() < 0.25:
                target, message = teacher_action(match, teacher_rng)
            targets.append(target)
            target_messages.append(message)
            outcomes.append(match.step(joint))
            if match.done:
                finished.append(match_result(match))
                matches[i] = source.next()
        reward, terminal = zip(*outcomes, strict=True)
        rewards.append(reward)
        terminals.append(terminal)
    next_samples = policy.prepare([match.view() for match in matches])
    bootstrap = policy.value(policy.spatial_features(next_samples)).squeeze(1).cpu()
    values = torch.stack(values)
    advantages = generalized_advantage(torch.tensor(rewards, dtype=torch.float32), values,
                                       torch.tensor(terminals, dtype=torch.bool), bootstrap, DISCOUNT, 0.95)
    rollout = SlotRollout(samples, prefixes or None, torch.cat(intentions), torch.cat(actions), torch.cat(masks),
                          torch.cat(old), advantages.flatten(), (advantages + values).flatten(),
                          torch.tensor(targets), torch.from_numpy(np.stack(target_messages)))
    stats = {"ticks": len(samples), "episodes": finished, "opponent_ticks": dict(opponent_ticks),
             "seconds": perf_counter() - start, "messages": message_counts(rollout.messages.numpy())}
    return rollout, stats


def update(policy, optimizer, rollout, args, rng, log):
    actionable = rollout.masks.sum(dim=-1).gt(1).any(dim=1)
    if policy.communication:
        actionable |= torch.as_tensor([bool(np.any(sample.intent_mask.sum(axis=1) > 1)) for sample in rollout.samples])
    advantages = rollout.advantages.clone()
    if actionable.any():
        values = advantages[actionable]
        advantages[actionable] = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)

    def backward(indices, microbatch):
        metrics = Counter()
        n_actions = max(1, int(actionable[indices].sum()))
        n_teacher = max(1, int(rollout.teacher_joints[indices].ge(0).sum()))
        for mini in batches(indices, microbatch):
            group = [rollout.samples[i] for i in mini]
            prefixes = [rollout.prefixes[i] for i in mini] if rollout.prefixes is not None else None
            hidden, _ = policy.hidden(group, prefixes)
            spatial = policy.spatial_features(group)
            features = policy.features(spatial, hidden)
            messages = rollout.messages[mini].to(policy.device)
            acts = rollout.actions[mini].to(policy.device)
            send = policy.intent_distribution(group, features)
            act = masked(policy.scores(group, features, messages), rollout.masks[mini].to(policy.device))
            logp = (send.log_prob(messages) + act.log_prob(acts)).sum(dim=1)
            ratio = (logp - rollout.old_log_probability[mini].to(policy.device)).exp()
            valid = actionable[mini].to(policy.device)
            actor = -clipped_objective(ratio, advantages[mini].to(policy.device)).masked_fill(~valid, 0).sum()
            entropy = ((send.entropy() + act.entropy()).sum(dim=1) * valid).sum()
            critic = (policy.value(spatial).squeeze(1) - rollout.returns[mini].to(policy.device)).square().sum()
            loss = (actor - 0.01 * entropy) / n_actions + 0.5 * critic / len(indices)
            targets = rollout.teacher_joints[mini]
            labeled = torch.where(targets >= 0)[0]
            if len(labeled):
                teacher_group = [group[i] for i in labeled]
                teacher_features = [x[labeled.to(policy.device)] for x in features]
                teacher_messages = rollout.teacher_messages[mini][labeled].to(policy.device)
                if not policy.communication:
                    teacher_messages.zero_()
                chosen = torch.from_numpy(JOINT_ACTIONS[targets[labeled].numpy()]).to(policy.device)
                _, teacher_masks = policy.commit(teacher_group, prescribed=chosen)
                teacher_send = policy.intent_distribution(teacher_group, teacher_features)
                teacher_act = masked(policy.scores(teacher_group, teacher_features, teacher_messages), teacher_masks)
                imitation = -(teacher_send.log_prob(teacher_messages) + teacher_act.log_prob(chosen)).sum()
                loss = loss + args.search_imitation_weight * imitation / n_teacher
                metrics["search_cross_entropy"] += imitation.item()
            loss.backward()
            metrics["actor_loss"] += actor.item()
            metrics["entropy"] += entropy.item()
            metrics["critic_mse"] += critic.item()
        return metrics

    sums, stats = optimize(policy, optimizer, len(rollout.samples), args, rng, log, backward, "ppo")
    stats.update({key: val / (len(rollout.samples) * args.epochs) for key, val in sums.items() if key != "gradient_norm"})
    stats["search_labels"] = int(rollout.teacher_joints.ge(0).sum())
    return stats


def train(args):
    hardware = configure_device(args.device)
    torch.manual_seed(args.seed)
    initial = load_checkpoint(args.initialize) if args.initialize is not None else None
    if initial is not None and (initial["backbone"] != args.backbone or initial["run"]["rules_prompt_version"] != 4):
        raise ValueError("expected a matching centralized v4 checkpoint")
    policy = SlotTeamPolicy(args.backbone, args.device, base=args.base,
                            initialize=None if initial is None else initial["model"],
                            communication=args.communication).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    log = RunLog(args.output / "metrics.jsonl")
    game = GameConfig(max_ticks=80)
    log.write("start", **hardware, seed=args.seed, ticks=args.ticks, warmup_ticks=args.warmup_ticks,
              communication=args.communication, base_revision=BASE_REVISION, rules_prompt_version=4,
              search_teacher=TEACHER, search_imitation_weight=args.search_imitation_weight,
              initialize=None if args.initialize is None else str(args.initialize), tactical_exercises=False,
              schedule=TRAINING_OPPONENTS, game_ticks=game.max_ticks,
              trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    if args.warmup_ticks:
        warm_up(policy, game, args, log)
    optimizer = optimizer_for(policy, args)
    message_rng = torch.Generator(device=policy.device).manual_seed(args.seed + 6)
    action_rng = torch.Generator(device=policy.device).manual_seed(args.seed + 7)
    replay_rng = np.random.default_rng(args.seed + 8)
    teacher_rng = np.random.default_rng(args.seed + 9)
    source = MatchSource(game, np.random.SeedSequence(args.seed, spawn_key=(4, 4)),
                         (TRAINING_OPPONENTS[0], "search_intro"), 10, True, weights=(0.8, 0.2),
                         terminal_margin_weight=args.terminal_margin_weight)
    matches = source.start(args.envs)
    ticks = updates = games = 0
    start = perf_counter()
    while ticks < args.ticks:
        opponent = TRAINING_OPPONENTS[min(ticks // OPPONENT_INTERVAL, len(TRAINING_OPPONENTS) - 1)]
        if source.opponents[0] != opponent:
            source.opponents = (opponent, "search_intro")
            log.write("opponents", ticks=ticks, opponents=source.opponents)
        remaining = min(args.ticks - ticks, OPPONENT_INTERVAL - ticks % OPPONENT_INTERVAL)
        active = matches if remaining >= len(matches) else matches[:remaining]
        steps = min(args.rollout_steps, remaining // len(active))
        rollout, stats = collect(policy, active, source, steps, args, message_rng, action_rng, teacher_rng)
        ticks += stats["ticks"]
        games += len(stats["episodes"])
        log.write("rollout_complete", cumulative_ticks=ticks, **stats)
        results = update(policy, optimizer, rollout, args, replay_rng, log)
        updates += results["optimizer_steps"]
        log.write("ppo_update", cumulative_ticks=ticks, **results)
        del rollout
        latest = args.output / "ppo.pt"
        checkpoint(latest, policy, game, args, {"ticks": ticks, "updates": updates, "games": games}, optimizer)
        if ticks % OPPONENT_INTERVAL == 0:
            os.link(latest, args.output / f"gate-{ticks}.pt")
    memory = {}
    if policy.device.type == "cuda":
        memory["peak_allocated_gib"] = torch.cuda.max_memory_allocated(policy.device) / 1024**3
        memory["peak_reserved_gib"] = torch.cuda.max_memory_reserved(policy.device) / 1024**3
    log.write("training_complete", ticks=ticks, optimizer_steps=updates, games=games, seconds=perf_counter() - start,
              **memory)


class SlotEvaluation:
    def __init__(self):
        self.messages = Counter()

    def __call__(self, policy, matches, inference_batch, generators=None):
        actions = []
        for group in batches(matches, 1 if generators is not None else inference_batch):
            samples = policy.prepare([match.view() for match in group])
            rng = None if generators is None else generators[group[0]]
            output = policy.decide(samples, message_rng=rng, action_rng=rng)
            self.messages.update(message_counts(output.intent.cpu().numpy()))
            actions.extend(output.joint.cpu().tolist())
        return actions


@torch.inference_mode()
def evaluate_slot_checkpoint(args):
    configure_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    policy = SlotTeamPolicy(payload["backbone"], args.device, base=args.base,
                            communication=payload["run"]["communication"]).eval()
    policy.load_trainable_weights(payload["model"])
    policy.mute_received = args.mute_received
    game = replace(GameConfig(**payload["game"]), max_ticks=96)
    match_selector, board_selector = SlotEvaluation(), SlotEvaluation()
    report = evaluate(policy, game, pairs=8, opponents=("search_tactical",), seed=640000, force_entry=True,
                      inference_batch=args.inference_batch, measure_tactics=True, action_selector=match_selector)
    report["tactical_probes"] = evaluate_exercises(policy, game, 16, True, args.inference_batch,
                                                   trace_actions=True, action_selector=board_selector)
    report["communication"] = {"enabled": policy.communication, "mute_received": policy.mute_received,
                               "match_intents": dict(match_selector.messages),
                               "probe_intents": dict(board_selector.messages)}
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Three territory policies with a shared Qwen backbone")
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train")
    evaluator = commands.add_parser("evaluate")
    for command in (trainer, evaluator):
        command.add_argument("--base", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--inference-batch", type=int, default=8)
    trainer.add_argument("--initialize", type=Path)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--backbone", choices=("bf16", "cnn"), default="bf16")
    trainer.add_argument("--communication", action=argparse.BooleanOptionalAction, default=True)
    trainer.add_argument("--seed", type=int, default=47)
    trainer.add_argument("--ticks", type=int, default=16384)
    trainer.add_argument("--warmup-ticks", type=int, default=4096)
    trainer.add_argument("--warmup-epochs", type=int, default=6)
    trainer.add_argument("--warmup-lr", type=float, default=3e-4)
    trainer.add_argument("--envs", type=int, default=16)
    trainer.add_argument("--rollout-steps", type=int, default=32)
    trainer.add_argument("--batch-size", type=int, default=128)
    trainer.add_argument("--microbatch", type=int, default=4)
    trainer.add_argument("--epochs", type=int, default=2)
    trainer.add_argument("--head-lr", type=float, default=7.5e-5)
    trainer.add_argument("--lora-lr", type=float, default=2.5e-6)
    trainer.add_argument("--terminal-margin-weight", type=float, default=0.25)
    trainer.add_argument("--search-imitation-weight", type=float, default=0.2)
    trainer.set_defaults(freeze_text_backbone=False)
    evaluator.add_argument("checkpoint", type=Path)
    evaluator.add_argument("--mute-received", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.ticks, args.envs, args.rollout_steps, args.batch_size, args.microbatch, args.epochs,
               args.inference_batch, args.warmup_epochs) < 1 or args.warmup_ticks < 0:
            parser.error("training sizes must be positive, with zero allowed for --warmup-ticks")
        train(args)
    else:
        if args.inference_batch < 1:
            parser.error("--inference-batch must be positive")
        evaluate_slot_checkpoint(args)


if __name__ == "__main__":
    main()
