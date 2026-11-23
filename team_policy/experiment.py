from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from territory_rl.config import GameConfig
from territory_rl.models import Action

from .cooperation_scenes import summarize_scenes
from .curriculum import Curriculum, LESSONS, LESSONS_WITH_INTERMEDIATE_SEARCH
from .distillation import distillation_update, tactical_feedback, teacher_targets, update_ema
from .fixed_scenes import evaluate_fixed
from .group_learning import collect_groups, grouped_update, reference_log_probs
from .match import DISCOUNT, TEAMS, TeamMatch, entry_scenario
from .optimization import batches, clipped_objective, generalized_advantage, optimize
from .policy import BASE_ID, BASE_REVISION, TeamPolicy
from .search import SEARCH_BUDGETS, choose_search_action
from .tactics import EXERCISE_ERRORS, EXERCISE_PENALTY, EXERCISE_TICKS, make_exercise, tactical_counts
from .teacher import RuleTeacher


class RunLog:
    def __init__(self, path):
        self.path = path

    def write(self, event, **fields):
        line = json.dumps({"event": event, **fields}, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        print(line, flush=True)


class MatchSource:
    def __init__(self, game, seed, opponents, scenario_interval, force_entry, *, weights=None,
                 frozen=(), frozen_weight=0.4, terminal_margin_weight=0.0, exercise_seed=None):
        self.game = game
        self.rng = np.random.default_rng(seed)
        self.frozen = {f"frozen_{index + 1}": opponent for index, opponent in enumerate(frozen)}
        self.opponents = (*opponents, *self.frozen)
        self.weights = None
        if weights is not None or frozen:
            named_weights = np.ones(len(opponents)) if weights is None else np.array(weights)
            named_weights = named_weights / named_weights.sum()
            self.weights = named_weights if not frozen else np.concatenate(
                ((1 - frozen_weight) * named_weights, np.full(len(frozen), frozen_weight / len(frozen))))
        self.scenario_interval = scenario_interval
        self.force_entry = force_entry
        self.terminal_margin_weight = terminal_margin_weight
        self.started = 0
        self.scenarios = 0
        self.exercise_rng = np.random.default_rng(exercise_seed) if exercise_seed is not None else None
        self.exercise_started = Counter()

    def start(self, count):
        if self.exercise_rng is None:
            return [self.next() for _ in range(count)]
        per_kind = max(1, count // 8)
        cases = [None] * (count - per_kind * len(EXERCISE_ERRORS))
        cases += [kind for kind in EXERCISE_ERRORS for _ in range(per_kind)]
        return [self.next(kind) for kind in cases]

    def next(self, exercise_kind=None):
        if exercise_kind is None:
            return self.next_group(1)[0]
        team = TEAMS[self.exercise_started[exercise_kind] % 2]
        self.exercise_started[exercise_kind] += 1
        seed = int(self.exercise_rng.integers(2**31))
        opponent = str(self.exercise_rng.choice(self.opponents, p=self.weights))
        exercise = make_exercise(self.game, exercise_kind, seed, team)
        return TeamMatch(replace(self.game, max_ticks=EXERCISE_TICKS), seed, team, opponent,
                         force_entry=self.force_entry, terminal_margin_weight=self.terminal_margin_weight,
                         exercise=exercise)

    def next_group(self, replicas):
        team = TEAMS[self.started % 2]
        self.started += 1
        scenario = None
        if self.scenario_interval and self.started % self.scenario_interval == 0:
            scenario = entry_scenario(self.game, team, self.rng)
            self.scenarios += 1
        seed = int(self.rng.integers(2**31))
        opponent = str(self.rng.choice(self.opponents, p=self.weights))
        return [TeamMatch(self.game, seed, team, opponent, force_entry=self.force_entry, scenario=scenario,
                          frozen_opponent=self.frozen.get(opponent), terminal_margin_weight=self.terminal_margin_weight)
                for _ in range(replicas)]


def optimizer_for(policy, args):
    adapter, heads = [], []
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            group = adapter if name.startswith("text_encoder.") else heads
            group.append(parameter)
    groups = [{"params": heads, "lr": args.head_lr}]
    if adapter and not args.freeze_text_backbone:
        groups.append({"params": adapter, "lr": args.lora_lr})
    return torch.optim.AdamW(groups)


def restore_optimizer(optimizer, checkpoint_payload):
    current_groups = optimizer.state_dict()["param_groups"]
    saved = checkpoint_payload["optimizer"]
    saved_groups = saved["param_groups"]
    new_adapter_state = (checkpoint_payload["run"].get("freeze_text_backbone", False)
                         and len(saved_groups) == 1 and len(current_groups) == 2)
    if new_adapter_state:
        saved = {**saved, "param_groups": [saved_groups[0], current_groups[1]]}
    optimizer.load_state_dict(saved)
    for group, configured in zip(optimizer.param_groups, current_groups, strict=True):
        group["lr"] = configured["lr"]
    return new_adapter_state


def checkpoint(path, policy, game, args, metrics, optimizer):
    pending = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "model": policy.trainable_weights(), "optimizer": optimizer.state_dict(),
                "backbone": policy.backbone, "additive_only": policy.additive_only, "game": asdict(game),
                "base_id": BASE_ID if policy.backbone != "cnn" else None,
                "base_revision": BASE_REVISION if policy.backbone != "cnn" else None,
                "run": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "metrics": metrics}, pending)
    os.replace(pending, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1 or (payload["backbone"] != "cnn" and payload["base_revision"] != BASE_REVISION):
        raise ValueError("unsupported team policy checkpoint")
    return payload


class FrozenOpponent:
    def __init__(self, path):
        payload = load_checkpoint(path)
        if payload["backbone"] != "cnn":
            raise ValueError("the frozen opponent pool expects CPU CNN checkpoints")
        with torch.random.fork_rng(devices=[]):
            self.policy = TeamPolicy("cnn", "cpu", additive_only=payload["additive_only"]).eval()
        self.policy.load_trainable_weights(payload["model"])
        self.force_entry = payload["run"]["force_entry"]

    @torch.inference_mode()
    def __call__(self, state, rng):
        view = (state, (), state.choices.mask(self.force_entry))
        dist = self.policy.actor(self.policy.prepare([view]))
        u = float(rng.random())
        probabilities = dist.probs[0].numpy().astype(np.float64)
        legal = np.flatnonzero(view[2])
        cumulative = probabilities[legal].cumsum()
        return int(legal[cumulative.searchsorted(u * cumulative[-1], side="right")])


@torch.no_grad()
def collect_teacher_data(policy, game, args, log):
    source = MatchSource(game, args.seed + 1, args.opponents, 5, args.force_entry, weights=args.opponent_weights)
    teacher = RuleTeacher(args.seed + 2, temperature=args.teacher_temperature)
    behavior_rng = np.random.default_rng(args.seed + 6)
    num_envs = min(args.warmup_envs, args.warmup_ticks)
    matches = [(source.next(), behavior_rng.random() < args.student_warmup_fraction) for _ in range(num_envs)]
    student_ticks = 0
    samples, targets, encodings = [], [], []
    start = perf_counter()
    while len(samples) < args.warmup_ticks:
        active = matches[:args.warmup_ticks - len(samples)]
        prepared = policy.prepare([match.view() for match, _ in active])
        predictions = {}
        for indices in batches([i for i, (_, student) in enumerate(active) if student], args.inference_batch):
            batch = [prepared[i] for i in indices]
            features = policy.text_hidden(batch) if policy.text_encoder is not None else None
            actions = policy.actor(batch, cached_text=features).logits.argmax(dim=-1).tolist()
            rows = [None] * len(batch) if features is None else list(features.cpu().unbind())
            predictions.update(zip(indices, zip(actions, rows, strict=True), strict=True))
        for index, (match, student) in enumerate(active):
            target = teacher.distribution(match)
            action, encoding = predictions[index] if student else (int(target.argmax()), None)
            samples.append(prepared[index])
            targets.append(target)
            encodings.append(encoding)
            student_ticks += int(student)
            _, done = match.step(action)
            if done:
                matches[index] = source.next(), behavior_rng.random() < args.student_warmup_fraction
        if len(samples) % 1024 < len(active):
            log.write("teacher_progress", ticks=len(samples), student_ticks=student_ticks, seconds=perf_counter() - start)
    log.write("teacher_rollout", ticks=len(samples), replays=teacher.replays, scenarios=source.scenarios,
              student_ticks=student_ticks, seconds=perf_counter() - start)
    cached = None
    if policy.text_encoder is not None:
        missing = [i for i, row in enumerate(encodings) if row is None]
        for indices in batches(missing, args.inference_batch):
            rows = policy.text_hidden([samples[i] for i in indices]).cpu().unbind()
            for index, row in zip(indices, rows, strict=True):
                encodings[index] = row
        cached = torch.stack(encodings)
    return samples, torch.from_numpy(np.stack(targets)), cached, teacher.replays, source.scenarios, student_ticks


def teacher_warmup(policy, game, args, log):
    start = perf_counter()
    samples, targets, cached, replays, scenarios, student_ticks = collect_teacher_data(policy, game, args, log)
    optimizer = optimizer_for(policy, args)
    rng = np.random.default_rng(args.seed + 3)
    losses = []
    for epoch in range(args.warmup_epochs):
        epoch_loss = 0.0
        for indices in batches(rng.permutation(len(samples)), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            for mini in batches(indices, args.microbatch):
                dist = policy.actor([samples[i] for i in mini], cached_text=None if cached is None else cached[mini])
                target = targets[mini].to(policy.device)
                log_probs = dist.logits.masked_fill(target == 0, 0.0)
                loss = -(target * log_probs).sum(dim=1).sum()
                (loss / len(indices)).backward()
                epoch_loss += loss.item()
            clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        losses.append(epoch_loss / len(samples))
        log.write("warmup_epoch", epoch=epoch + 1, cross_entropy=losses[-1])
    return {"ticks": len(samples), "teacher_replays": replays, "scenarios": scenarios,
            "student_ticks": student_ticks, "cross_entropy": losses, "seconds": perf_counter() - start}, optimizer


@dataclass
class Rollout:
    inputs: list
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    feedback: list | None = None
    cached_text: torch.Tensor | None = None
    search_targets: torch.Tensor | None = None


@torch.no_grad()
def collect(policy, matches, source, steps, args, log, prior_ticks, feedback_rng=None, search_rng=None):
    inputs, actions, log_probs, values, rewards, terminals = [], [], [], [], [], []
    stats = Counter()
    text_features = [] if args.freeze_text_backbone else None
    search_targets = [] if search_rng is not None else None
    search_teacher = getattr(args, "search_teacher", None) or "search_strong"
    search_budget = SEARCH_BUDGETS[search_teacher] if search_rng is not None else None
    opponent_ticks = Counter()
    opponent_opportunities = Counter()
    exercise_ticks, exercise_openings, exercise_mistakes = Counter(), Counter(), Counter()
    finished = []
    feedback = [] if feedback_rng is not None else None
    start = perf_counter()
    for step in range(steps):
        samples = policy.prepare([match.view() for match in matches])
        selected, lp, estimations = [], [], []
        for chunk in batches(samples, args.inference_batch):
            features = policy.text_hidden(chunk) if text_features is not None else None
            distribution, value = policy.distribution(chunk, cached_text=features)
            if features is not None:
                text_features.append(features.cpu())
            action = distribution.sample()
            selected.append(action.cpu())
            lp.append(distribution.log_prob(action).cpu())
            estimations.append(value.cpu())
        action = torch.cat(selected)
        results = []
        for index, (match, joint) in enumerate(zip(matches, action.tolist(), strict=True)):
            opportunity = int(match.state.choices.certified.any())
            opponent_ticks[match.opponent] += 1
            opponent_opportunities[match.opponent] += opportunity
            stats["entry_opportunities"] += opportunity
            stats["entry_successes"] += int(match.state.choices.certified[joint])
            before = match.env.core.snapshot() if feedback is not None else None
            state = match.state
            if search_targets is not None:
                label = -1
                if samples[index].mask.sum() > 1 and search_rng.random() < 0.25:
                    label = choose_search_action(state, search_rng, search_budget, force_entry=match.force_entry,
                                                 respawn_delay=match.env.core.config.respawn_delay)
                search_targets.append(label)
            reward, done = match.step(joint)
            if match.exercise is not None:
                kind = match.exercise.kind
                exercise_ticks[kind] += 1
                if match.ticks == 1:
                    counts = tactical_counts(state, match.team, match.env.last_result.events,
                                             force_entry=match.force_entry, target=match.exercise.target)
                    mistakes = counts[EXERCISE_ERRORS[kind]]
                    reward -= EXERCISE_PENALTY * mistakes
                    exercise_openings[kind] += 1
                    exercise_mistakes[kind] += mistakes
            results.append((reward, done))
            if feedback is not None:
                feedback.append(tactical_feedback(match, before, state, samples[index].mask, joint, feedback_rng))
            if match.done:
                finished.append(match_result(match))
                matches[index] = source.next(match.exercise.kind if match.exercise is not None else None)
        reward, terminal = zip(*results, strict=True)
        inputs.extend(samples)
        actions.append(action)
        log_probs.append(torch.cat(lp))
        values.append(torch.cat(estimations))
        rewards.append(reward)
        terminals.append(terminal)
        if (step + 1) % 32 == 0:
            log.write("rollout_progress", ticks=prior_ticks + len(inputs), episodes=len(finished),
                      seconds=perf_counter() - start)
    next_inputs = policy.prepare([match.view() for match in matches])
    bootstrap = torch.cat([policy(chunk)[1].cpu() for chunk in batches(next_inputs, args.inference_batch)])
    values = torch.stack(values)
    advantages = generalized_advantage(torch.tensor(rewards, dtype=torch.float32), values,
                                       torch.tensor(terminals, dtype=torch.bool), bootstrap, DISCOUNT, 0.95)
    seconds = perf_counter() - start
    metrics = {"ticks": len(inputs), "seconds": seconds, "ticks_per_second": len(inputs) / seconds,
               "episodes": finished, "entry_opportunities": stats["entry_opportunities"],
               "entry_successes": stats["entry_successes"],
               "opponent_ticks": dict(opponent_ticks), "opponent_entry_opportunities": dict(opponent_opportunities)}
    if exercise_ticks:
        metrics.update(exercise_ticks=dict(exercise_ticks), exercise_openings=dict(exercise_openings),
                       exercise_mistakes=dict(exercise_mistakes))
    if policy.tokenizer is not None:
        lengths = [len(sample.tokens) for sample in inputs]
        metrics.update(mean_tokens=float(np.mean(lengths)), max_tokens=max(lengths))
    rollout = Rollout(inputs, torch.cat(actions), torch.cat(log_probs), values.flatten(), advantages.flatten(),
                      (advantages + values).flatten(), feedback,
                      None if text_features is None else torch.cat(text_features),
                      None if search_targets is None else torch.tensor(search_targets, dtype=torch.long))
    return rollout, metrics


def ppo_backward(policy, rollout, indices, microbatch, actionable, advantages, imitation_weight):
    sums = Counter()
    n_actions = max(1, int(actionable[indices].sum()))
    n_targets = 1 if rollout.search_targets is None else max(1, int((rollout.search_targets[indices] >= 0).sum()))
    for mini in batches(indices, microbatch):
        cached = None if rollout.cached_text is None else rollout.cached_text[mini]
        distribution, values = policy.distribution([rollout.inputs[i] for i in mini], cached_text=cached)
        actions = rollout.actions[mini].to(policy.device)
        old_log_probs = rollout.log_probs[mini].to(policy.device)
        returns = rollout.returns[mini].to(policy.device)
        valid = actionable[mini].to(policy.device)
        advantage = advantages[mini].to(policy.device)
        log_ratio = distribution.log_prob(actions) - old_log_probs
        ratio = log_ratio.exp()
        objective = clipped_objective(ratio, advantage)
        actor = -(objective * valid).sum()
        entropy = (distribution.entropy() * valid).sum()
        critic = (values - returns).square().sum()
        loss = (actor - 0.01 * entropy) / n_actions + 0.5 * critic / len(indices)
        if rollout.search_targets is not None:
            labels = rollout.search_targets[mini].to(policy.device)
            imitation = -distribution.log_prob(labels.clamp_min(0)).masked_fill(labels < 0, 0.0).sum()
            loss = loss + imitation_weight * imitation / n_targets
            sums["search_imitation"] += imitation.detach().item()
        loss.backward()
        sums["actor"] += actor.detach().item()
        sums["critic"] += critic.detach().item()
        sums["entropy"] += entropy.detach().item()
        sums["kl"] += ((ratio.detach() - 1 - log_ratio.detach()) * valid).sum().item()
        sums["clips"] += ((ratio.detach().sub(1).abs() > 0.2) & valid).sum().item()
    return sums


def ppo_update(policy, optimizer, rollout, args, rng, log, *, lora_scheduler=None):
    actionable = torch.tensor([sample.mask.sum() > 1 for sample in rollout.inputs], dtype=torch.bool)
    advantages = rollout.advantages.clone()
    advantages[~actionable] = 0
    if actionable.any():
        group = advantages[actionable]
        advantages[actionable] = (group - group.mean()) / (group.std(unbiased=False) + 1e-8)
    def backward(indices, microbatch):
        return ppo_backward(policy, rollout, indices, microbatch, actionable, advantages, args.search_imitation_weight)

    sums, stats = optimize(policy, optimizer, len(rollout.inputs), args, rng, log, backward, "ppo",
                           lora_scheduler=lora_scheduler)
    n_policy = max(1, int(actionable.sum()) * args.epochs)
    stats.update(actor_loss=sums["actor"] / n_policy,
                 critic_mse=sums["critic"] / (len(rollout.inputs) * args.epochs),
                 entropy=sums["entropy"] / n_policy, approx_kl=sums["kl"] / n_policy,
                 clip_fraction=sums["clips"] / n_policy, actionable_ticks=int(actionable.sum()))
    if rollout.search_targets is not None:
        labels = int((rollout.search_targets >= 0).sum())
        stats.update(search_labels=labels,
                     search_cross_entropy=sums["search_imitation"] / (max(1, labels) * args.epochs))
    return stats


def match_result(match):
    result = {"seed": match.seed, "team": match.team, "opponent": match.opponent, "ticks": match.ticks,
              "gap": match.gap, "entry_opportunities": match.entry_opportunities,
              "entry_choices": match.entry_choices, "entry_successes": match.entry_successes}
    if match.exercise is not None:
        result["exercise"] = match.exercise.kind
    return result


def summarize_matches(matches):
    results = [match_result(match) for match in matches]
    wins = sum(result["gap"] > 0 for result in results)
    draws = sum(result["gap"] == 0 for result in results)
    counts = Counter()
    for match in matches:
        counts.update(match.scenes.counts)
    summary = {"games": len(results), "wins": wins, "draws": draws, "losses": len(results) - wins - draws,
               "score": (wins + 0.5 * draws) / len(results),
               "mean_gap": sum(result["gap"] for result in results) / len(results),
               "entry_opportunities": sum(result["entry_opportunities"] for result in results),
               "entry_successes": sum(result["entry_successes"] for result in results),
               "scenes": summarize_scenes(counts)}
    return summary, results


@torch.inference_mode()
def evaluation_actions(policy, matches, inference_batch, generators=None):
    actions = []
    for group in batches(matches, inference_batch):
        distribution = policy.actor(policy.prepare([match.view() for match in group]))
        if generators is None:
            selected = distribution.logits.argmax(dim=-1)
        else:
            selected = torch.cat([torch.multinomial(probs, 1, generator=generators[match])
                                  for match, probs in zip(group, distribution.probs, strict=True)])
        actions.extend(selected.cpu().tolist())
    return actions


def territory_frame(match):
    events = match.env.last_result.events
    captures = Counter(e.new_owner for e in events if e.kind == "capture")
    attacks = Counter(e.agent_id for e in events if e.kind == "attack")
    deaths = Counter((e.reason, e.agent_id) for e in events if e.kind == "death")
    teams = match.allies, match.enemies
    return {"tick": match.ticks, "areas": match.state.areas,
            "captures": [captures[match.team_number], captures[1 - match.team_number]],
            "attacks": [sum(attacks[aid] for aid in aids) for aids in teams],
            "attack_deaths": [sum(deaths["attack", aid] for aid in aids) for aids in teams],
            "movement_deaths": [sum(deaths["movement", aid] for aid in aids) for aids in teams]}


def evaluate(policy, game, *, pairs, opponents, seed, force_entry, inference_batch, measure_tactics=False,
             record_timeline=False, sample_seed=None):
    start = perf_counter()
    fixtures = [TeamMatch(game, seed + offset, team, opponent, force_entry=force_entry, track_scenes=True)
                for opponent in opponents for offset in range(pairs) for team in TEAMS]
    generators = None
    if sample_seed is not None:
        generators = {}
        for match in fixtures:
            stream = np.random.SeedSequence(sample_seed, spawn_key=(match.seed, match.team_number))
            generators[match] = torch.Generator(device=policy.device).manual_seed(
                int(stream.generate_state(1, dtype=np.uint64)[0]))
    active = fixtures
    errors = {match: Counter() for match in fixtures} if measure_tactics else None
    timeline = ({match: {"seed": match.seed, "team": match.team, "opponent": match.opponent,
                         "initial_areas": match.state.areas, "ticks": []} for match in fixtures}
                if record_timeline else None)
    while active:
        actions = evaluation_actions(policy, active, inference_batch, generators)
        for match, joint in zip(active, actions, strict=True):
            state = match.state
            match.step(joint)
            if errors is not None:
                errors[match].update(tactical_counts(state, match.team, match.env.last_result.events,
                                                    force_entry=force_entry))
            if timeline is not None:
                timeline[match]["ticks"].append(territory_frame(match))
        active = [match for match in active if not match.done]
    overall, results = summarize_matches(fixtures)
    fixed = []
    if sample_seed is None and min(game.height, game.width) >= 6:
        scenarios = [TeamMatch(game, seed, team, "aggressive", force_entry=entry, track_scenes=True,
                               scenario=entry_scenario(game, team)) for entry in (True, False) for team in TEAMS]
        for match, joint in zip(scenarios, evaluation_actions(policy, scenarios, inference_batch), strict=True):
            match.step(joint)
            fixed.append({**match_result(match), "force_entry": match.force_entry,
                          "capture_entry": summarize_scenes(match.scenes.counts)["capture_entry"]})
    report = {"overall": overall,
              "by_opponent": {name: summarize_matches([match for match in fixtures if match.opponent == name])[0]
                              for name in opponents},
              "matches": results, "fixed_entry": fixed, "seconds": perf_counter() - start}
    if errors is not None:
        totals = Counter()
        for counts in errors.values():
            totals.update(counts)
        report["tactical_errors"] = {"overall": dict(totals),
                                     "matches": [{"seed": m.seed, "team": m.team, "opponent": m.opponent, **errors[m]}
                                                 for m in fixtures]}
    if timeline is not None:
        report["timeline"] = list(timeline.values())
    return report


def evaluate_exercises(policy, game, pairs, force_entry, inference_batch, *, trace_actions=False):
    lessons = []
    for kind in EXERCISE_ERRORS:
        for seed in range(650000, 650000 + pairs):
            for team in TEAMS:
                exercise = make_exercise(game, kind, seed, team)
                lessons.append(TeamMatch(replace(game, max_ticks=EXERCISE_TICKS), seed, team, "search_intermediate",
                                         force_entry=force_entry, exercise=exercise))
    cases = []
    for match, joint in zip(lessons, evaluation_actions(policy, lessons, inference_batch), strict=True):
        state = match.state
        start = match.env.core.snapshot() if trace_actions else None
        match.step(joint)
        exercise = match.exercise
        transition = match.env.last_result
        counts = tactical_counts(state, match.team, transition.events,
                                 force_entry=force_entry, target=exercise.target)
        case = {"kind": exercise.kind, "seed": match.seed, "team": match.team, "target": exercise.target,
                "error": counts[EXERCISE_ERRORS[exercise.kind]], "gap": match.gap}
        if start is not None:
            case["trace"] = {"joint": joint, "start_owner": start.owner.tolist(),
                             "start_positions": {agent.agent_id: agent.position for agent in start.agents},
                             "actions": {aid: Action.from_index(index).name for aid, index in match.last_actions.items()},
                             "events": [asdict(event) for event in transition.events],
                             "final_owner": match.env.core.snapshot().owner.tolist()}
        cases.append(case)
    return {"cases": cases, "by_kind": {kind: {"games": 2 * pairs,
                                              "errors": sum(c["error"] for c in cases if c["kind"] == kind)}
                                       for kind in EXERCISE_ERRORS}}


def configure_device(device):
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(2)
        return {"device": "cpu"}
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("choose cpu or an available CUDA device")
    device = torch.device("cuda", device.index if device.index is not None else 0)
    torch.cuda.set_device(device)
    gib = 1024**3
    free, total = torch.cuda.mem_get_info(device)
    if free <= 2 * gib:
        raise ValueError("CUDA has insufficient free memory for the run")
    torch.cuda.set_per_process_memory_fraction((free - 2 * gib) / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    return {"device": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
            "free_vram_gib": free / gib, "total_vram_gib": total / gib}


def probe_sdpo(policy, game, args, ema, log):
    rng = np.random.default_rng(args.seed + 8)
    views, feedback = [], []
    for team in TEAMS:
        match = TeamMatch(game, args.seed, team, "aggressive", force_entry=False, scenario=entry_scenario(game, team, rng))
        view = match.view()
        views.append(view)
        before = match.env.core.snapshot()
        match.step(0)
        feedback.append(tactical_feedback(match, before, view[0], view[2], 0, rng))
    _, _, metrics = teacher_targets(policy, policy.prepare(views), feedback, ema, args.inference_batch)
    log.write("feedback_probe", scenes=len(views), **metrics)


def train(args):
    game = GameConfig()
    grouped = args.algorithm in ("grpo", "dapo")
    max_group_ticks = args.group_size * game.max_ticks
    if grouped and 0 < args.ticks < max_group_ticks:
        raise ValueError(f"--ticks must admit at least one complete group ({max_group_ticks} ticks)")
    initial = load_checkpoint(args.initialize) if args.initialize is not None else None
    saved_run = initial["run"] if initial is not None else {}
    if args.rules_prompt is None:
        args.rules_prompt = saved_run.get("rules_prompt", False)
    if args.rules_prompt_version is None:
        args.rules_prompt_version = saved_run.get("rules_prompt_version", 1)
    if getattr(args, "search_teacher", None) is None:
        args.search_teacher = saved_run.get("search_teacher", "search_strong")
    args.output.mkdir(parents=True, exist_ok=False)
    log = RunLog(args.output / "metrics.jsonl")
    hardware = configure_device(args.device)
    torch.manual_seed(args.seed)
    start = perf_counter()
    log.write("start", **hardware, torch=torch.__version__, backbone=args.backbone, seed=args.seed, algorithm=args.algorithm,
              game=asdict(game), opponents=args.opponents, force_entry=args.force_entry, rules_prompt=args.rules_prompt,
              rules_prompt_version=args.rules_prompt_version,
              ticks=args.ticks, warmup_ticks=args.warmup_ticks, warmup_envs=args.warmup_envs, envs=args.envs,
              inference_batch=args.inference_batch, batch_size=args.batch_size, microbatch=args.microbatch,
              epochs=args.epochs, base_revision=BASE_REVISION if args.backbone != "cnn" else None,
              opponent_weights=args.opponent_weights, frozen_opponents=args.frozen_opponents,
              frozen_weight=args.frozen_weight if args.frozen_opponents else 0,
              head_lr=args.head_lr, lora_lr=args.lora_lr, teacher_temperature=args.teacher_temperature,
              student_warmup_fraction=args.student_warmup_fraction, activation_checkpointing=args.activation_checkpointing,
              group_size=args.group_size if grouped else None, terminal_margin_weight=args.terminal_margin_weight,
              freeze_text_backbone=args.freeze_text_backbone, curriculum=args.curriculum,
              curriculum_fine_search=args.curriculum_fine_search,
              curriculum_fine_tactical=args.curriculum_fine_tactical,
              curriculum_leaf_budget=args.curriculum_leaf_budget,
              curriculum_tactical_exploration=args.curriculum_tactical_exploration,
              curriculum_exploration_schedule=args.curriculum_exploration_schedule,
              tactical_exercises=args.tactical_exercises, tactical_metrics=args.tactical_metrics,
              lora_lr_warmup_steps=args.lora_lr_warmup_steps,
              search_imitation_weight=args.search_imitation_weight, search_teacher=args.search_teacher)
    policy = TeamPolicy(args.backbone, args.device, base=args.base, additive_only=args.additive_only,
                        rules_prompt=args.rules_prompt, rules_prompt_version=args.rules_prompt_version).eval()
    log.write("model_loaded", seconds=perf_counter() - start,
              trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    if initial is not None:
        if initial["backbone"] != args.backbone or initial["additive_only"] != args.additive_only:
            raise ValueError("initial checkpoint architecture does not match the requested policy")
        policy.load_trainable_weights(initial["model"])
        log.write("initialized", checkpoint=str(args.initialize))
    if args.warmup_ticks:
        warmup, warm_optimizer = teacher_warmup(policy, game, args, log)
        checkpoint(args.output / "warmup.pt", policy, game, args, warmup, warm_optimizer)
        del warm_optimizer
        log.write("warmup_complete", **warmup)
    if args.eval_pairs:
        log.write("evaluation", stage=f"pre_{args.algorithm}", **evaluate(
            policy, game, pairs=args.eval_pairs, opponents=args.eval_opponents, seed=args.eval_seed,
            force_entry=args.force_entry, inference_batch=args.inference_batch, measure_tactics=args.tactical_metrics))

    before_rl = policy.trainable_weights()
    optimizer = optimizer_for(policy, args)
    if args.resume_optimizer:
        if initial["run"].get("algorithm", "ppo") != args.algorithm or "updates" not in initial["metrics"]:
            raise ValueError("optimizer resume requires an online checkpoint for the same algorithm")
        new_adapter_state = restore_optimizer(optimizer, initial)
        log.write("optimizer_restored", checkpoint=str(args.initialize), checkpoint_updates=initial["metrics"]["updates"],
                  new_adapter_state=new_adapter_state)
    lora_scheduler = None
    if args.lora_lr_warmup_steps:
        warmup_steps = args.lora_lr_warmup_steps
        lora_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [lambda step: 1.0, lambda step: min((step + 1) / warmup_steps, 1.0)],
        )
    frozen = [FrozenOpponent(path) for path in args.frozen_opponents]
    lessons = LESSONS_WITH_INTERMEDIATE_SEARCH if args.curriculum_intermediate_search else LESSONS
    curriculum = (Curriculum(args.curriculum_start, args.curriculum_interval, args.curriculum_target, lessons,
                             fine_search=args.curriculum_fine_search, fine_tactical=args.curriculum_fine_tactical,
                             leaf_budget=args.curriculum_leaf_budget,
                             tactical_exploration=args.curriculum_tactical_exploration,
                             exploration_schedule=args.curriculum_exploration_schedule or ())
                  if args.curriculum else None)
    tactical_start_seed = args.curriculum_seed + max(6000, args.curriculum_pairs)
    tactical_check_seed = tactical_start_seed + max(100, args.curriculum_pairs)
    tactical_event_prefix = "curriculum_exploration" if args.curriculum_tactical_exploration else "curriculum_leaf"

    def tactical_probe(opponents, seed):
        return evaluate(policy, curriculum.lesson.game(game), pairs=args.curriculum_pairs, opponents=opponents,
                        seed=seed, force_entry=args.force_entry, inference_batch=args.inference_batch,
                        sample_seed=args.seed)

    def make_source():
        if curriculum is None:
            return MatchSource(game, args.seed + 4, args.opponents, 10, args.force_entry, weights=args.opponent_weights,
                               frozen=frozen, frozen_weight=args.frozen_weight,
                               terminal_margin_weight=args.terminal_margin_weight)
        lesson = curriculum.lesson
        if curriculum.tactical_schedule:
            log.write("curriculum_exploration_schedule", level=curriculum.level, interval=curriculum.interval,
                      opponents=curriculum.tactical_schedule)
        elif curriculum.uses_adaptive_tactical and args.ticks:
            calibration = tactical_probe(curriculum.calibration_opponents, tactical_start_seed)
            curriculum.select_tactical_level({opponent: row["score"] for opponent, row in calibration["by_opponent"].items()})
            log.write(f"{tactical_event_prefix}_calibration", level=curriculum.level, seed=tactical_start_seed,
                      sample_seed=args.seed, selected=curriculum.tactical_opponent, **calibration)
        opponents, weights = curriculum.opponents()
        seed = np.random.SeedSequence(args.seed, spawn_key=(4, curriculum.level))
        use_exercises = args.tactical_exercises and lesson.opponent == "search_intermediate"
        exercise_seed = np.random.SeedSequence(args.seed, spawn_key=(11, curriculum.level)) if use_exercises else None
        log.write("curriculum_lesson", level=curriculum.level, opponent=lesson.opponent, max_ticks=lesson.max_ticks,
                  opponents=opponents, weights=weights, paired_validation_games=2 * args.curriculum_pairs,
                  target_score=curriculum.target, tactical_exercises=use_exercises)
        return MatchSource(lesson.game(game), seed, opponents, 10, args.force_entry,
                           weights=weights, terminal_margin_weight=args.terminal_margin_weight, exercise_seed=exercise_seed)

    source = make_source()
    matches = [] if grouped else source.start(args.envs)
    ema = {name: tensor.clone() for name, tensor in before_rl.items()} if args.algorithm == "sdpo" else None
    feedback_rng = np.random.default_rng(args.seed + 7) if ema is not None else None
    search_rng = np.random.default_rng(args.seed + 9) if args.search_imitation_weight else None
    if ema is not None and args.ticks:
        probe_sdpo(policy, game, args, ema, log)
    rng = np.random.default_rng(args.seed + 5)
    ticks, iteration = 0, 0
    completed_games, old_scenarios = 0, 0
    lesson_games = Counter()
    training_start = perf_counter()
    collection_time, update_time, auxiliary_time, validation_time, updates = 0.0, 0.0, 0.0, 0.0, 0
    while ticks < args.ticks:
        remaining = args.ticks - ticks
        if curriculum is not None:
            remaining = min(remaining, curriculum.next_check - curriculum.stage_ticks)
        if grouped:
            if remaining < max_group_ticks:
                break
            rollout, collection = collect_groups(policy, source, args, remaining, log, ticks)
        else:
            active = matches if remaining >= len(matches) else matches[:remaining]
            steps = min(args.rollout_steps, remaining // len(active))
            rollout, collection = collect(policy, active, source, steps, args, log, ticks, feedback_rng, search_rng)
        ticks += collection["ticks"]
        iteration += 1
        collection_time += collection["seconds"]
        if grouped:
            completed_games += sum(len(group["lengths"]) for group in collection["groups"])
        else:
            completed_games += len(collection["episodes"])
        if curriculum is not None:
            curriculum.stage_ticks += collection["ticks"]
            lesson_games[curriculum.level] += len(collection["episodes"])
            collection.update(curriculum_level=curriculum.level, lesson_ticks=curriculum.stage_ticks)
        log.write("rollout_complete", iteration=iteration, cumulative_ticks=ticks, **collection)
        skip_grpo = args.algorithm == "grpo" and not updates and not collection.get("signal_ticks", 0)
        reference = None
        if args.algorithm == "grpo" and rollout.inputs and not skip_grpo:
            if updates == 0:
                reference = rollout.log_probs
                log.write("reference_reused", iteration=iteration, ticks=len(rollout.inputs))
            else:
                reference_start = perf_counter()
                reference = reference_log_probs(policy, rollout, before_rl, args.inference_batch)
                reference_time = perf_counter() - reference_start
                auxiliary_time += reference_time
                log.write("reference_forward", iteration=iteration, ticks=len(rollout.inputs), seconds=reference_time)
        if ema is not None:
            sdpo_inputs, sdpo_targets, teacher_metrics = teacher_targets(policy, rollout.inputs, rollout.feedback, ema,
                                                                         args.inference_batch)
            auxiliary_time += teacher_metrics["teacher_seconds"]
            log.write("teacher_forward", iteration=iteration, **teacher_metrics)
        activations = policy.checkpoint_activations() if args.activation_checkpointing else nullcontext()
        with activations:
            if skip_grpo:
                update = {"optimizer_steps": 0, "seconds": 0.0, "actionable_ticks": len(rollout.inputs),
                          "reason": "zero_return_signal_and_unchanged_reference"}
            elif grouped:
                update = grouped_update(policy, optimizer, rollout, args, rng, log, reference)
            elif ema is not None:
                update = distillation_update(policy, optimizer, sdpo_inputs, sdpo_targets, args, rng, log)
            else:
                update = ppo_update(policy, optimizer, rollout, args, rng, log, lora_scheduler=lora_scheduler)
        if ema is not None and update["optimizer_steps"]:
            update_ema(ema, policy)
        update_time += update["seconds"]
        updates += update["optimizer_steps"]
        log.write(f"{args.algorithm}_update", iteration=iteration, cumulative_ticks=ticks, **update)
        path = args.output / f"{args.algorithm}.pt"
        metrics = {"ticks": ticks, "updates": updates}
        if curriculum is not None:
            metrics.update(curriculum_level=curriculum.level, lesson_ticks=curriculum.stage_ticks,
                           completed_games=completed_games)
        checkpoint(path, policy, game, args, metrics, optimizer)
        if args.keep_snapshots:
            os.link(path, args.output / f"{args.algorithm}-{ticks}.pt")
        if curriculum is not None and curriculum.check_due():
            lesson = curriculum.lesson
            if curriculum.uses_adaptive_tactical:
                profile = curriculum.tactical_opponent
                check_number = curriculum.stage_ticks // curriculum.interval - 1
                seed = tactical_check_seed + check_number * args.curriculum_pairs
                probe = tactical_probe((profile,), seed)
                validation_time += probe["seconds"]
                if ticks < args.ticks:
                    curriculum.adjust_tactical_level(probe["overall"]["score"])
                log.write(f"{tactical_event_prefix}_check", ticks=ticks, level=curriculum.level, seed=seed,
                          sample_seed=args.seed, opponent=profile, next_opponent=curriculum.tactical_opponent, **probe)
            result = evaluate(policy, lesson.game(game), pairs=args.curriculum_pairs, opponents=(lesson.opponent,),
                              seed=args.curriculum_seed, force_entry=args.force_entry, inference_batch=args.inference_batch,
                              measure_tactics=args.tactical_metrics)
            validation_time += result["seconds"]
            level = curriculum.level
            score = result["overall"]["score"]
            if curriculum.uses_adaptive_tactical:
                result["qualified"] = score >= curriculum.target
            promoted = curriculum.check(score)
            if args.keep_curriculum_checkpoints:
                os.link(path, args.output / f"gate-{ticks}.pt")
            log.write("curriculum_check", ticks=ticks, level=level, opponent=lesson.opponent, max_ticks=lesson.max_ticks,
                      promoted=promoted, next_level=curriculum.level, target_score=curriculum.target, **result)
            if promoted:
                os.link(path, args.output / f"lesson-{level}.pt")
                if ticks < args.ticks:
                    old_scenarios += source.scenarios
                    partial = sum(match.ticks > 0 for match in matches)
                    log.write("curriculum_reset", ticks=ticks, unfinished_games=partial)
                    source = make_source()
                    matches = source.start(args.envs)
            elif (args.curriculum_fine_search or args.curriculum_fine_tactical or curriculum.uses_adaptive_tactical) and ticks < args.ticks:
                opponents, weights = curriculum.opponents()
                if opponents != source.opponents:
                    source.opponents = opponents
                    log.write("curriculum_training_opponents", ticks=ticks, level=level, opponents=opponents,
                              weights=weights)

    changes = Counter()
    for name, parameter in policy.trainable_weights().items():
        group = "adapter" if name.startswith("text_encoder.") else "action_head" if name.startswith("action_head.") else "encoder_value"
        changes[group] += (parameter - before_rl[name]).float().square().sum().item()
    gpu_memory = {}
    if policy.device.type == "cuda":
        gib = 1024**3
        gpu_memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated(policy.device) / gib,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(policy.device) / gib}
    elapsed = perf_counter() - training_start
    log.write("training_complete", ticks=ticks, optimizer_steps=updates, rollout_seconds=collection_time,
              update_seconds=update_time, auxiliary_forward_seconds=auxiliary_time, seconds=elapsed,
              ticks_per_second=ticks / elapsed, unused_ticks=args.ticks - ticks, scenarios=old_scenarios + source.scenarios,
              completed_games=completed_games, curriculum_validation_seconds=validation_time,
              lesson_games=dict(lesson_games), curriculum_level=curriculum.level if curriculum is not None else None,
              parameter_delta_l2={name: math.sqrt(value) for name, value in changes.items()}, **gpu_memory)
    if args.eval_pairs:
        log.write("evaluation", stage=f"post_{args.algorithm}", **evaluate(
            policy, game, pairs=args.eval_pairs, opponents=args.eval_opponents, seed=args.eval_seed,
            force_entry=args.force_entry, inference_batch=args.inference_batch, measure_tactics=args.tactical_metrics))
    log.write("complete", seconds=perf_counter() - start)


def evaluate_checkpoint(args):
    configure_device(args.device)
    payload = load_checkpoint(args.checkpoint)
    rules_prompt = payload["run"].get("rules_prompt", False) if args.rules_prompt is None else args.rules_prompt
    version = payload["run"].get("rules_prompt_version", 1) if args.rules_prompt_version is None else args.rules_prompt_version
    policy = TeamPolicy(payload["backbone"], args.device, base=args.base, additive_only=payload["additive_only"],
                        rules_prompt=rules_prompt, rules_prompt_version=version,
                        explicit_map_prompt=args.explicit_map_prompt).eval()
    policy.load_trainable_weights(payload["model"])
    game = GameConfig(**payload["game"])
    if args.eval_max_ticks is not None:
        game = replace(game, max_ticks=args.eval_max_ticks)
    report = {}
    if args.eval_pairs:
        report = evaluate(policy, game, pairs=args.eval_pairs, opponents=args.eval_opponents, seed=args.eval_seed,
                          force_entry=args.force_entry, inference_batch=args.inference_batch, measure_tactics=args.tactical_metrics,
                          record_timeline=args.eval_timeline)
    if args.tactical_probe_pairs:
        report["tactical_probes"] = evaluate_exercises(policy, game, args.tactical_probe_pairs,
                                                       args.force_entry, args.inference_batch,
                                                       trace_actions=args.tactical_probe_actions)
    if args.scene_pairs:
        def choose(matches):
            return evaluation_actions(policy, matches, args.inference_batch)

        report["fixed_scenes"] = evaluate_fixed(game, args.scene_split, args.scene_pairs, args.eval_seed,
                                                args.force_entry, choose)
    print(json.dumps(report, allow_nan=False), flush=True)
