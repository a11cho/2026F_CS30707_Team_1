"""Paired, one-tick tactical evaluation of an existing feedback PPO checkpoint."""

import argparse
from dataclasses import asdict, dataclass
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
import torch

from territory_rl.agent import AgentState
from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.models import AGENT_IDS, Action

from .policy import action_masks, encode_proposals, encode_states, policy_from_checkpoint
from .rollout import FeedbackMatch, collect_tick, current_states


@dataclass(frozen=True)
class Scenario:
    kind: str
    row: int
    team: str
    layout: str
    roles: tuple[int, int, int]
    tick: int
    approach: str = "flank"
    cooldown: bool = False


@dataclass(frozen=True)
class PreparedScenario:
    match: FeedbackMatch
    target: tuple[int, int]
    starts: tuple[tuple[int, int], ...]
    protected_agent: str | None


@dataclass(frozen=True)
class FixedAction:
    action: Action

    def act(self, observation):
        if not observation["action_mask"][self.action.index]:
            raise RuntimeError("scenario opponent was assigned a masked action")
        return self.action


def scenarios(game, kind):
    if game.height < 5 or game.width < 6 or game.max_ticks < 2:
        raise ValueError("tactical layouts need at least 5 rows, 6 columns, and 2 ticks")
    ticks = sorted({max(1, tick) for tick in (1, game.max_ticks // 2, game.max_ticks * 7 // 8)})
    return [
        Scenario(kind, row, team, layout, roles, tick, approach, cooldown)
        for row, team, layout, roles, tick, approach, cooldown in product(
            range(2, game.height - 2), ("red", "blue"), ("flat", "inset"), permutations(range(3)), ticks,
            ("flank", "rear"), (False, True),
        )
    ]


def prepare(scenario, game, experiment, seed):
    row = scenario.row
    column = game.width // 2 - 1
    own = int(scenario.team == "blue")
    target = (row, column)
    opponent_front = (row, column + 1)
    cover = scenario.kind == "cover"
    north = (row - 1, column)
    other = (row + 1, column) if scenario.approach == "flank" else (row, column - 1)
    positions = (target, north, other) if cover else (north, other, (row + 2, column - 2))
    opponents = (opponent_front, (row - 2, column + 1), (row + 2, column + 1))
    owner = np.full((game.height, game.width), 1 - own, dtype=np.int8)
    owner[:, :column + 1] = own
    if scenario.layout == "inset":
        owner[row - 1, column + 1] = own
        owner[row + 1, column + 1] = own
    if own:
        owner = owner[:, ::-1].copy()

    def orient(position):
        r, c = position
        return (r, game.width - 1 - c) if own else (r, c)

    starts = tuple(orient(positions[role]) for role in scenario.roles)
    them = "red" if own else "blue"
    learner_ids = [f"{scenario.team}_{slot}" for slot in range(3)]
    opponent_ids = [f"{them}_{slot}" for slot in range(3)]
    placements = dict(zip(learner_ids + opponent_ids, (*starts, *map(orient, opponents)), strict=True))
    protected = learner_ids[scenario.roles.index(0)] if cover else None
    cooling = set() if not scenario.cooldown else ({protected} if cover else set(learner_ids))
    agents = [
        AgentState(aid, index // 3, placements[aid], last_attack_tick=scenario.tick if aid in cooling else -2)
        for index, aid in enumerate(AGENT_IDS)
    ]
    match = FeedbackMatch(game, experiment)
    match.reset(seed, scenario.team, "random", options={"owner": owner, "agents": agents, "tick": scenario.tick})
    match.bots = dict.fromkeys(opponent_ids, FixedAction(Action.NOOP))
    if cover:
        match.bots[opponent_ids[0]] = FixedAction(Action.ATTACK_RIGHT if own else Action.ATTACK_LEFT)
    return PreparedScenario(match, orient(target), starts, protected)


def outcome(prepared):
    match = prepared.match
    target = prepared.target
    protected = prepared.protected_agent
    result = match.env.last_result
    if protected is None:
        target_moves = [event for event in result.events if event.kind == "move"
                        and event.position == target and event.agent_id in match.learners]
        return {
            "collision": any(event.reason == "contested" for event in target_moves),
            "reached_target": any(event.reason == "success" for event in target_moves),
        }

    attacks = [event for event in result.events if event.kind == "attack" and event.position == target]
    if sum(event.agent_id not in match.learners for event in attacks) != 1:
        raise RuntimeError("cover scenario must expose exactly one attack on the protected tile")
    own = int(match.team == "blue")
    snapshot = match.env.core.snapshot()
    held = snapshot.owner[target] == own
    survived = not any(event.kind == "death" and event.agent_id == protected for event in result.events)
    return {
        "defended": bool(held and survived),
        "held_tile": bool(held),
        "ally_counterattacked": any(event.agent_id in match.learners for event in attacks),
    }


def is_converging(prepared, proposals):
    actions = (Action.from_index(index) for index in proposals)
    return sum(
        action.is_move and action.target(position) == prepared.target
        for action, position in zip(actions, prepared.starts, strict=True)
    ) >= 2


@torch.no_grad()
def choose_actions(policy, cases, *, hide_teammates, force_proposals):
    matches = [case.match for case in cases]
    if not force_proposals:
        tick = collect_tick(policy, matches, hide_teammates=hide_teammates, greedy=True)
        return tick.proposal.reshape(-1, 3).tolist(), tick.action.reshape(-1, 3).tolist()

    first = current_states(matches)
    latent = policy.encode(encode_states(first))
    masks = action_masks(first)
    proposals = policy.proposal(latent, masks).probs.argmax(dim=-1).reshape(-1, 3).tolist()
    masks = masks.reshape(-1, 3, len(Action))
    second = []
    for i, (case, indices) in enumerate(zip(cases, proposals, strict=True)):
        directions = [
            action for action in Action
            if (action.is_move if case.protected_agent is None else action.is_attack)
        ]
        for slot, position in enumerate(case.starts):
            for direction in directions:
                if direction.target(position) == case.target:
                    if not masks[i, slot, direction.index]:
                        raise RuntimeError("forced proposal is masked by the scenario")
                    indices[slot] = direction.index
                    break
        second.extend(case.match.begin(indices))

    messages, known = encode_proposals(second, hide_teammates)
    actions = policy.action(latent, messages, known, action_masks(second)).probs.argmax(dim=-1)
    actions = actions.reshape(-1, 3).tolist()
    for case, indices in zip(cases, actions, strict=True):
        case.match.finish(indices)
    return proposals, actions


def summarize(pairs):
    summary = {
        "paired_cases": len(pairs),
        "cases_with_changed_actions": sum(pair["final_on"] != pair["final_off"] for pair in pairs),
    }
    if not pairs:
        return summary
    for name in pairs[0]["with_messages"]:
        on = sum(pair["with_messages"][name] for pair in pairs)
        off = sum(pair["without_messages"][name] for pair in pairs)
        summary[name] = {
            "on_count": on,
            "off_count": off,
            "on_rate": on / len(pairs),
            "off_rate": off / len(pairs),
            "on_minus_off_pp": 100 * (on - off) / len(pairs),
            "on_only": sum(pair["with_messages"][name] and not pair["without_messages"][name] for pair in pairs),
            "off_only": sum(pair["without_messages"][name] and not pair["with_messages"][name] for pair in pairs),
        }
    return summary


def paired_evaluation(policy, game, experiment, kind, seed, *, force_proposals=False):
    variants = scenarios(game, kind)
    pairs = []
    for offset in range(0, len(variants), 32):
        chunk = variants[offset:offset + 32]
        seeded = list(enumerate(chunk, seed + offset))
        on = [prepare(scenario, game, experiment, case_seed) for case_seed, scenario in seeded]
        off = [prepare(scenario, game, experiment, case_seed) for case_seed, scenario in seeded]
        proposals, on_actions = choose_actions(policy, on, hide_teammates=False, force_proposals=force_proposals)
        off_proposals, off_actions = choose_actions(policy, off, hide_teammates=True, force_proposals=force_proposals)
        if proposals != off_proposals:
            raise RuntimeError("a paired test must start with the same proposals")
        for i, (case_seed, scenario) in enumerate(seeded):
            pair = {
                "scenario": asdict(scenario),
                "seed": case_seed,
                "target": on[i].target,
                "starts": on[i].starts,
                "protected_agent": on[i].protected_agent,
                "proposals": [Action.from_index(index).name for index in proposals[i]],
                "final_on": [Action.from_index(index).name for index in on_actions[i]],
                "final_off": [Action.from_index(index).name for index in off_actions[i]],
                "with_messages": outcome(on[i]),
                "without_messages": outcome(off[i]),
            }
            if kind == "collision":
                pair["converging_proposals"] = is_converging(on[i], proposals[i])
            pairs.append(pair)

    stats = summarize(pairs)
    by_setup = {
        f"{approach}_{'cooldown' if cooldown else 'ready'}": summarize([
            pair for pair in pairs
            if pair["scenario"]["approach"] == approach and pair["scenario"]["cooldown"] == cooldown
        ])
        for approach, cooldown in product(("flank", "rear"), (False, True))
    }
    if kind == "collision":
        stats["when_proposals_already_converged"] = summarize(
            [pair for pair in pairs if pair["converging_proposals"]]
        )
    return {"summary": stats, "by_setup": by_setup, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="path to a new paired evaluation JSON")
    parser.add_argument("--seed", type=int, default=300_000)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    game = GameConfig(**checkpoint["game"])
    experiment = ExperimentConfig(**checkpoint["experiment"])
    if experiment.decision_strategy != "feedback" or checkpoint["training"]["hide_teammates"]:
        parser.error("provide a feedback PPO checkpoint trained with teammate messages enabled")
    policy = policy_from_checkpoint(checkpoint)
    comparisons = {
        kind: {
            "learned_proposals": paired_evaluation(policy, game, experiment, kind, args.seed),
            "forced_proposals": paired_evaluation(policy, game, experiment, kind, args.seed, force_proposals=True),
        }
        for kind in ("collision", "cover")
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "training_seed": checkpoint["training"]["seed"],
        "update": checkpoint["update"],
        "decision": "greedy",
        "ticks_per_case": 1,
        "game": asdict(game),
        **comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        kind: {mode: comparison["summary"] for mode, comparison in modes.items()}
        for kind, modes in comparisons.items()
    }, indent=2))


if __name__ == "__main__":
    main()
