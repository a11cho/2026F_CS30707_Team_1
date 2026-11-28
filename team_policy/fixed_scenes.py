from dataclasses import dataclass
import hashlib
from itertools import permutations

import numpy as np

from territory_rl.agent import AgentState
from territory_rl.models import AGENT_IDS, Action

from .cooperation_scenes import SCENES
from .match import TEAMS, TeamMatch
from .state import REFLECT_ACTION, joint_index


A = Action
HOLD = (A.NOOP,) * 3
VERTICAL = np.array([0, 2, 1, 3, 4, 6, 5, 7, 8])
ORDERS = tuple(permutations(range(3)))
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Template:
    allies: tuple
    enemies: tuple
    patches: tuple
    team_actions: tuple
    enemy_actions: tuple
    remove: tuple[int, int]


TEMPLATES = {
    "breakthrough": Template(
        ((3, 3), (2, 4), (4, 4)), ((3, 5), (0, 6), (6, 6)), (((2, 4), 0), ((4, 4), 0)),
        ((A.ATTACK_RIGHT, A.ATTACK_DOWN, A.UP),), ((A.ATTACK_LEFT, A.NOOP, A.NOOP),), (0, 1),
    ),
    "cover": Template(
        ((3, 3), (2, 3), (5, 1)), ((3, 4), (0, 6), (6, 6)), (),
        ((A.NOOP, A.ATTACK_DOWN, A.NOOP),), ((A.ATTACK_LEFT, A.NOOP, A.NOOP),), (0, 1),
    ),
    "capture_entry": Template(
        ((3, 3), (2, 4), (3, 2)), ((4, 3), (1, 6), (6, 5)), (((2, 4), 0), ((4, 3), 1)),
        ((A.RIGHT, A.ATTACK_DOWN, A.ATTACK_RIGHT),), ((A.ATTACK_UP, A.NOOP, A.NOOP),), (0, 1),
    ),
    "yield_move": Template(
        ((3, 3), (3, 2), (6, 1)), ((2, 5), (3, 5), (6, 6)), (),
        ((A.UP, A.RIGHT, A.NOOP),), (HOLD,), (0, 0),
    ),
    "split_targets": Template(
        ((3, 3), (2, 4), (5, 1)), ((3, 5), (2, 6), (6, 6)), (((2, 4), 0),),
        ((A.ATTACK_RIGHT, A.ATTACK_RIGHT, A.NOOP),), (HOLD,), (0, 1),
    ),
    "cooldown_rotation": Template(
        ((3, 3), (4, 3), (5, 1)), ((1, 6), (3, 6), (6, 6)), (),
        ((A.ATTACK_RIGHT, A.NOOP, A.NOOP), (A.UP, A.ATTACK_RIGHT, A.NOOP)),
        (HOLD, HOLD), (1, 1),
    ),
    "role_switch": Template(
        ((3, 3), (2, 3), (4, 3)), ((2, 4), (5, 3), (6, 6)), (((1, 3), 1), ((5, 3), 1)),
        ((A.ATTACK_UP, A.ATTACK_UP, A.ATTACK_RIGHT), (A.DOWN, A.NOOP, A.RIGHT),
         (A.ATTACK_DOWN, A.ATTACK_RIGHT, A.ATTACK_LEFT)),
        ((A.ATTACK_LEFT, A.NOOP, A.NOOP), HOLD, (A.NOOP, A.ATTACK_UP, A.NOOP)), (2, 2),
    ),
}


@dataclass(frozen=True)
class SceneCase:
    name: str
    seed: int
    team: str
    owner: np.ndarray
    agents: tuple[AgentState, ...]
    oracle: tuple[int, ...]
    interventions: tuple[int, ...]
    opponent: tuple[tuple[int, ...], ...]

    def initial_hash(self):
        positions = np.asarray([agent.position for agent in self.agents], dtype="<i2")
        return hashlib.sha256(self.owner.tobytes() + positions.tobytes()).hexdigest()

    def match(self, game, force_entry):
        return TeamMatch(game, self.seed, self.team, "random", force_entry=force_entry, track_scenes=True,
                         scenario={"owner": self.owner, "agents": self.agents})

    def step(self, match, action):
        commands = {}
        for aid, index in zip(match.enemies, self.opponent[match.ticks], strict=True):
            mask = match.observations[aid]["auxiliary"]["action_mask"]
            commands[aid] = index if mask[index] else A.NOOP.index
        match.step(action, opponent_actions=commands)


def make_case(name, game, seed, team, template, decoy, vertical, ally_order, enemy_order):
    owner = np.broadcast_to(np.arange(game.width) >= game.width // 2, (game.height, game.width)).astype(np.int8).copy()
    for cell, side in template.patches:
        owner[cell] = side
    allies = [template.allies[index] for index in ally_order]
    enemies = [(*template.enemies[:2], decoy)[index] for index in enemy_order]
    allies_plan = [[turn[index].index for index in ally_order] for turn in template.team_actions]
    enemies_plan = [[turn[index].index for index in enemy_order] for turn in template.enemy_actions]
    if vertical:
        owner = owner[::-1].copy()
        allies = [(game.height - 1 - row, col) for row, col in allies]
        enemies = [(game.height - 1 - row, col) for row, col in enemies]
        allies_plan = VERTICAL[allies_plan].tolist()
        enemies_plan = VERTICAL[enemies_plan].tolist()
    interventions = [turn.copy() for turn in allies_plan]
    tick, role = template.remove
    interventions[tick][ally_order.index(role)] = A.NOOP.index
    positions = allies + enemies
    if team == "blue":
        owner = 1 - owner[:, ::-1]
        positions = [(row, game.width - 1 - col) for row, col in enemies + allies]
        enemies_plan = REFLECT_ACTION[enemies_plan].tolist()
    agents = tuple(AgentState(aid, index // 3, cell) for index, (aid, cell) in enumerate(zip(AGENT_IDS, positions, strict=True)))
    return SceneCase(name, seed, team, owner, agents, tuple(map(joint_index, allies_plan)),
                     tuple(map(joint_index, interventions)), tuple(tuple(turn) for turn in enemies_plan))


def oracle_passes(case, game, force_entry, actions):
    match = case.match(game, force_entry)
    for action in actions:
        if match.done or not match.view()[2][action]:
            break
        case.step(match, action)
    return match.scenes.counts[(case.name, "count")] > 0


def fixed_cases(game, split, pairs, seed):
    if (game.height, game.width) != (8, 8):
        raise ValueError("the seven fixed scenes use an 8x8 board")
    split_id = SPLITS.index(split)
    cases = []
    for scene_id, (name, template) in enumerate(TEMPLATES.items()):
        decoys = [(row, col) for row in range(game.height) for col in (6, 7)
                  if (row, col) not in (*template.allies, *template.enemies[:2])]
        geometries = [(p, vertical) for vertical in (False, True) for p in decoys]
        rng = np.random.default_rng(np.random.SeedSequence(0, spawn_key=(scene_id,)))
        # Geometry is partitioned before role-slot permutations. Both colors stay in the same partition.
        indices = rng.permutation(len(geometries))[split_id::len(SPLITS)]
        variants = [(geometries[i], allies, enemies) for i in indices for allies in ORDERS for enemies in ORDERS]
        choose = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(scene_id, split_id)))
        count = 0
        for i in choose.permutation(len(variants)):
            (decoy, vertical), ally_order, enemy_order = variants[i]
            pair = [make_case(name, game, seed + count, team, template, decoy, vertical, ally_order, enemy_order)
                    for team in TEAMS]
            if all(oracle_passes(case, game, True, case.oracle)
                   and not oracle_passes(case, game, False, case.interventions) for case in pair):
                cases.extend(pair)
                count += 1
                if count == pairs:
                    break
        if count < pairs:
            raise ValueError(f"{name} has only {count} verified pairs in the {split} split")
    return cases


def evaluate_fixed(game, split, pairs, seed, force_entry, choose):
    cases = fixed_cases(game, split, pairs, seed)
    active = [(case, case.match(game, force_entry)) for case in cases]
    matches = active.copy()
    while active:
        for (case, match), action in zip(active, choose([m for _, m in active]), strict=True):
            case.step(match, action)
        active = [(case, match) for case, match in active if not match.done and match.ticks < len(case.oracle)]
    results = []
    for case, match in matches:
        counts = match.scenes.counts
        results.append({"scene": case.name, "seed": case.seed, "team": case.team,
                        "initial_hash": case.initial_hash(), "success": bool(counts[(case.name, "count")])})
    scenes = {}
    for name in SCENES:
        successes = sum(row["success"] for row in results if row["scene"] == name)
        scenes[name] = {"successes": successes, "pairs": pairs, "cases": 2 * pairs, "rate": successes / (2 * pairs)}
    return {"split": split, "scenes": scenes, "mean_rate": sum(row["rate"] for row in scenes.values()) / len(scenes),
            "cases": results}
