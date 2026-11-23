from dataclasses import dataclass

import numpy as np

from territory_rl.agent import AgentState
from territory_rl.models import AGENT_IDS, Action

from .state import JOINT_ATTACKS


EXERCISE_ERRORS = {"enemy_entry": "enemy_entry_deaths", "tile_defense": "defensible_losses"}
EXERCISE_TICKS = 16
EXERCISE_PENALTY = 0.15


@dataclass(frozen=True)
class TacticalExercise:
    kind: str
    scenario: dict
    opening_actions: dict[str, int]
    target: tuple[int, int]


def make_exercise(game, kind, seed, team):
    """Vary the frontier layout, slot assignments, and orientation around one tactical decision."""
    size = game.width
    if game.height != size or size < 6:
        raise ValueError("the tactical exercises require a square board at least 6x6")
    if kind not in EXERCISE_ERRORS:
        raise ValueError(f"unknown tactical exercise {kind}")
    rng = np.random.default_rng(seed)
    edge = size // 2
    row = int(rng.integers(1, size - 1))
    target = (row, edge)
    battleground = np.broadcast_to(np.arange(size) >= edge, (size, size)).astype(np.int8).copy()
    far_rows = [r for r in range(size) if abs(r - row) >= 2]
    frontier_loss, frontier_gain = rng.choice(far_rows, 2, replace=False)
    battleground[frontier_loss, edge - 1] = 1
    battleground[frontier_gain, edge] = 0
    allies = [(row, edge - 1), (int(rng.integers(size)), edge - 2), ((row + 1) % size, 0)]
    if kind == "enemy_entry":
        allies[2] = (row - 1, edge)
        battleground[allies[2]] = 0
    else:
        battleground[target] = 0
        if rng.random() < 0.5:
            allies[2] = target

    attacker = (row, edge + 1)
    enemy_cells = [(r, c) for r in range(size) for c in range(edge + 1, size) if (r, c) != attacker]
    enemies = [attacker, *(enemy_cells[i] for i in rng.choice(len(enemy_cells), 2, replace=False))]
    rotations = int(rng.integers(4))
    team_number = int(team == "blue")

    def world(cell):
        r, c = cell
        for _ in range(rotations):
            r, c = size - 1 - c, r
        return r, size - 1 - c if team_number else c

    battleground = np.rot90(battleground, rotations)
    if team_number:
        battleground = 1 - battleground[:, ::-1]
    ally_ids = AGENT_IDS[3 * team_number:3 * team_number + 3]
    enemy_ids = AGENT_IDS[3 * (1 - team_number):3 * (1 - team_number) + 3]
    ally_slots = rng.permutation(3)
    enemy_slots = rng.permutation(3)
    positions = {aid: world(allies[i]) for aid, i in zip(ally_ids, ally_slots, strict=True)}
    positions.update({aid: world(enemies[i]) for aid, i in zip(enemy_ids, enemy_slots, strict=True)})
    agents = [AgentState(aid, i // 3, positions[aid]) for i, aid in enumerate(AGENT_IDS)]
    opening = dict.fromkeys(enemy_ids, Action.NOOP.index)
    target = world(target)
    if kind == "tile_defense" or rng.random() < 0.5:
        attacker_id = enemy_ids[int(np.flatnonzero(enemy_slots == 0)[0])]
        action = next(a for a in Action if a.is_attack and a.target(positions[attacker_id]) == target)
        opening[attacker_id] = action.index
    return TacticalExercise(kind, {"owner": battleground.copy(), "agents": agents}, opening, target)


def tactical_counts(state, team, events, *, force_entry, target=None):
    """Count mistakes using the actual simultaneous opponent attacks and the decision-time mask."""
    side = int(team == "blue")

    def canonical(cell):
        r, c = cell
        return r, state.board.shape[1] - 1 - c if side else c

    def applies(cell):
        return target is None or target == cell

    def is_enemy(cell):
        code = state.board[canonical(cell)]
        return code == 1 or code >= 5

    allies = AGENT_IDS[side * 3:side * 3 + 3]
    moves = [e for e in events if e.kind == "move" and e.reason == "success" and e.agent_id in allies]
    entry_deaths = sum(e.kind == "death" and e.reason == "movement" and e.agent_id in allies
                       and applies(e.position) and is_enemy(e.position) for e in events)
    attacks_on_ours = {e.position: e.counts[1 - side] for e in events
                       if e.kind == "attack" and e.counts[1 - side] and applies(e.position) and not is_enemy(e.position)}
    lost = {e.position for e in events if e.kind == "capture" and e.old_owner == side}
    defendable = set()
    if attacks_on_ours:
        mask = state.choices.mask(force_entry)
        for cell, count in attacks_on_ours.items():
            targets = (state.targets == canonical(cell)).all(axis=-1)
            if (((targets & JOINT_ATTACKS).sum(axis=1) >= count) & mask).any():
                defendable.add(cell)
    return {"successful_moves": len(moves), "enemy_entry_deaths": int(entry_deaths),
            "defensible_attacks": len(defendable), "defensible_losses": len(defendable & lost)}
