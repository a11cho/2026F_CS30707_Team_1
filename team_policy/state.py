from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, permutations

import numpy as np

from territory_rl.models import Action


CANVAS = (8, 10)
ACTION_COUNT = len(Action)
JOINT_ACTIONS = np.indices((ACTION_COUNT,) * 3, dtype=np.int64).reshape(3, -1).T.copy()
REFLECT_ACTION = np.array([
    Action.NOOP.index, Action.UP.index, Action.DOWN.index, Action.RIGHT.index, Action.LEFT.index,
    Action.ATTACK_UP.index, Action.ATTACK_DOWN.index, Action.ATTACK_RIGHT.index, Action.ATTACK_LEFT.index,
])
STEPS = np.array([(0, 0) if action == Action.NOOP else action.target((0, 0)) for action in Action])
JOINT_MOVES = (JOINT_ACTIONS != Action.NOOP.index) & (JOINT_ACTIONS < Action.ATTACK_UP.index)
JOINT_ATTACKS = JOINT_ACTIONS >= Action.ATTACK_UP.index

RULES_AND_COVER = (
    "Choose one legal joint action for allies a, b, c.\n"
    "Own more tiles than the enemy at the end.\n\n"
    "A ready agent can attack one orthogonally adjacent tile. Count valid attacks from both teams on each tile. "
    "The team with more attacks takes the tile. A tie keeps its current owner. An allied attack on our tile "
    "counts as defense. One allied attack can hold against one enemy attack.\n\n"
    "Attacks resolve before movement. An ally on a tile captured by the enemy dies before moving. Enter an enemy "
    "tile only if the joint action can capture it in that tick. Use nearby enemies and their ready flags to "
    "estimate threats. When possible, assign a nearby ready ally to match the attacks on a threatened ally's "
    "starting tile while that ally moves or attacks.\n"
    "Ready flags are in abcxyz order.\n"
)

RULES_AND_COVER_V2 = (
    "Choose one legal joint action for a, b, c. A standard 8x8 match starts with our tiles on the left, "
    "enemy tiles on the right, and three agents per team near the frontier. Captures reshape that frontier. "
    "Read the current tiles and area totals. At the time limit, the team with more tiles wins and equal "
    "areas draw. A team with no tiles and no surviving agents is eliminated.\n\n"
    "Each ready agent can attack one orthogonally adjacent tile. Count valid attacks from both teams per "
    "tile. The higher count takes ownership. A tie keeps the current owner. Attacks resolve before movement. "
    "An ally whose starting tile is captured dies before moving. A survivor that moves onto a tile still "
    "owned by the enemy dies there.\n\n"
    "To save a threatened ally, assign a nearby ready teammate to attack the ally's current tile. That adds "
    "one defensive attack even if the threatened ally is about to move. Estimate how many enemies can hit "
    "the tile from their positions and ready flags. Match that count to hold our tile. If an enemy is likely "
    "to contest an enemy tile we need, concentrate enough allies on that same tile to outnumber the enemy. "
    "Against one enemy attack, one ally holds our tile and two allies can take an enemy tile. Keep enough "
    "coverage at threatened starting tiles before assigning extra attacks elsewhere. Pair movement into an "
    "enemy tile with enough attacks to capture it in the same tick.\n"
    "Ready flags are in abcxyz order.\n"
)

RULES_AND_COVER_V3 = (
    "Choose one joint action for a, b, c. Both teams choose one hold, move, or attack per agent from the same starting board. "
    "Each tile belongs to one team and there are no internal obstacles. A standard 8x8 game starts with 32 tiles "
    "per team and three agents per side in mirrored positions along the frontier. The letters a, b, c occupy our tiles "
    "and x, y, z occupy enemy tiles.\n\n"
    "Movement and attacks target one orthogonally adjacent tile inside the board. Each attack requires a ready agent "
    "and prevents it from attacking on the next tick, even on a friendly tile or in a tie. All valid attacks count "
    "simultaneously, including those by agents killed that tick. On each tile the team with more attacks takes ownership. "
    "A tie keeps the current owner. Capturing an occupied enemy tile kills the enemy before it can move. "
    "The same rule applies when enemies capture a tile under an ally.\n\n"
    "After attacks, surviving agents move simultaneously. Moves to the same destination all fail, as do two-agent swaps. "
    "An agent can follow into a tile that another agent successfully vacates. Longer move cycles are allowed. "
    "A successful move onto a tile still owned by the enemy kills the mover.\n\n"
    "At the end of a nonterminal tick, eligible fallen agents respawn on random empty friendly tiles. "
    "Earlier due agents have priority, with ties randomized. With the standard zero delay they can respawn that tick "
    "and act on the next one. They wait if no friendly tile is empty. Respawning keeps the attack cooldown. "
    "An absent agent can only hold. Ready bits and respawn wait ticks are in abcxyz order.\n\n"
    "Each team's area share is its owned tiles divided by all tiles and updates every tick. Area lead is our share minus "
    "the enemy share. Team RL uses the discounted change in this lead for reward shaping, with zero potential at "
    "termination. The match ends at the tick limit or if a team has no tiles and no survivors. More final area wins "
    "and equal areas draw. The team RL terminal outcome is +1 for a win, 0 for a draw, and -1 for a loss, "
    "with an optional final lead term.\n\n"
    "To save a threatened ally, attack its current tile with a nearby ready teammate even if it plans to move. "
    "Estimate enemy attacks from positions and ready bits. Match opposing attacks to hold our tile and outnumber them "
    "to take theirs. Against one enemy attack, one ally can hold our tile and two allies can take theirs. "
    "Focus attacks on an occupied or contested target when it helps win. Keep enough cover at threatened starting tiles "
    "and pair entries into enemy tiles with captures on the same tick.\n"
    "History lists action digits in abc order and changed cells as zero-based row and column with old and new symbols.\n"
)
RULES_AND_COVER_V4 = (
    "Choose one joint action for a,b,c. Both teams choose hold, move, or attack per agent from the same tick-start board. "
    "All tiles are team-owned with no obstacles. Standard 8x8 starts with 32 tiles each and three agents per team "
    "mirrored at the frontier. abc occupy our tiles and xyz occupy theirs.\n\n"
    "Move or attack one in-bounds orthogonal neighbor. Attacks need readiness and block next-tick attacks, "
    "even on friendly tiles or ties. Count all valid attacks before movement, including from agents killed that tick. "
    "More attacks capture the tile. Ties keep ownership. Capturing an occupied tile kills the occupant before movement.\n\n"
    "Survivors move simultaneously. Shared destinations and two-agent swaps fail. Following a successful departure "
    "and longer cycles are allowed. Landing on enemy territory kills the mover.\n\n"
    "On nonterminal ticks, due agents respawn after movement on random empty friendly tiles, earliest due first, "
    "with random ties. Delay 0 by default permits same-tick respawn and next-tick action. They wait if no tile is free. "
    "Cooldown persists. Absent agents hold. Ready bits and respawn waits follow abcxyz order.\n\n"
    "Each tick, area share is owned tiles divided by total tiles. Lead is our share minus theirs. Team RL shapes reward "
    "with discounted lead change and zero terminal potential. End at the tick limit or when either team has no tiles "
    "and no survivors. More final tiles wins and equal areas draw. Win, draw, and loss give +1, 0, and -1 in terminal "
    "reward, plus an optional final lead term.\n\n"
    "Estimate enemy attacks from positions and readiness. Cover a threatened ally's starting tile with a ready "
    "neighbor's attack even if the ally moves. Match attacks to hold and outnumber them to capture. Against one enemy "
    "attack, use one ally to hold and two to take. Keep cover before focusing extra attacks on valuable occupied or "
    "contested enemy tiles. Pair entry with same-tick capture.\n"
    "History gives abc action digits and changed cells as zero-based row, column, old, and new symbols.\n"
)
RULES_PROMPTS = {1: RULES_AND_COVER, 2: RULES_AND_COVER_V2, 3: RULES_AND_COVER_V3, 4: RULES_AND_COVER_V4}


def joint_index(indices):
    return int(np.ravel_multi_index(tuple(indices), (ACTION_COUNT,) * 3))


@dataclass(frozen=True)
class JointChoices:
    legal: np.ndarray
    entry_movers: np.ndarray

    @cached_property
    def certified(self):
        return self.entry_movers.any(axis=1)

    def mask(self, force_entry):
        return self.certified if force_entry and self.certified.any() else self.legal


@dataclass(frozen=True)
class TeamState:
    board: np.ndarray
    positions: np.ndarray
    ready: np.ndarray
    waits: np.ndarray
    action_masks: np.ndarray
    tick: int
    max_ticks: int
    areas: tuple[int, int]

    @classmethod
    def from_observations(cls, observations, encoding, team):
        """Read the public full-map window with environment rotation fixed to zero."""
        window = observations[0]["tilemap"]
        rows, columns = np.nonzero(window != encoding.outside)
        crop = window[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        if team == "blue":
            crop = crop[:, ::-1]
        board = np.empty(crop.shape, dtype=np.int8)
        positions = np.full((6, 2), -1, dtype=np.int64)
        for code, tile in enumerate((encoding.friendly, encoding.enemy, *encoding.friendly_agents,
                                     *encoding.enemy_agents)):
            cells = crop == tile
            board[cells] = code
            if code >= 2 and cells.any():
                positions[code - 2] = np.argwhere(cells)[0]
        auxiliary = observations[0]["auxiliary"]
        masks = np.stack([obs["auxiliary"]["action_mask"] for obs in observations]).astype(bool)
        if team == "blue":
            masks = masks[:, REFLECT_ACTION]
        return cls(board, positions, auxiliary["agent_attack_ready"].astype(bool),
                   auxiliary["agent_respawn_wait"], masks, auxiliary["tick"], auxiliary["max_ticks"],
                   tuple(map(int, auxiliary["areas"])))

    @cached_property
    def targets(self):
        return self.positions[None, :3, :] + STEPS[JOINT_ACTIONS]

    @cached_property
    def choices(self):
        return joint_choices(self)

    def spatial_input(self):
        height, width = self.board.shape
        padded_height, padded_width = CANVAS
        if height > padded_height or width > padded_width:
            raise ValueError(f"board {self.board.shape} exceeds the spatial canvas {CANVAS}")
        board = np.zeros((9, padded_height, padded_width), dtype=np.float32)
        area = board[:, :height, :width]
        area[0] = (self.board == 0) | ((2 <= self.board) & (self.board <= 4))
        area[1] = ~area[0].astype(bool)
        area[2:8] = self.board == np.arange(2, 8)[:, None, None]
        area[8] = 1
        features = np.concatenate((self.positions[:, 0] >= 0, self.ready,
                                   self.waits / (1 + self.waits), [1 - self.tick / self.max_ticks]))
        return board, features.astype(np.float32)

    def text_input(self, history, *, rules_prompt=False, rules_prompt_version=1, explicit_map_prompt=False):
        signs = "REabcxyz"
        board_rows = ["".join(signs[cell] for cell in row) for row in self.board]
        board = "\n".join(board_rows)
        past_states = [state for state, _ in history] + [self]
        changes = []
        for time, (_, action) in enumerate(history):
            before, after = past_states[time:time + 2]
            cells = " ".join(f"{row}{col}{signs[before.board[row, col]]}{signs[after.board[row, col]]}"
                             for row, col in np.argwhere(before.board != after.board))
            moves = "".join(map(str, JOINT_ACTIONS[action]))
            changes.append(f"{before.tick} {moves} {cells}".rstrip())
        ready = "".join(map(str, self.ready.astype(int)))
        waits = " ".join(map(str, self.waits))
        history_text = "\n".join(changes)
        height, width = self.board.shape
        if explicit_map_prompt:
            columns = " ".join(map(str, range(width)))
            rows = "\n".join(f"r{row} {' '.join(cells)}" for row, cells in enumerate(board_rows))
            agents = " ".join(f"{slot}=({row},{col})" if row >= 0 else f"{slot}=absent"
                              for slot, (row, col) in zip("abcxyz", self.positions, strict=True))
            board = (f"Current map in team-relative zero-based coordinates\nColumns {columns}\n"
                     f"{rows}\nAgent positions (row,column) {agents}")
        rules = RULES_PROMPTS[rules_prompt_version] if rules_prompt else ""
        area = f"Area {self.areas[0]}/{self.areas[1]}. "
        if rules_prompt and rules_prompt_version >= 3:
            ours, enemy = (tiles / self.board.size for tiles in self.areas)
            area += (f"Area shares {ours:.2%} ours, {enemy:.2%} enemy. "
                     f"Lead {100 * (ours - enemy):+.2f} percentage points. ")
        return (
            f"{height}x{width} territory. R ours, E enemy, abc allies, xyz enemies.\n{rules}{board}\n"
            f"Tick {self.tick}/{self.max_ticks}. {area}"
            f"Attack ready {ready}. Respawn wait {waits}.\n"
            f"Past tick, our actions, changed cells: {history_text}\n"
            "Actions 0 hold, 1-4 move up/down/left/right, 5-8 attack up/down/left/right.\nTeam action:"
        )


def joint_choices(state):
    legal = state.action_masks[np.arange(3), JOINT_ACTIONS].all(axis=1)
    targets = state.targets
    starts = state.positions[:3]
    for first, second in combinations(range(3), 2):
        same_cell = (targets[:, first] == targets[:, second]).all(axis=-1)
        swap = ((targets[:, first] == starts[second]).all(axis=-1)
                & (targets[:, second] == starts[first]).all(axis=-1))
        legal &= ~(JOINT_MOVES[:, first] & JOINT_MOVES[:, second] & (same_cell | swap))

    enemies = state.positions[3:]
    alive = enemies[:, 0] >= 0
    distance_to_origins = np.abs(enemies[:, None] - starts).sum(axis=-1)
    threats = ((distance_to_origins == 1) & state.ready[3:, None]).sum(axis=0)
    at_origins = (targets[:, :, None] == starts[None, None, :]).all(axis=-1)
    protected = (at_origins & JOINT_ATTACKS[:, :, None]).sum(axis=1) >= threats

    safe_cells = state.board == 1
    for row, col in enemies[alive]:
        neighbors = np.array([row, col]) + STEPS[1:5]
        for r, c in neighbors:
            if 0 <= r < safe_cells.shape[0] and 0 <= c < safe_cells.shape[1]:
                safe_cells[r, c] = False
    # Out-of-bounds candidates are already excluded by the individual masks.
    rows = targets[:, :, 0].clip(0, safe_cells.shape[0] - 1)
    columns = targets[:, :, 1].clip(0, safe_cells.shape[1] - 1)
    safe_destination = safe_cells[rows, columns]
    entry_movers = np.zeros_like(JOINT_MOVES)
    for mover, capturer in permutations(range(3), 2):
        at_target = (targets == targets[:, mover, None]).all(axis=-1)
        unique_attacker = (at_target & JOINT_ATTACKS).sum(axis=1) == 1
        entry_movers[:, mover] |= (
            legal & JOINT_MOVES[:, mover] & safe_destination[:, mover] & JOINT_ATTACKS[:, capturer]
            & at_target[:, capturer] & unique_attacker & protected[:, mover] & protected[:, capturer]
        )
    return JointChoices(legal, entry_movers)
