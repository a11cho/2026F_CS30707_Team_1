from collections import deque

import numpy as np

from territory_rl.agent import AgentState
from territory_rl.config import ExperimentConfig
from territory_rl.controller import FrontierController, NoopController, RandomController
from territory_rl.models import AGENT_IDS, Action
from territory_rl.parallel_env import TerritoryParallelEnv

from .cooperation_scenes import SceneTick, SceneTracker
from .search import SEARCH_BUDGETS, choose_search_action
from .state import JOINT_ACTIONS, REFLECT_ACTION, TeamState
from .teacher import cooperative_joint


SCRIPTED = ("random", "expand", "aggressive", "defend")
OPPONENTS = (*SCRIPTED, "noop", "cooperative", "cooperative_holdout", *SEARCH_BUDGETS)
TEAMS = ("red", "blue")
DISCOUNT = 0.995


class TeamMatch:
    def __init__(self, game, seed, team, opponent, *, force_entry=True, track_scenes=False, scenario=None,
                 frozen_opponent=None, terminal_margin_weight=0.0, exercise=None):
        if opponent not in OPPONENTS and frozen_opponent is None:
            raise ValueError(f"unknown opponent {opponent}")
        observation_size = 2 * max(game.height, game.width) - 1
        self.env = TerritoryParallelEnv(game, ExperimentConfig(observation_window_size=observation_size))
        self.seed, self.team, self.opponent = seed, team, opponent
        self.team_number = TEAMS.index(team)
        self.allies = AGENT_IDS[3 * self.team_number:3 * self.team_number + 3]
        self.enemies = tuple(aid for aid in AGENT_IDS if aid not in self.allies)
        self.force_entry = force_entry
        self.scenes = SceneTracker() if track_scenes else None
        self.history = deque(maxlen=4)
        self.exercise = exercise
        if exercise is not None:
            scenario = exercise.scenario
        self.observations, _ = self.env.reset(seed=seed, options=scenario)
        self.state = self._state()
        seeds = np.random.SeedSequence(seed, spawn_key=(1,)).generate_state(3)
        self.bots = None
        if opponent == "noop":
            self.bots = {aid: NoopController() for aid in self.enemies}
        elif opponent in SCRIPTED:
            self.bots = {aid: RandomController(int(bot_seed)) if opponent == "random"
                         else FrontierController(int(bot_seed), style=opponent)
                         for aid, bot_seed in zip(self.enemies, seeds, strict=True)}
        self.opponent_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
        self.frozen_opponent = frozen_opponent
        self.terminal_margin_weight = terminal_margin_weight
        self.ticks = self.entry_opportunities = self.entry_choices = self.entry_successes = 0
        self.done = False
        self.gap = (self.state.areas[0] - self.state.areas[1]) / self.state.board.size

    def _state(self):
        observations = [self.observations[aid] for aid in self.allies]
        return TeamState.from_observations(observations, self.env.core.experiment.tile_encoding, self.team)

    def view(self):
        return self.state, tuple(self.history), self.state.choices.mask(self.force_entry)

    def world_indices(self, joint):
        indices = JOINT_ACTIONS[joint]
        return REFLECT_ACTION[indices] if self.team == "blue" else indices

    def opponent_actions(self):
        if self.exercise is not None and self.ticks == 0:
            return self.exercise.opening_actions
        if self.bots is not None:
            return {aid: bot.act(self.observations[aid]).index for aid, bot in self.bots.items()}
        opponent_team = TEAMS[1 - self.team_number]
        enemy_obs = [self.observations[aid] for aid in self.enemies]
        state = TeamState.from_observations(enemy_obs, self.env.core.experiment.tile_encoding, opponent_team)
        if self.frozen_opponent is not None:
            joint = self.frozen_opponent(state, self.opponent_rng)
        elif self.opponent in SEARCH_BUDGETS:
            joint = choose_search_action(state, self.opponent_rng, SEARCH_BUDGETS[self.opponent],
                                         respawn_delay=self.env.core.config.respawn_delay)
        else:
            joint = cooperative_joint(state, self.opponent_rng, defensive=self.opponent == "cooperative_holdout")
        actions = JOINT_ACTIONS[joint]
        if opponent_team == "blue":
            actions = REFLECT_ACTION[actions]
        return dict(zip(self.enemies, map(int, actions), strict=True))

    def step(self, joint, *, opponent_actions=None):
        old = self.state
        choices = old.choices
        if not 0 <= joint < len(JOINT_ACTIONS) or not choices.mask(self.force_entry)[joint]:
            raise ValueError("joint action is outside the current policy mask")
        before = self.env.core.snapshot() if self.scenes is not None else None
        actions = dict(zip(self.allies, map(int, self.world_indices(joint)), strict=True))
        if opponent_actions is not None and set(opponent_actions) != set(self.enemies):
            raise ValueError("scripted actions must cover exactly the three opponent slots")
        actions.update(self.opponent_actions() if opponent_actions is None else opponent_actions)
        self.observations, _, terminals, truncations, _ = self.env.step(actions)
        self.last_actions = actions
        if any(truncations.values()):
            raise RuntimeError("unexpected external truncation of a territory match")
        result = self.env.last_result
        if any(event.kind == "rejected" for event in result.events):
            raise RuntimeError("a controller submitted an illegal action")
        self._record_entry(old, joint, result.events)
        if self.scenes is not None:
            applied = {aid: Action.from_index(index) for aid, index in actions.items()}
            self.scenes.update(SceneTick(before, result, self.team_number, self.allies, None, applied))
        self.history.append((old, joint))
        self.state = self._state()
        self.done = all(terminals.values())
        self.ticks += 1
        gap = (self.state.areas[0] - self.state.areas[1]) / self.state.board.size
        terminal = (float(np.sign(gap)) + self.terminal_margin_weight * gap) if self.done else 0.0
        reward = terminal + 0.25 * (DISCOUNT * (0.0 if self.done else gap) - self.gap)
        self.gap = gap
        return reward, self.done

    def _record_entry(self, state, joint, events):
        self.entry_opportunities += int(state.choices.certified.any())
        movers = np.flatnonzero(state.choices.entry_movers[joint])
        if not len(movers):
            return
        self.entry_choices += 1
        deaths = {event.agent_id for event in events if event.kind == "death"}
        landed = {(event.agent_id, event.position) for event in events
                  if event.kind == "move" and event.reason == "success" and event.agent_id not in deaths}
        captures = {event.position for event in events if event.kind == "capture" and event.new_owner == self.team_number}
        for slot in movers:
            row, col = map(int, state.targets[joint, slot])
            if self.team == "blue":
                col = state.board.shape[1] - 1 - col
            if (self.allies[slot], (row, col)) not in landed or (row, col) not in captures:
                raise RuntimeError("a certified capture-entry action failed in the engine")
        self.entry_successes += 1


def entry_scenario(game, team, rng=None):
    """An empty enemy tile that requires both capture and cover to enter safely."""
    height, width = game.height, game.width
    if height < 6 or width < 6:
        raise ValueError("the capture-entry scenario needs a board at least 6x6")
    col = width // 2 - 1
    row = 3 if rng is None else int(rng.choice([r for r in range(1, height - 1) if r != 3]))
    allies = [(row, col), (row - 1, col + 1), (row, col - 1)]
    enemies = [(row + 1, col), (1, width - 2), (height - 2, width - 3)]
    if rng is not None:
        allies = [allies[index] for index in rng.permutation(3)]
        rows, columns = np.indices((height, width))
        safe_positions = ((columns > col) & (np.abs(rows - row) + np.abs(columns - col - 1) > 1)
                          & (np.abs(rows - (row - 1)) + np.abs(columns - col - 1) > 1))
        cells = np.argwhere(safe_positions)
        enemies[1:] = [tuple(map(int, cells[i])) for i in rng.choice(len(cells), 2, replace=False)]
    # For the fixed evaluation layout, keep the upper enemy away from the entry tile.
    if rng is None and width == 6:
        enemies[1] = (0, width - 1)
        enemies[2] = (height - 2, width - 2)
    owner = np.broadcast_to(np.arange(width) > col, (height, width)).astype(np.int8).copy()
    for cell in allies:
        owner[cell] = 0
    for cell in enemies:
        owner[cell] = 1
    positions = allies + enemies
    if team == "blue":
        owner = 1 - owner[:, ::-1]
        positions = [(r, width - 1 - c) for r, c in enemies + allies]
    return {"owner": owner, "agents": [AgentState(aid, i // 3, pos)
                                       for i, (aid, pos) in enumerate(zip(AGENT_IDS, positions, strict=True))]}
