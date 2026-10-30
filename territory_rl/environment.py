from dataclasses import replace
from numbers import Integral
from types import MappingProxyType
import numpy as np
from .agent import AgentState
from .attack import resolve_attacks
from .config import ExperimentConfig, GameConfig
from .events import Event
from .map import TileMap
from .models import AGENT_IDS, Action, StateSnapshot, StepResult
from .movement import resolve_movements
from .respawn import respawn_due_agents
from .reward import calculate_rewards


class TerritoryEnvironment:
    """Owns game state. All mutations occur in reset/step; no wall-clock logic."""

    def __init__(self, config=None, experiment=None):
        self.config = config or GameConfig()
        self.experiment = experiment or ExperimentConfig()
        self.episode_id = 0
        self._initialized = False

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if set(options) - {"owner", "agents", "tick"}:
            raise ValueError("unsupported scenario option")
        cfg = self.config
        rng = np.random.default_rng(seed)
        tile_map = TileMap(cfg.height, cfg.width, options.get("owner"))
        tick = options.get("tick", 0)
        if type(tick) is not int or not 0 <= tick < cfg.max_ticks:
            raise ValueError("scenario tick must be an integer in [0,max_ticks)")
        if "agents" in options:
            sequence = list(options["agents"])
            if len(sequence) != 6 or any(not isinstance(a, AgentState) for a in sequence):
                raise ValueError("scenario agents must contain six AgentState objects")
            agents = {a.agent_id: replace(a) for a in sequence}
        else:
            if "owner" in options:
                raise ValueError("custom owner requires explicit scenario agents")
            if cfg.initial_placement == "frontier":
                rows = sorted(int(r) for r in rng.choice(cfg.height, 3, replace=False))
                left = [(r, cfg.width // 2 - 1) for r in rows]
            else:
                cells = [(r, c) for r in range(cfg.height) for c in range(cfg.width // 2)]
                left = [cells[int(i)] for i in rng.choice(len(cells), 3, replace=False)]
            right = [(r, cfg.width - 1 - c) for r, c in left]
            agents = {aid: AgentState(aid, i // 3, (left + right)[i]) for i, aid in enumerate(AGENT_IDS)}
        self._validate_state(tile_map, agents, tick)
        self._map, self._agents, self._rng = tile_map, agents, rng
        self.tick, self.terminated = tick, False
        self.episode_id += 1
        self._initialized = True
        return self.snapshot()

    @staticmethod
    def _validate_state(tile_map, agents, tick):
        if set(agents) != set(AGENT_IDS):
            raise ValueError("expected red_0..2 and blue_0..2")
        seen = set()
        for i, aid in enumerate(AGENT_IDS):
            a = agents[aid]
            if a.team != i // 3:
                raise ValueError("agent team does not match ID")
            if not isinstance(a.last_attack_tick, Integral) or a.last_attack_tick > tick:
                raise ValueError("last attack cannot be in the future")
            a.last_action = Action.parse(a.last_action)
            if a.alive:
                p = a.position
                if len(p) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool) for x in p):
                    raise ValueError("positions must contain two integers")
                p = tuple(map(int, p))
                if not tile_map.contains(p) or tile_map.owner[p] != a.team or p in seen:
                    raise ValueError("living agents require unique friendly positions")
                if a.respawn_at is not None:
                    raise ValueError("living agents cannot have a pending respawn")
                a.position = p
                seen.add(p)
            elif not isinstance(a.respawn_at, Integral) or a.respawn_at < 0:
                raise ValueError("dead agents require a nonnegative respawn_at")

    def _require_reset(self):
        if not self._initialized:
            raise RuntimeError("call reset before accessing the environment")

    def snapshot(self):
        self._require_reset()
        owner = self._map.owner.copy()
        owner.flags.writeable = False
        return StateSnapshot(owner, tuple(self._agents[a].snapshot() for a in AGENT_IDS),
                             self.tick, self.config.max_ticks, self._map.areas(), self.terminated)

    def action_mask(self, aid):
        self._require_reset()
        a = self._agents[aid]
        mask = np.zeros(9, dtype=np.int8)
        mask[0] = 1
        if a.alive and not self.terminated:
            for action in list(Action)[1:]:
                if self._map.contains(action.target(a.position)) and (not action.is_attack or a.can_attack(self.tick + 1)):
                    mask[action] = 1
        return mask

    def observe(self, aid):
        from .observation import build_observation
        return build_observation(self.snapshot(), aid, self.config, self.experiment, self.action_mask(aid))

    def step(self, actions):
        self._require_reset()
        if self.terminated:
            raise RuntimeError("episode is over; call reset")
        if set(actions) - set(AGENT_IDS):
            raise ValueError("unknown agent ID")
        # Parse ALL actions before mutating anything, including RNG.
        requested = {aid: Action.parse(actions.get(aid, Action.NOOP)) for aid in AGENT_IDS}
        applied = {}
        events = []
        for aid in AGENT_IDS:
            action = requested[aid]
            applied[aid] = action if self.action_mask(aid)[action] else Action.NOOP
            if applied[aid] != action:
                events.append(Event("rejected", aid, reason="action_mask", requested=int(action), applied=0))
        tick = self.tick + 1
        before = self._map.areas()
        attack_events = resolve_attacks(self._map.owner, self._agents, applied)
        events.extend(attack_events)
        for aid, action in applied.items():
            a = self._agents[aid]
            a.last_action, a.action_valid = action, True
            if action.is_attack:
                a.last_attack_tick = tick
        for event in attack_events:
            if event.kind == "capture":
                self._map.owner[event.position] = event.new_owner

        def kill_exposed(phase):
            for aid in AGENT_IDS:
                a = self._agents[aid]
                if a.alive and self._map.owner[a.position] != a.team:
                    events.append(Event("death", aid, a.position, reason=phase,
                                        due_tick=tick + self.config.respawn_delay))
                    if phase == "attack" and applied[aid].is_move:
                        events.append(Event("move", aid, applied[aid].target(a.position), a.position, "killed_before_move"))
                    a.position = None
                    a.respawn_at = tick + self.config.respawn_delay

        kill_exposed("attack")
        positions = {aid: a.position for aid, a in self._agents.items() if a.alive}
        positions, moves = resolve_movements(positions, applied, self._map.contains)
        events.extend(moves)
        for aid, position in positions.items():
            self._agents[aid].position = position
        kill_exposed("movement")
        after = self._map.areas()
        terminated = tick == self.config.max_ticks
        if not terminated:
            events.extend(respawn_due_agents(self._agents, self._map.owner, tick, self._rng))
        rewards, credits = calculate_rewards(self.experiment, before, after, events, terminated)
        self.tick, self.terminated = tick, terminated
        self._validate_state(self._map, self._agents, tick)
        return StepResult(tick, tuple(events), before, after, MappingProxyType(rewards),
                          MappingProxyType(credits), terminated)

