from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Mapping
import numpy as np

Position = tuple[int, int]  # row, column
AGENT_IDS = tuple(f"{team}_{i}" for team in ("red", "blue") for i in range(3))


class Action(IntEnum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    ATTACK_UP = 5
    ATTACK_DOWN = 6
    ATTACK_LEFT = 7
    ATTACK_RIGHT = 8

    @classmethod
    def parse(cls, value):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"action must be an integer in [0,8], got {value!r}")
        return cls(int(value))

    @property
    def is_attack(self):
        return self.value >= 5

    @property
    def is_move(self):
        return 1 <= self.value <= 4

    def target(self, position: Position) -> Position:
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[(self.value - 1) % 4]
        return position[0] + dr, position[1] + dc


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    team: int
    position: Position | None
    last_attack_tick: int
    respawn_at: int | None
    last_action: Action
    action_valid: bool

    @property
    def alive(self):
        return self.position is not None


@dataclass(frozen=True)
class StateSnapshot:
    owner: np.ndarray
    agents: tuple[AgentSnapshot, ...]
    tick: int
    max_ticks: int
    areas: tuple[int, int]
    terminated: bool


@dataclass(frozen=True)
class StepResult:
    tick: int
    events: tuple
    area_before: tuple[int, int]
    area_after: tuple[int, int]
    rewards: Mapping[str, float]
    credits: Mapping[str, float]
    terminated: bool
    truncated: bool = False

    def to_dict(self):
        from dataclasses import asdict
        return {"tick": self.tick, "events": [asdict(e) for e in self.events],
                "area_before": self.area_before, "area_after": self.area_after,
                "rewards": dict(self.rewards), "credits": dict(self.credits),
                "terminated": self.terminated, "truncated": self.truncated}

