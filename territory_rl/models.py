from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Mapping
import numpy as np

Position = tuple[int, int]  # row, column
AGENT_IDS = tuple(f"{team}_{i}" for team in ("red", "blue") for i in range(3))


class Action(IntEnum):
    NOOP = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5
    ATTACK_UP = 6
    ATTACK_DOWN = 7
    ATTACK_LEFT = 8
    ATTACK_RIGHT = 9

    @classmethod
    def parse(cls, value):
        """입력 정수를 1~9 범위의 Action으로 변환한다.
        NumPy 정수도 허용하지만 bool·실수·문자열·범위 밖 값은 ValueError로 거부한다.
        """
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"action must be an integer in [1,9], got {value!r}")
        return cls(int(value))

    @property
    def index(self):
        """길이 9 마스크/정책 배열의 0 기반 인덱스를 반환한다."""
        return int(self) - 1

    @classmethod
    def from_index(cls, index):
        """Discrete(9) 인덱스를 실행 코드 1..9로 변환한다. UNKNOWN은 실행하지 않는다."""
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < 9:
            raise ValueError("action index must be an integer in [0,8]")
        return cls(int(index) + 1)

    @property
    def is_attack(self):
        """이 행동이 네 방향 공격 중 하나인지 반환한다."""
        return self.value >= 6

    @property
    def is_move(self):
        """이 행동이 네 방향 이동 중 하나인지 반환한다."""
        return 2 <= self.value <= 5

    def target(self, position: Position) -> Position:
        """현재 (행, 열)에서 행동 방향으로 한 칸 떨어진 목표 좌표를 반환한다.
        이동·공격 행동에만 호출해야 한다. NOOP은 대상이 아니며 경계 검사도 호출자가 수행한다.
        """
        if self == Action.NOOP:
            raise ValueError("NOOP has no target")
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[(self.value - 2) % 4]
        return position[0] + dr, position[1] + dc


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    team: int
    position: Position
    last_attack_tick: int
    respawn_at: int | None
    last_action: Action
    action_valid: bool
    alive: bool = True


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
    reward_components: Mapping[str, Mapping[str, float]]
    terminated: bool
    truncated: bool = False
    termination_reasons: tuple[str, ...] = ()
    eliminated_teams: tuple[str, ...] = ()

    def to_dict(self):
        """한 틱의 결과를 JSON 로그에 사용할 일반 사전으로 변환한다.
        이벤트는 필드 사전으로, 읽기 전용 보상·기여도 매핑은 일반 dict로 복사한다.
        """
        from dataclasses import asdict
        return {"tick": self.tick, "events": [asdict(e) for e in self.events],
                "area_before": self.area_before, "area_after": self.area_after,
                "rewards": dict(self.rewards), "credits": dict(self.credits),
                "reward_components": {a: dict(parts) for a, parts in self.reward_components.items()},
                "terminated": self.terminated, "truncated": self.truncated,
                "termination_reasons": self.termination_reasons, "eliminated_teams": self.eliminated_teams}

