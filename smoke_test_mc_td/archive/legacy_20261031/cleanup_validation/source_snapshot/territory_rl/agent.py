from dataclasses import dataclass
from .models import Action, AgentSnapshot, Position


@dataclass
class AgentState:
    agent_id: str
    team: int
    position: Position | None
    last_attack_tick: int = -2
    respawn_at: int | None = None
    last_action: Action = Action.NOOP
    action_valid: bool = False

    @property
    def alive(self):
        """현재 좌표가 존재하면 생존 상태인지 나타내는 True를 반환한다.
        사망·부활 대기 상태는 position=None으로 표현하며 별도 생존 필드를 중복 저장하지 않는다.
        """
        return self.position is not None

    def can_attack(self, next_tick: int):
        """next_tick에 공격을 시도할 수 있는지 반환한다.
        생존 중이고 마지막 공격 틱으로부터 2틱 이상 지났을 때만 허용한다. 상태는 변경하지 않는다.
        """
        return self.alive and next_tick - self.last_attack_tick >= 2

    def snapshot(self):
        """현재 에이전트의 필드를 변경 불가능한 AgentSnapshot 값 객체로 반환한다."""
        return AgentSnapshot(**vars(self))

