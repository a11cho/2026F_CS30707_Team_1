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
        return self.position is not None

    def can_attack(self, next_tick: int):
        return self.alive and next_tick - self.last_attack_tick >= 2

    def snapshot(self):
        return AgentSnapshot(**vars(self))

