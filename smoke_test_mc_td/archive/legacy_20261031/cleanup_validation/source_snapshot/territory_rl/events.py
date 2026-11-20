from dataclasses import dataclass
from .models import Position


@dataclass(frozen=True)
class Event:
    kind: str
    agent_id: str | None = None
    position: Position | None = None
    origin: Position | None = None
    reason: str = ""
    old_owner: int | None = None
    new_owner: int | None = None
    contributors: tuple[str, ...] = ()
    counts: tuple[int, int] = (0, 0)
    requested: int | None = None
    applied: int | None = None
    due_tick: int | None = None

