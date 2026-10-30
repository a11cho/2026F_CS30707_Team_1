from itertools import groupby
import numpy as np
from .events import Event


def respawn_due_agents(agents, owner, tick, rng):
    """Oldest due time first; seeded tie breaking and sampling without replacement."""
    occupied = {a.position for a in agents.values() if a.alive}
    events = []
    for team in (0, 1):
        due = sorted((a for a in agents.values() if a.team == team and not a.alive and a.respawn_at <= tick),
                     key=lambda a: (a.respawn_at, a.agent_id))
        ordered = []
        for _, group in groupby(due, key=lambda a: a.respawn_at):
            members = list(group)
            ordered.extend(members[int(i)] for i in rng.permutation(len(members)))
        candidates = [tuple(map(int, p)) for p in np.argwhere(owner == team) if tuple(p) not in occupied]
        for a in ordered:
            if not candidates:
                events.append(Event("respawn_wait", a.agent_id, reason="no_free_friendly_tile", due_tick=a.respawn_at))
                continue
            p = candidates.pop(int(rng.integers(len(candidates))))
            due_tick = a.respawn_at
            a.position, a.respawn_at = p, None
            occupied.add(p)
            events.append(Event("respawn", a.agent_id, p, due_tick=due_tick))
    return events
