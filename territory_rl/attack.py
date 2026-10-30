from collections import defaultdict
from .events import Event


def resolve_attacks(owner, agents, actions):
    """Read-only resolution using all valid attacks from the same start state."""
    targets = defaultdict(lambda: [[], []])
    for aid in sorted(actions):
        a, action = agents[aid], actions[aid]
        if a.alive and action.is_attack:
            targets[action.target(a.position)][a.team].append(aid)
    events = []
    for cell, teams in sorted(targets.items()):
        counts = tuple(map(len, teams))
        old = int(owner[cell])
        new = old if counts[0] == counts[1] else int(counts[1] > counts[0])
        reason = "tie" if counts[0] == counts[1] else ("captured" if old != new else "unchanged")
        for members in teams:
            for aid in members:
                events.append(Event("attack", aid, cell, reason=reason, counts=counts))
        if old != new:
            events.append(Event("capture", position=cell, old_owner=old, new_owner=new,
                                contributors=tuple(teams[new]), counts=counts))
    return tuple(events)

