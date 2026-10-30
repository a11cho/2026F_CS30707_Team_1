from collections import Counter
from .events import Event


def resolve_movements(positions, actions, contains):
    """No in-place moves. Failed dependencies propagate; cycles > 2 succeed."""
    occupied = {p: aid for aid, p in positions.items()}
    targets = {aid: actions[aid].target(p) for aid, p in positions.items() if actions[aid].is_move}
    counts = Counter(p for p in targets.values() if contains(p))
    outcomes = {}
    for aid, p in targets.items():
        if not contains(p):
            outcomes[aid] = "boundary"
        elif counts[p] > 1:
            outcomes[aid] = "contested"

    def visit(aid, path):
        if aid in outcomes:
            return outcomes[aid] == "success"
        if aid in path:
            cycle = path[path.index(aid):]
            reason = "swap" if len(cycle) == 2 else "success"
            for member in cycle:
                outcomes[member] = reason
            return reason == "success"
        blocker = occupied.get(targets[aid])
        if blocker is None:
            success = True
        elif blocker not in targets:
            success = False
        else:
            success = visit(blocker, path + [aid])
        if aid not in outcomes:
            outcomes[aid] = "success" if success else "blocked"
        return outcomes[aid] == "success"

    for aid in sorted(targets):
        visit(aid, [])
    final = dict(positions)
    for aid, reason in outcomes.items():
        if reason == "success":
            final[aid] = targets[aid]
    events = tuple(Event("move", aid, targets[aid], positions[aid], outcomes[aid]) for aid in sorted(targets))
    return final, events

