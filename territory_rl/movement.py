from collections import Counter
from .events import Event


def resolve_movements(positions, actions, contains):
    """모든 이동 의도를 함께 판정하여 최종 위치 사전과 이동 이벤트 튜플을 반환한다.

    positions는 생존자의 ID별 시작 좌표, actions는 적용 행동, contains는 경계 검사 함수다.
    같은 목적지 경쟁은 전원 실패하고, 점유자의 이동 성공 여부에 따라 연쇄 이동을 결정한다.
    두 명의 자리 교환은 실패하지만 더 긴 순환은 허용한다. 입력 위치는 직접 바꾸지 않는다.
    """
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
        """aid의 목적지를 점유한 에이전트를 재귀적으로 따라가며 이동 성공 여부를 반환한다.
        path는 현재 탐색 경로이며 재방문 지점에서 순환을 찾는다.
        판정은 바깥 함수의 outcomes에 저장하므로 한 에이전트의 실패가 뒤따르는 이동에도 전파된다.
        """
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

