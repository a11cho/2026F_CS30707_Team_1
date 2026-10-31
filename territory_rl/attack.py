from collections import defaultdict
from .events import Event


def resolve_attacks(owner, agents, actions):
    """틱 시작 상태에서 유효한 공격을 목표 타일별로 모아 공격·점령 이벤트를 반환한다.

    owner는 소유권 배열, agents는 ID별 상태, actions는 검증된 적용 행동이다.
    목표별 팀 공격 수가 같으면 기존 소유권을 유지하며, 다르면 많은 팀이 점령한다.
    변경된 타일에는 승리 팀 공격자 목록을 남겨 개인 기여도를 계산할 수 있게 한다.
    입력 상태를 수정하지 않는다. 경계·생존·cooldown 검증은 호출하는 환경이 담당한다.
    """
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

