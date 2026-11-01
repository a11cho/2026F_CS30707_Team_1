from itertools import groupby
import numpy as np
from .events import Event


def respawn_due_agents(agents, owner, tick, rng):
    """부활 예정 틱이 지난 사망자를 빈 아군 타일에 배치하고 관련 이벤트를 반환한다.

    agents의 위치·부활 예정 틱을 직접 갱신하고 전달받은 환경 RNG를 소비한다.
    오래 기다린 순서를 우선하며 예정 틱이 같으면 시드 기반 추첨으로 순서를 정한다.
    후보 타일은 중복 없이 선택한다. 빈 아군 칸이 없으면 예정 틱을 유지한 채 계속 대기한다.
    호출자는 종료 틱에서 이 함수를 생략해야 한다.
    """
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
