import numpy as np
from .models import AGENT_IDS


def build_observation(snapshot, aid, config, experiment, action_mask):
    """스냅샷에서 aid의 관측 사전을 생성하고 내부 상태와 독립적인 배열들을 반환한다.

    지역 격자의 6개 채널은 아군 땅·적 땅·경계 밖·본인·보이는 아군·보이는 적이다.
    자기 상태는 정규화된 좌표·생존·공격 가능·부활 대기·남은 시간으로 구성한다.
    aware에서만 두 팀원의 상태와 직전 완료 행동을 제공하며 local의 추가 팀원 정보는 0이다.
    사망자의 지역 격자는 0으로 채운다. 현재 틱의 미완료 제출 행동은 참조하지 않는다.
    """
    states = {a.agent_id: a for a in snapshot.agents}
    me = states[aid]
    r = config.observation_radius
    grid = np.zeros((6, 2*r+1, 2*r+1), dtype=np.float32)
    if me.alive:
        row, col = me.position
        for y in range(2*r+1):
            for x in range(2*r+1):
                p = row + y-r, col + x-r
                if not (0 <= p[0] < config.height and 0 <= p[1] < config.width):
                    grid[2, y, x] = 1
                else:
                    grid[int(snapshot.owner[p] != me.team), y, x] = 1
        for a in snapshot.agents:
            if a.alive:
                y, x = a.position[0]-row+r, a.position[1]-col+r
                if 0 <= y <= 2*r and 0 <= x <= 2*r:
                    channel = 3 if a.agent_id == aid else (4 if a.team == me.team else 5)
                    grid[channel, y, x] = 1

    def fields(a):
        """에이전트 a를 [정규화 행, 정규화 열, 생존, 공격 가능, 부활 대기 비율]로 변환한다.
        사망자의 좌표는 0으로 채우므로 사용자는 생존/유효 mask를 함께 해석해야 한다.
        """
        position = (0, 0) if not a.alive else (a.position[0] / (config.height-1), a.position[1] / (config.width-1))
        wait = 0 if a.respawn_at is None else max(0, a.respawn_at - snapshot.tick) / max(1, config.respawn_delay)
        return [*position, float(a.alive), float(a.alive and snapshot.tick + 1 - a.last_attack_tick >= 2), min(1., wait)]

    own = np.array([*fields(me), (config.max_ticks - snapshot.tick)/config.max_ticks], dtype=np.float32)
    teammates = np.zeros((2, 14), dtype=np.float32)  # row,col,alive,ready,wait + 9 one-hot actions
    validity = np.zeros((2, 14), dtype=np.int8)
    if experiment.observation_mode == "aware":
        allies = [states[key] for key in AGENT_IDS if states[key].team == me.team and key != aid]
        for index, a in enumerate(allies):
            teammates[index, :5] = fields(a)
            validity[index, :5] = [int(a.alive), int(a.alive), 1, 1, 1]
            if a.action_valid:
                teammates[index, 5+int(a.last_action)] = 1
                validity[index, 5:] = 1
    return {"local_grid": grid, "self_features": own, "teammate_features": teammates,
            "teammate_feature_mask": validity, "action_mask": action_mask.copy()}

