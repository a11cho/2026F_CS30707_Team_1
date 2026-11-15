import numpy as np
from .models import AGENT_IDS


def build_observation(snapshot, aid, config, strategy, action_mask, context):
    """전체 맵을 요청자 기준 아군=1/적군=-1로 복사하고 전략별 현재 행동을 결합한다.

    에이전트 배열은 아군 번호순 다음 적군 번호순이다. 좌표/방향은 변환하지 않는다.
    context는 환경 내부 수집기가 필터링한 코드·회차·출처이며 외부 입력이 아니다.
    """
    states = {a.agent_id: a for a in snapshot.agents}
    me = states[aid]
    ordered = [states[key] for team in (me.team, 1-me.team)
               for key in AGENT_IDS if states[key].team == team]
    codes, decision_round, source = context
    known = (codes != 0).astype(np.int8)
    features = np.zeros((3, 9), dtype=np.float32)
    for i in np.flatnonzero(known):
        features[i, int(codes[i])-1] = 1
    return {
        "owner": np.where(snapshot.owner == me.team, 1, -1).astype(np.int8),
        "agent_positions": np.array([a.position if a.alive else (-1, -1) for a in ordered], dtype=np.int32),
        "agent_teams": np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
        "agent_alive": np.array([a.alive for a in ordered], dtype=np.int8),
        "agent_attack_ready": np.array([a.alive and not snapshot.terminated and
            snapshot.tick + 1 - a.last_attack_tick >= 2 for a in ordered], dtype=np.int8),
        "agent_respawn_wait": np.array([0 if a.respawn_at is None else max(0, a.respawn_at-snapshot.tick)
                                        for a in ordered], dtype=np.int32),
        "self_index": int(aid.rsplit("_", 1)[1]), "self_team": 1,
        "tick": snapshot.tick, "max_ticks": config.max_ticks,
        "remaining_ticks": max(0, config.max_ticks-snapshot.tick),
        "areas": np.array([snapshot.areas[me.team], snapshot.areas[1-me.team]], dtype=np.int32),
        "decision_strategy": strategy, "decision_round": decision_round, "action_source": source,
        "team_action_codes": codes.copy(), "team_actions_known": known,
        "team_action_features": features, "action_mask": action_mask.copy(),
    }
