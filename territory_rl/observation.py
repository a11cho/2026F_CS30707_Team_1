import numpy as np
from .models import AGENT_IDS, Action


def rotate_action(action, quarter_turns):
    """이동/공격을 시계 방향 90도 단위로 회전한다. 음수는 역변환이다."""
    action = Action.parse(action)
    if action == Action.NOOP:
        return action
    directions = ((Action.ATTACK_UP, Action.ATTACK_RIGHT, Action.ATTACK_DOWN, Action.ATTACK_LEFT)
                  if action.is_attack else (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT))
    return directions[(directions.index(action) + quarter_turns) % len(directions)]


def rotate_action_mask(mask, quarter_turns):
    """실제 맵 방향 마스크를 관측 방향으로 옮긴다."""
    rotated = np.empty_like(mask)
    for action in Action:
        rotated[rotate_action(action, quarter_turns).index] = mask[action.index]
    return rotated


def build_observation(snapshot, aid, experiment, strategy, action_mask, context, rotation):
    """자기 중심 타일맵·아군 행동·보조 정보를 독립적인 배열로 반환한다.

    position은 사망 중에도 유지된다. 격자에는 생존자만 그린다.
    에이전트 배열/행동 슬롯은 팀 내 번호순이며 모든 방향은 관측 좌표계다.
    """
    states = {a.agent_id: a for a in snapshot.agents}
    me = states[aid]
    ordered = [states[key] for team in (me.team, 1-me.team)
               for key in AGENT_IDS if states[key].team == team]
    encoding = experiment.tile_encoding
    size = experiment.observation_window_size
    center = size // 2
    tilemap = np.full((size, size), encoding.outside, dtype=np.int8)
    row_offset, column_offset = center - me.position[0], center - me.position[1]
    height, width = snapshot.owner.shape
    tilemap[row_offset:row_offset+height, column_offset:column_offset+width] = np.where(
        snapshot.owner == me.team, encoding.friendly, encoding.enemy)
    for index, agent in enumerate(ordered):
        if agent.alive:
            values = encoding.friendly_agents if index < 3 else encoding.enemy_agents
            row, column = agent.position
            tilemap[row+row_offset, column+column_offset] = values[index % 3]

    world_codes, decision_round, source = context
    codes = np.array([int(rotate_action(code, rotation)) if code else 0 for code in world_codes], dtype=np.int8)
    known = (codes != 0).astype(np.int8)
    features = np.zeros((3, 9), dtype=np.float32)
    for index in np.flatnonzero(known):
        features[index, int(codes[index])-1] = 1
    return {
        "tilemap": np.rot90(tilemap, -rotation).copy(),
        "team_actions": codes,
        "auxiliary": {
            "agent_teams": np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
            "agent_alive": np.array([a.alive for a in ordered], dtype=np.int8),
            "agent_attack_ready": np.array([a.alive and not snapshot.terminated and
                snapshot.tick + 1 - a.last_attack_tick >= 2 for a in ordered], dtype=np.int8),
            "agent_respawn_wait": np.array([0 if a.respawn_at is None else max(0, a.respawn_at-snapshot.tick)
                                            for a in ordered], dtype=np.int32),
            "self_index": int(aid.rsplit("_", 1)[1]), "self_team": 1,
            "tick": snapshot.tick, "max_ticks": snapshot.max_ticks,
            "remaining_ticks": max(0, snapshot.max_ticks-snapshot.tick),
            "areas": np.array([snapshot.areas[me.team], snapshot.areas[1-me.team]], dtype=np.int32),
            "decision_strategy": strategy, "decision_round": decision_round, "action_source": source,
            "team_actions_known": known, "team_action_features": features,
            "action_mask": rotate_action_mask(action_mask, rotation),
        },
    }
