from dataclasses import replace
from itertools import combinations
import numpy as np
import pytest
from territory_rl import Action as A, ExperimentConfig, GameConfig, TerritoryEnvironment
from territory_rl.agent import AgentState
from territory_rl.models import AGENT_IDS
from territory_rl.movement import resolve_movements


def scenario(positions=None, owner=None, delay=2, ticks=20, experiment=None, tick=0):
    """지정 좌표·소유권·부활 지연으로 6×6 테스트 환경을 만들어 반환한다.
    생존자의 시작 칸은 해당 팀 소유로 맞추고, 좌표가 None인 에이전트는 부활 대기로 설정한다.
    """
    cfg = GameConfig(height=6, width=6, max_ticks=ticks, respawn_delay=delay, observation_radius=2)
    if owner is None:
        owner = np.zeros((6, 6), dtype=np.int8)
        owner[:, 3:] = 1
    else:
        owner = owner.copy()
    base = dict(zip(AGENT_IDS, [(0, 0), (1, 0), (2, 0), (0, 5), (1, 5), (2, 5)]))
    base.update(positions or {})
    agents = []
    for i, aid in enumerate(AGENT_IDS):
        p = base[aid]
        if p is not None:
            owner[p] = i//3
        agents.append(AgentState(aid, i//3, p, respawn_at=0 if p is None else None))
    env = TerritoryEnvironment(cfg, experiment)
    env.reset(123, {"owner": owner, "agents": agents, "tick": tick})
    return env


def agent(env, aid):
    """스냅샷에서 aid에 해당하는 에이전트를 찾아 반환하는 테스트용 조회 함수다."""
    return next(a for a in env.snapshot().agents if a.agent_id == aid)


def test_attack_majority_tie_credit_and_cooldown():
    """공격 동률 유지, 다음 틱 공격 제한, 다수 공격 성공과 공동 기여도 분배를 검증한다."""
    env = scenario({"red_0": (2, 2), "red_1": (3, 3), "blue_0": (2, 4)})
    original = env.snapshot().owner[2, 3]
    r = env.step({"red_0": A.ATTACK_RIGHT, "blue_0": A.ATTACK_LEFT})
    assert env.snapshot().owner[2, 3] == original == 1
    assert sum(r.credits.values()) == 0
    assert not env.action_mask("red_0")[5:].any()
    r = env.step({"red_0": A.ATTACK_RIGHT})
    assert any(e.kind == "rejected" for e in r.events)
    assert env.action_mask("red_0")[5:].all()
    r = env.step({"red_0": A.ATTACK_RIGHT, "red_1": A.ATTACK_UP, "blue_0": A.ATTACK_LEFT})
    assert env.snapshot().owner[2, 3] == 0
    assert r.credits["red_0"] == r.credits["red_1"] == .5
    assert sum(r.credits.values()) == 1


def test_friendly_attack_consumes_cooldown_without_credit():
    """아군 타일 공격은 점령 보상을 만들지 않지만 공격 cooldown은 소비하는지 검증한다."""
    env = scenario()
    r = env.step({"red_0": A.ATTACK_RIGHT})
    assert sum(r.credits.values()) == 0
    assert not env.action_mask("red_0")[5:].any()
    env.step({})
    assert env.action_mask("red_0")[A.ATTACK_RIGHT]


def test_mutual_attacks_survive_simultaneous_attacker_death():
    """서로를 죽이는 동시 공격에서도 양쪽의 유효 공격과 점령 기여도가 모두 남는지 검증한다."""
    env = scenario({"red_0": (2, 2), "blue_0": (2, 3)})
    r = env.step({"red_0": A.ATTACK_RIGHT, "blue_0": A.ATTACK_LEFT})
    assert not agent(env, "red_0").alive and not agent(env, "blue_0").alive
    assert r.credits["red_0"] == r.credits["blue_0"] == 1
    assert len([e for e in r.events if e.kind == "death"]) == 2


def test_attack_kill_cancels_escape_and_releases_occupancy():
    """공격으로 사망한 에이전트의 도주 이동이 취소되고 빈 자리에 다른 에이전트가 이동하는지 검증한다."""
    env = scenario({"red_0": (2, 2), "blue_0": (2, 3), "blue_1": (1, 2)})
    r = env.step({"red_0": A.LEFT, "blue_0": A.ATTACK_LEFT, "blue_1": A.DOWN})
    assert not agent(env, "red_0").alive
    assert agent(env, "blue_1").position == (2, 2)
    assert any(e.reason == "killed_before_move" for e in r.events)


def test_move_into_tile_captured_same_tick_and_enemy_move_death():
    """같은 틱 아군이 칠한 타일에는 이동할 수 있고, 적 타일로 그대로 이동하면 죽는지 검증한다."""
    env = scenario({"red_0": (2, 2), "red_1": (1, 3)})
    assert env.action_mask("red_0")[A.RIGHT]  # Enemy ownership is not an invalid action.
    env.step({"red_0": A.RIGHT, "red_1": A.ATTACK_DOWN})
    assert agent(env, "red_0").position == (2, 3)
    r = env.step({"red_0": A.RIGHT})
    assert not agent(env, "red_0").alive
    assert any(e.kind == "death" and e.reason == "movement" for e in r.events)


@pytest.mark.parametrize("delay", [0, 1, 2, 4])
def test_exact_respawn_timing(delay):
    """여러 delay 값에 대해 사망 후 대기 틱 수와 실제 부활 틱이 정의대로 일치하는지 검증한다."""
    env = scenario({"red_0": (2, 2)}, delay=delay)
    env.step({"red_0": A.RIGHT})
    if delay == 0:
        assert agent(env, "red_0").alive
    else:
        assert not agent(env, "red_0").alive
        for _ in range(delay-1):
            env.step({})
            assert not agent(env, "red_0").alive
        env.step({})
        assert env.tick == 1+delay
        assert agent(env, "red_0").alive


def test_no_final_tick_respawn_and_terminal_rewards():
    """종료 틱에 부활을 생략하고 termination을 반환하며 종료 후 step 호출을 거부하는지 검증한다."""
    env = scenario({"red_0": (2, 2)}, delay=0, ticks=1)
    r = env.step({"red_0": A.RIGHT})
    assert r.terminated and not r.truncated
    assert not agent(env, "red_0").alive
    assert not any(e.kind == "respawn" for e in r.events)
    with pytest.raises(RuntimeError):
        env.step({})


def test_zero_territory_remains_to_horizon():
    """영역이 0인 팀은 부활 없이 대기하면서도 게임은 T까지 지속되고 종료 패배 보상을 받는지 검증한다."""
    env = scenario({a: None for a in AGENT_IDS[3:]}, owner=np.zeros((6, 6)), ticks=4)
    for t in range(1, 5):
        r = env.step({})
        assert env.snapshot().areas == (36, 0)
        assert all(not agent(env, a).alive for a in AGENT_IDS[3:])
        assert r.terminated == (t == 4)
    assert r.rewards["blue_0"] == -1


def test_respawn_shortage_no_overlap():
    """부활 후보가 한 칸뿐이면 한 명만 부활하고 나머지는 기다리며 좌표 중복이 없는지 검증한다."""
    owner = np.zeros((6, 6))
    owner[5, 5] = 1
    env = scenario({a: None for a in AGENT_IDS[3:]}, owner=owner)
    r = env.step({})
    assert sum(agent(env, a).alive for a in AGENT_IDS[3:]) == 1
    assert len([e for e in r.events if e.kind == "respawn_wait"]) == 2
    positions = [a.position for a in env.snapshot().agents if a.alive]
    assert len(positions) == len(set(positions))


@pytest.mark.parametrize("positions,actions,expected,reasons", [
    ({"a": (0, 0), "b": (0, 1)}, {"a": A.RIGHT, "b": A.RIGHT}, {"a": (0, 1), "b": (0, 2)}, ["success", "success"]),
    ({"a": (0, 0), "b": (0, 1)}, {"a": A.RIGHT, "b": A.NOOP}, {"a": (0, 0), "b": (0, 1)}, ["blocked"]),
    ({"a": (0, 0), "b": (0, 1)}, {"a": A.RIGHT, "b": A.LEFT}, {"a": (0, 0), "b": (0, 1)}, ["swap", "swap"]),
    ({"a": (0, 0), "b": (0, 2)}, {"a": A.RIGHT, "b": A.LEFT}, {"a": (0, 0), "b": (0, 2)}, ["contested", "contested"]),
    ({"a": (0, 0), "b": (0, 1), "c": (0, 2)}, {"a": A.RIGHT, "b": A.RIGHT, "c": A.RIGHT}, {"a": (0, 0), "b": (0, 1), "c": (0, 2)}, ["blocked", "blocked", "boundary"]),
    ({"a": (0, 0), "b": (0, 1), "c": (1, 1), "d": (1, 0)}, {"a": A.RIGHT, "b": A.DOWN, "c": A.LEFT, "d": A.UP}, {"a": (0, 1), "b": (1, 1), "c": (1, 0), "d": (0, 0)}, ["success"]*4),
])
def test_movement_cases(positions, actions, expected, reasons):
    """연쇄 이동·점유 방해·자리 교환·목적지 경쟁·경계·4명 순환의 위치와 판정 이유를 검증한다."""
    final, events = resolve_movements(positions, actions, lambda p: 0 <= p[0] < 3 and 0 <= p[1] < 3)
    assert final == expected
    assert [e.reason for e in events] == reasons


def test_movement_against_exhaustive_simultaneous_constraint_solver():
    """작은 임의 배치에서 모든 이동 부분집합을 탐색한 제약 해와 resolver 결과를 비교한다.
    같은 알고리즘을 반복 구현하지 않고 최종 좌표 유일성·교환 금지 등 독립 제약으로 검증한다.
    """
    rng = np.random.default_rng(91)
    cells = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(150):
        positions = {str(i): cells[int(j)] for i, j in enumerate(rng.choice(9, 4, replace=False))}
        actions = {a: A(int(rng.integers(5))) for a in positions}
        targets = {a: action.target(positions[a]) for a, action in actions.items() if action.is_move}
        eligible = [a for a, p in targets.items() if p in cells and list(targets.values()).count(p) == 1]
        best = []
        for size in range(len(eligible)+1):
            for movers in combinations(eligible, size):
                candidate = {a: targets[a] if a in movers else p for a, p in positions.items()}
                if len(set(candidate.values())) != 4:
                    continue
                if any(candidate[a] == positions[b] and candidate[b] == positions[a] for a, b in combinations(movers, 2)):
                    continue
                best = list(movers)
        expected = {a: targets[a] if a in best else p for a, p in positions.items()}
        actual, _ = resolve_movements(positions, actions, lambda p: p in cells)
        assert actual == expected


@pytest.mark.parametrize("bad", [-1, 9, 1.5, True, "1", None])
def test_invalid_action_is_atomic(bad):
    """잘못된 행동이 섞이면 다른 에이전트의 유효 행동도 적용되지 않고 틱·맵·상태가 유지되는지 검증한다."""
    env = scenario()
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step({"red_0": A.ATTACK_RIGHT, "blue_2": bad})
    assert env.tick == before.tick
    np.testing.assert_array_equal(env.snapshot().owner, before.owner)
    assert env.snapshot().agents == before.agents


def test_reset_scenario_validation():
    """서로 겹친 초기 위치를 가진 시나리오가 reset 검증에서 거부되는지 확인한다."""
    env = scenario()
    states = [AgentState(a.agent_id, a.team, a.position) for a in env.snapshot().agents]
    states[1] = replace(states[1], position=states[0].position)
    with pytest.raises(ValueError):
        env.reset(1, {"agents": states})
    assert env.tick == 0


@pytest.mark.parametrize("mode,signal,bonus,expected", [
    ("individual", "net", 0, (1., 0., 0.)),
    ("team", "net", 0, (1/3, 1/3, 1/3)),
    ("team", "gross", 0, (1/3, 1/3, 1/3)),
    ("team", "terminal", 1, (1., 1., 1.)),
])
def test_reward_modes(mode, signal, bonus, expected):
    """개인 기여·팀 순증가·팀 총획득·종료 승패 모드의 보상 수치가 예상과 일치하는지 검증한다."""
    env = scenario({"red_0": (2, 2)}, ticks=1, experiment=ExperimentConfig(reward_mode=mode, team_signal=signal, terminal_bonus=bonus))
    r = env.step({"red_0": A.ATTACK_RIGHT})
    assert tuple(r.rewards[a] for a in AGENT_IDS[:3]) == pytest.approx(expected)


def test_dead_team_member_receives_shared_reward():
    """틱 시작에 부활 대기 중인 팀원도 같은 틱의 팀 공유 보상을 받는지 검증한다."""
    env = scenario({"red_0": (2, 2), "red_2": None})
    r = env.step({"red_0": A.ATTACK_RIGHT})
    assert r.rewards["red_2"] == r.rewards["red_0"] == pytest.approx(1/3)
