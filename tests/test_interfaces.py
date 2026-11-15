from dataclasses import replace
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from territory_rl import Action, GameConfig, ExperimentConfig, TerritoryEnvironment
from territory_rl.controller import FrontierController, NoopController, RandomController
from territory_rl.keyboard import KeyboardController
from territory_rl.manager import EpisodeManager
from territory_rl.models import AGENT_IDS
from territory_rl.parallel_env import TerritoryParallelEnv
from territory_rl.visualization import render_rgb
from test_rules import scenario


@pytest.mark.parametrize("strategy", ["independent", "sequential", "feedback"])
@pytest.mark.parametrize("reward_mode", ["individual", "team"])
def test_spaces_lifecycle_and_invariants(strategy, reward_mode):
    """관측/보상 네 조합에서 공간 형식·전체 면적·좌표 유일성·종료 시점 등 환경 계약을 검증한다.
    임의 행동으로 환경만 진행하며 정책 학습은 수행하지 않는다.
    """
    env = TerritoryParallelEnv(GameConfig(height=6, width=8, max_ticks=64), ExperimentConfig(strategy, reward_mode))
    obs, _ = env.reset(13)
    rng = np.random.default_rng(8)
    for tick in range(1, 65):
        for aid, o in obs.items():
            assert env.observation_space(aid).contains(o)
            assert not o["team_actions_known"].any()
        actions = {a: int(rng.integers(9)) for a in env.agents}
        if strategy == "independent":
            obs, rewards, terms, truncs, _ = env.step(actions)
        else:
            env.begin_decision()
            result = None
            while result is None:
                for aid in AGENT_IDS:
                    team, number = aid.rsplit("_", 1)
                    session = env.agent_session(team, int(number))
                    assert env.observation_space(aid).contains(session.observe())
                    session.submit_action(Action.from_index(actions[aid]))
                result = env.finish_decision_round()
            obs, rewards, terms, truncs, _ = result
        assert set(obs) == set(rewards) == set(AGENT_IDS)
        assert not any(truncs.values())
        assert all(terms.values()) == env.core.terminated
        assert env.state_space.contains(env.state())
        s = env.core.snapshot()
        alive = [a for a in s.agents if a.alive]
        assert len({a.position for a in alive}) == len(alive)
        assert all(s.owner[a.position] == a.team for a in alive)
        assert sum(s.areas) == 48
        if all(terms.values()):
            break
    assert not env.agents


def test_official_parallel_contract_and_seed_reproducibility():
    """PettingZoo 공식 검사로 Parallel API 반환 계약과 시드 재현성을 검증한다. 학습 테스트가 아니다."""
    from pettingzoo.test import parallel_api_test, parallel_seed_test
    factory = lambda: TerritoryParallelEnv(GameConfig(height=6, width=8, max_ticks=40))
    parallel_api_test(factory(), num_cycles=80)
    parallel_seed_test(factory, num_cycles=80)


def test_same_trace_ignores_reward_observation_mode_and_submission_order():
    """동일 시드·행동열에서 관측/보상 모드와 사전 삽입 순서가 상태·이벤트를 바꾸지 않는지 검증한다."""
    game = GameConfig(height=6, width=8, max_ticks=80)
    envs = [TerritoryEnvironment(game, ExperimentConfig(o, r)) for o in ("independent", "sequential", "feedback") for r in ("individual", "team")]
    for e in envs:
        e.reset(77)
    rng = np.random.default_rng(9)
    for _ in range(80):
        actions = {a: int(rng.integers(1, 10)) for a in AGENT_IDS}
        results = [e.step(dict(reversed(list(actions.items()))) if i % 2 else actions) for i, e in enumerate(envs)]
        for e, result in zip(envs[1:], results[1:]):
            np.testing.assert_array_equal(e.snapshot().owner, envs[0].snapshot().owner)
            assert e.snapshot().agents == envs[0].snapshot().agents
            assert result.events == results[0].events
        if results[0].terminated:
            break


def test_snapshot_and_observation_do_not_alias_internal_state():
    """반환된 스냅샷·관측 배열을 바꾸어도 내부 맵과 행동 mask가 손상되지 않는지 검증한다."""
    env = scenario()
    s = env.snapshot()
    with pytest.raises(ValueError):
        s.owner[0, 0] = 1
    s.owner.flags.writeable = True
    s.owner[:] = 1
    obs = env.observe("red", 0)
    obs["owner"][:] = 9
    obs["action_mask"][:] = 0
    assert env.snapshot().owner[0, 0] == 0
    assert env.observe("red", 0)["owner"].max() <= 1
    assert env.action_mask("red_0")[0] == 1


def test_full_map_relative_team_and_positions():
    """양 팀이 동일 물리 맵을 반대 부호로 보고 모든 위치를 아군 우선 순서로 받는다."""
    env = scenario()
    red, blue = env.observe("red", 0), env.observe("blue", 2)
    np.testing.assert_array_equal(red["owner"], -blue["owner"])
    assert set(np.unique(red["owner"])) == {-1, 1}
    np.testing.assert_array_equal(red["agent_positions"][:3], blue["agent_positions"][3:])
    np.testing.assert_array_equal(red["areas"], blue["areas"][::-1])
    assert blue["self_index"] == 2 and blue["self_team"] == 1


def test_dead_observation_keeps_full_map_and_unknown_current_actions():
    """사망자도 전체 맵을 받고 과거 실행 행동은 현재 행동 관측에 누출되지 않는다."""
    env = scenario({"red_0": (2, 2)})
    env.step({"red_0": Action.RIGHT, "red_1": Action.ATTACK_RIGHT})
    dead = env.observe("red", 0)
    assert not dead["agent_alive"][0]
    assert dead["owner"].shape == (6, 6)
    assert tuple(dead["agent_positions"][0]) == (-1, -1)
    assert not dead["team_action_codes"].any()
    np.testing.assert_array_equal(dead["action_mask"], [1, 0, 0, 0, 0, 0, 0, 0, 0])


def test_sessions_independent_duplicate_and_stale_tokens():
    """개별 전략의 제출 비공개와 동일 회차 중복·이전 세션 사용 거부를 검증한다."""
    env = scenario()
    env.begin_decision()
    red, teammate = env.agent_session("red", 0), env.agent_session("red", 1)
    before = teammate.observe()
    red.submit_action(Action.ATTACK_RIGHT)
    for key, value in before.items():
        np.testing.assert_array_equal(value, teammate.observe()[key])
    with pytest.raises(ValueError):
        red.submit_action(Action.NOOP)
    env.finish_decision_round()
    with pytest.raises(RuntimeError):
        red.observe()
    env.begin_decision()
    stale = env.agent_session("red", 0)
    env.reset(2)
    with pytest.raises(RuntimeError):
        stale.submit_action(Action.NOOP)


@pytest.mark.parametrize("key,expected", [("w", Action.UP), ("S", Action.DOWN), ("a", Action.LEFT),
    ("d", Action.RIGHT), ("Up", Action.ATTACK_UP), ("Down", Action.ATTACK_DOWN),
    ("Left", Action.ATTACK_LEFT), ("Right", Action.ATTACK_RIGHT), ("space", Action.NOOP)])
def test_keyboard_mapping_and_one_shot_consumption(key, expected):
    """키→Action 매핑, 대소문자 처리, 예약 행동의 한 번 소비 및 미지원 키 무시를 검증한다."""
    keyboard = KeyboardController()
    assert keyboard.feed_key(key)
    assert keyboard.act({}) == expected
    assert keyboard.act({}) == Action.NOOP
    assert not keyboard.feed_key("Escape")


def test_keyboard_connects_through_manager_without_advancing_on_input():
    """키 입력은 예약만 하고 manager가 전이를 수행하며, 최신 키 우선과 cooldown 거부 기록이 연결되는지 검증한다."""
    keyboard = KeyboardController()
    env = scenario({"red_0": (2, 2)})
    controllers = {a: NoopController() for a in AGENT_IDS}
    controllers["red_0"] = keyboard
    manager = EpisodeManager(env, controllers)
    keyboard.feed_key("d")
    keyboard.feed_key("right")  # Latest staged key wins before collection.
    assert env.tick == 0
    r = manager.step()
    assert env.tick == 1 and r.credits["red_0"] == 1
    keyboard.feed_key("right")
    r = manager.step()
    assert any(e.kind == "rejected" and e.agent_id == "red_0" for e in r.events)
    assert keyboard.pending is None


def test_render_is_state_and_rng_independent():
    """RGB 이미지 형식과 렌더링·반환 이미지 수정이 이후 상태/RNG 결과에 영향 없는지 검증한다."""
    a, b = scenario(), scenario()
    frame = render_rgb(a.snapshot())
    assert frame.shape == (144, 144, 3) and frame.dtype == np.uint8
    frame[:] = 0
    for _ in range(10):
        render_rgb(a.snapshot())
        assert a.step({}).to_dict() == b.step({}).to_dict()


@pytest.mark.parametrize("style", ["expand", "aggressive", "defend"])
def test_scripted_controllers_only_use_observations_and_valid_actions(style):
    """세 scripted 스타일이 관측 사전만 받아 mask 안의 행동을 선택하며 경기를 진행할 수 있는지 검증한다."""
    env = TerritoryEnvironment(GameConfig(height=6, width=8, max_ticks=40))
    env.reset(9)
    bots = {a: FrontierController(i, style) for i, a in enumerate(AGENT_IDS)}
    for _ in range(40):
        obs = {a: env.observe(a.rsplit("_", 1)[0], int(a[-1])) for a in AGENT_IDS}
        actions = {a: bots[a].act(obs[a]) for a in AGENT_IDS}
        assert all(obs[a]["action_mask"][v.index] for a, v in actions.items())
        if env.step(actions).terminated:
            break


def test_tk_controls_step_reset_selection_and_cleanup():
    """숨긴 Tk 창에서 키 처리·한 틱 진행·선택·reset·타이머·종료 자원 정리를 검증한다.
    실제 키 주입 대신 핸들러에 이벤트 객체를 전달하는 UI 통합 검사다.
    """
    import tkinter as tk
    from territory_rl.keyboard_demo import KeyboardDemo
    root = tk.Tk()
    root.withdraw()
    app = KeyboardDemo(GameConfig(height=6, width=8, max_ticks=3), root=root, manual_all=True)
    try:
        root.update_idletasks()
        app.on_key(SimpleNamespace(keysym="Right"))
        assert app.env.tick == 0
        app.on_key(SimpleNamespace(keysym="Return"))
        assert app.env.tick == 1
        assert app.last_result.credits["red_0"] == 1
        app.on_key(SimpleNamespace(keysym="2"))
        assert app.selected == "red_1"
        app.on_key(SimpleNamespace(keysym="w"))
        app.on_key(SimpleNamespace(keysym="r"))
        assert app.env.tick == 0 and app.keyboards["red_1"].pending is None
        app.toggle()
        assert app.running and app.after_id is not None
        app.pause()
        assert not app.running and app.after_id is None
        app.step(); app.step(); app.step(); app.step()
        assert app.env.tick == 3 and app.env.terminated
    finally:
        app.close()
    assert app.closed
