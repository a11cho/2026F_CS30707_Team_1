"""새 보상·관측·결정 프로토콜의 환경 기능 검사. 학습기나 gradient를 실행하지 않는다."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from territory_rl import Action, ExperimentConfig, GameConfig, TerritoryEnvironment
from territory_rl.cli import main, replay
from territory_rl.controller import NoopController
from territory_rl.keyboard import KeyboardController
from territory_rl.manager import EpisodeManager
from territory_rl.models import AGENT_IDS
from territory_rl.parallel_env import TerritoryParallelEnv
from test_rules import scenario, agent


def session(env, aid):
    """테스트의 내부 ID를 공개 팀·번호 API로 변환한다."""
    team, number = aid.rsplit("_", 1)
    return env.agent_session(team, int(number))


def submit_round(env, actions=None, order=AGENT_IDS):
    """회차의 모든 행동을 지정 순서로 제출한다. 회차 마감은 호출자가 수행한다."""
    actions = actions or {}
    for aid in order:
        session(env, aid).submit_action(actions.get(aid, Action.NOOP))


def test_defaults_and_zero_delay_death_respawn_events():
    """기본값과 사망 틱 후반의 부활을 확인한다. 부활 후 같은 틱에 다시 이동하지 않는다."""
    cfg = GameConfig()
    assert (cfg.height, cfg.width, cfg.max_ticks, cfg.respawn_delay) == (8, 8, 128, 0)
    env = scenario({"red_0": (2, 2)}, delay=cfg.respawn_delay)
    r = env.step({"red_0": Action.RIGHT})
    death = next(e for e in r.events if e.kind == "death")
    spawn = next(e for e in r.events if e.kind == "respawn")
    assert death.due_tick == spawn.due_tick == r.tick == 1
    assert agent(env, "red_0").position == spawn.position
    assert len([e for e in r.events if e.kind == "move" and e.agent_id == "red_0"]) == 1
    assert agent(env, "red_0").alive


@pytest.mark.parametrize("bad", [0, -1, 10, True, 1.5])
def test_action_codes_and_unknown_rejection(bad):
    """UNKNOWN 및 잘못된 코드는 실행하지 않고 유효 코드와 선택 인덱스는 왕복 변환한다."""
    for index in range(9):
        assert Action.from_index(index).index == index
        assert int(Action.from_index(index)) == index+1
    with pytest.raises(ValueError):
        Action.parse(bad)
    with pytest.raises(ValueError):
        Action.from_index(9)
    with pytest.raises(ValueError):
        Action.NOOP.target((0, 0))


@pytest.mark.parametrize("team,number", [("green", 0), ("red", -1), ("blue", 3), ("red", True), ("red", 1.0)])
def test_observation_identity_validation(team, number):
    """잘못된 팀·번호 요청을 거부한다."""
    env = scenario()
    with pytest.raises(ValueError):
        env.observe(team, number)


def test_sequential_visibility_and_normalized_noop():
    """앞선 확정 행동만 공개하고 경계 밖 요청은 알려진 정지로 구분한다."""
    env = scenario(experiment=ExperimentConfig(decision_strategy="sequential"))
    env.begin_decision()
    first = session(env, "red_0")
    initial = first.observe()
    assert initial["team_actions"].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="order"):
        session(env, "red_1").submit_action(Action.NOOP)
    first.submit_action(Action.UP)  # red_0 is at row 0: normalize to NOOP.
    second = session(env, "red_1").observe()
    assert second["team_actions"].tolist() == [1, 0, 0]
    assert second["auxiliary"]["team_actions_known"].tolist() == [1, 0, 0]
    assert second["auxiliary"]["team_action_features"][0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert initial["team_actions"].tolist() == [0, 0, 0]
    session(env, "red_1").submit_action(Action.ATTACK_RIGHT)
    third = session(env, "red_2").observe()
    assert third["team_actions"].tolist() == [1, 9, 0]
    assert not env.observe("blue", 0)["auxiliary"]["team_actions_known"].any()
    result = env.finish_decision_round()  # Missing decisions are filled with NOOP.
    assert any(e.kind == "rejected" and e.requested == 2 and e.applied == 1 for e in result.events)
    assert not env.observe("red", 1)["auxiliary"]["team_actions_known"].any()


def test_feedback_freezes_first_proposals_and_executes_only_final_actions():
    """되먹임의 첫 제안·물리 상태·RNG가 고정되고 두 번째 행동만 실행됨을 검증한다."""
    env = scenario({"red_0": (2, 2)}, experiment=ExperimentConfig(decision_strategy="feedback", reward_mode="individual"))
    before, rng = env.snapshot(), deepcopy(env._rng.bit_generator.state)
    env.begin_decision()
    stale = session(env, "red_0")
    stale.submit_action(Action.ATTACK_RIGHT)
    assert not env.observe("red", 1)["auxiliary"]["team_actions_known"].any()
    for aid in AGENT_IDS[1:]:
        session(env, aid).submit_action(Action.NOOP)
    assert env.finish_decision_round() is None
    assert env.tick == 0 and env.decision_round == 2
    assert env._rng.bit_generator.state == rng
    assert env.snapshot().agents == before.agents
    np.testing.assert_array_equal(env.snapshot().owner, before.owner)
    with pytest.raises(RuntimeError):
        stale.submit_action(Action.NOOP)
    frozen = env.observe("red", 1)
    assert frozen["team_actions"].tolist() == [9, 1, 1]
    assert frozen["auxiliary"]["team_actions_known"].tolist() == [1, 1, 1]
    assert frozen["auxiliary"]["action_source"] == "first_pass_proposal"
    session(env, "red_0").submit_action(Action.NOOP)
    np.testing.assert_array_equal(env.observe("red", 1)["team_actions"], frozen["team_actions"])
    with pytest.raises(ValueError):
        session(env, "red_0").submit_action(Action.NOOP)
    result = env.finish_decision_round()
    assert result.tick == 1 and not any(e.kind == "attack" for e in result.events)
    assert sum(result.rewards.values()) == 0
    assert agent(env, "red_0").last_attack_tick == before.agents[0].last_attack_tick
    assert len(env.last_decision) == 2
    assert env.decision_round == 0 and not env.observe("red", 0)["team_actions"].any()


def test_feedback_final_submission_order_does_not_change_observations_or_transition():
    """2차 제출 순서가 달라도 관측과 동일 최종 행동의 전이는 일치한다."""
    envs = [scenario(experiment=ExperimentConfig(decision_strategy="feedback")) for _ in range(2)]
    results = []
    for env, order in zip(envs, (AGENT_IDS, tuple(reversed(AGENT_IDS)))):
        env.begin_decision()
        submit_round(env)
        env.finish_decision_round()
        expected = env.observe("red", 0)["team_actions"].copy()
        for aid in order:
            session(env, aid).submit_action(Action.RIGHT if aid == "red_0" else Action.NOOP)
            np.testing.assert_array_equal(env.observe("red", 0)["team_actions"], expected)
        results.append(env.finish_decision_round().to_dict())
    assert results[0] == results[1]
    assert envs[0].snapshot().agents == envs[1].snapshot().agents


def test_reset_clears_rounds_and_strategy_can_only_change_at_reset():
    """되먹임 중 reset은 버퍼·이전 세션을 무효화하고 새 전략을 저장한다."""
    env = scenario(experiment=ExperimentConfig(decision_strategy="feedback"))
    env.begin_decision()
    submit_round(env)
    env.finish_decision_round()
    stale = session(env, "red_0")
    with pytest.raises(AttributeError):
        env.decision_strategy = "sequential"
    with pytest.raises(RuntimeError):
        env.step({})
    with pytest.raises(RuntimeError):
        env.begin_decision()
    env.reset(4, {"decision_strategy": "sequential"})
    assert env.decision_strategy == "sequential" and env.decision_round == 0
    assert not env.observe("red", 0)["auxiliary"]["team_actions_known"].any()
    with pytest.raises(RuntimeError):
        stale.observe()


@pytest.mark.parametrize("strategy", ["sequential", "feedback"])
def test_parallel_adapter_cannot_bypass_decision_strategy(strategy):
    """전략 adapter를 일괄 입력으로 우회할 수 없고 명시 수집은 표준 반환 형식으로 종료한다."""
    env = TerritoryParallelEnv(GameConfig(max_ticks=1), ExperimentConfig(decision_strategy=strategy))
    env.reset(1)
    with pytest.raises(RuntimeError, match="requires"):
        env.step(dict.fromkeys(AGENT_IDS, 0))
    env.begin_decision()
    result = env.finish_decision_round()
    if result is None:
        result = env.finish_decision_round()
    observations, rewards, term, trunc, infos = result
    assert set(observations) == set(rewards) == set(AGENT_IDS)
    assert all(term.values()) and not any(trunc.values()) and env.agents == []
    assert infos["red_0"]["termination_reasons"] == ("time_limit",)


@pytest.mark.parametrize("signal", ["area", "terminal"])
def test_team_signals_do_not_mix_implicitly(signal):
    """area는 현재 면적을 매 틱 지급하고 terminal은 마지막에만 승패값을 지급한다."""
    cfg = ExperimentConfig(team_signal=signal, area_weight=2, win_reward=7, loss_reward=-3, draw_reward=.5)
    env = scenario({"red_0": (2, 2)}, ticks=2, experiment=cfg)
    first = env.step({"red_0": Action.ATTACK_RIGHT})
    final = env.step({})
    if signal == "area":
        assert first.rewards["red_0"] == final.rewards["red_0"] == 2 * 19
        assert final.reward_components["red_0"]["terminal"] == 0
    else:
        assert not any(first.rewards.values())
        assert final.rewards["red_0"] == 7 and final.rewards["blue_0"] == -3
    for r in (first, final):
        for aid in AGENT_IDS:
            assert sum(r.reward_components[aid].values()) == pytest.approx(r.rewards[aid])
    draw = scenario(ticks=1, experiment=cfg).step({})
    if signal == "terminal":
        assert set(draw.rewards.values()) == {.5}


def test_personal_capture_weight_shared_capture_and_explicit_terminal_mix():
    """공동 점령자마다 가중치 전액을 지급하며 명시적인 종료 보상을 합산한다."""
    positions = {"red_0": (2, 2), "red_1": (3, 3)}
    config = ExperimentConfig(reward_mode="individual", capture_weight=4, win_reward=10)
    env = scenario(positions, ticks=1, experiment=config)
    r = env.step({"red_0": Action.ATTACK_RIGHT, "red_1": Action.ATTACK_UP})
    assert [r.rewards[a] for a in AGENT_IDS[:3]] == [4, 4, 0]
    mixed = scenario(positions, ticks=1, experiment=ExperimentConfig(
        reward_mode="individual", capture_weight=4, win_reward=10, include_terminal_reward=True))
    r = mixed.step({"red_0": Action.ATTACK_RIGHT, "red_1": Action.ATTACK_UP})
    assert [r.rewards[a] for a in AGENT_IDS[:3]] == [14, 14, 10]


def test_elimination_and_time_limit_both_reported_by_adapter():
    """마지막 타일 점령과 T가 겹치면 두 종료 이유 및 여섯 최종 결과를 보존한다."""
    core = scenario({"red_0": (2, 2), "blue_0": (2, 3), "blue_1": None, "blue_2": None},
                    owner=np.zeros((6, 6)), ticks=1, delay=0)
    snapshot = core.snapshot()
    from territory_rl.agent import AgentState
    states = [AgentState(a.agent_id, a.team, a.position, respawn_at=a.respawn_at, alive=a.alive) for a in snapshot.agents]
    env = TerritoryParallelEnv(core.config)
    env.reset(123, {"owner": snapshot.owner, "agents": states})
    actions = dict.fromkeys(AGENT_IDS, Action.NOOP.index)
    actions["red_0"] = Action.ATTACK_RIGHT.index
    obs, _, term, _, info = env.step(actions)
    assert all(term.values()) and len(obs) == 6 and env.agents == []
    assert info["red_0"]["termination_reasons"] == ("time_limit", "team_eliminated")
    assert info["blue_0"]["eliminated_teams"] == ("blue",)


@pytest.mark.parametrize("strategy", ["independent", "sequential", "feedback"])
def test_cli_trace_roundtrip_and_config_overrides(strategy, tmp_path, capsys):
    """실제 CLI를 짧게 실행해 설정 override·회차 로그·재생 일치를 검증한다."""
    path = tmp_path / "trace.jsonl"
    main(["--mode", "headless", "--ticks", "5", "--height", "6", "--width", "10",
          "--respawn-delay", "1", "--strategy", strategy, "--reward", "individual",
          "--observation-window-size", "19", "--log", str(path)])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["game"]["width"] == 10 and records[0]["game"]["respawn_delay"] == 1
    assert records[0]["experiment"]["decision_strategy"] == strategy
    assert len(records[1]["decision"]) == (2 if strategy == "feedback" else 1)
    replay(path)
    assert '"verified": true' in capsys.readouterr().out


def test_cli_defaults_come_from_config(tmp_path, capsys):
    """CLI에서 생략한 크기·딜레이가 GameConfig 기본값과 같음을 기록으로 확인한다."""
    path = tmp_path / "defaults.jsonl"
    main(["--mode", "headless", "--ticks", "1", "--log", str(path)])
    game = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["game"]
    expected = asdict(GameConfig(max_ticks=1))
    assert game == expected
    capsys.readouterr()


@pytest.mark.parametrize("strategy", ["sequential", "feedback"])
def test_keyboard_ui_exposes_decision_stages_and_reset(strategy):
    """숨긴 Tk 화면에서 단계별 Enter·제안 표시·재선택·리셋을 실제 환경과 연결해 확인한다."""
    import tkinter as tk
    from territory_rl.keyboard_demo import KeyboardDemo
    root = tk.Tk()
    root.withdraw()
    app = KeyboardDemo(GameConfig(max_ticks=3), ExperimentConfig(decision_strategy=strategy),
                       manual_all=True, root=root)
    try:
        root.update_idletasks()
        app.on_key(SimpleNamespace(keysym="Right"))
        app.step()
        assert app.env.tick == 0
        if strategy == "feedback":
            assert app.env.decision_round == 2
            assert "first_pass_proposal" in app.status.cget("text")
            # Select NOOP for the second pass: the proposed attack must not execute.
            app.on_key(SimpleNamespace(keysym="space"))
            app.step()
            assert app.env.tick == 1 and not any(e.kind == "attack" for e in app.last_result.events)
        else:
            assert app.selected == "red_1"
            for _ in range(5):
                app.step()
            assert app.env.tick == 1 and app.last_result.credits["red_0"] == 1
            assert app.selected == "red_0"
        app.step()
        app.reset()
        assert app.env.tick == 0 and app.env.decision_round == 0
        assert all(k.pending is None for k in app.keyboards.values())
    finally:
        app.close()
