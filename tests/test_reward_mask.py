"""면적·상대 팀 격차·종료 보상의 마스킹을 검사한다. 학습은 실행하지 않는다."""
from dataclasses import FrozenInstanceError, asdict
from itertools import product
import json

import numpy as np
import pytest

from territory_rl import Action, ExperimentConfig, GameConfig, TerritoryEnvironment
from territory_rl.cli import main, replay
from territory_rl.models import AGENT_IDS
from territory_rl.parallel_env import TerritoryParallelEnv
from test_rules import scenario


@pytest.mark.parametrize("mask", list(product((0, 1), repeat=3)))
def test_all_masks_gate_exact_terms_at_each_tick(mask):
    """8개 마스크 조합에서 각 보상 항과 최종 합계가 중간/종료 틱 모두 정확한지 확인한다."""
    config = ExperimentConfig(reward_signal_mask=mask, area_weight=2, gap_weight=3,
                              win_reward=7, loss_reward=-4, team_signal="terminal",
                              include_terminal_reward=True)
    env = scenario({"red_0": (2, 2)}, ticks=2, experiment=config)
    results = [env.step({"red_0": Action.ATTACK_RIGHT}), env.step({})]
    for tick, result in enumerate(results, 1):
        for team, ids in enumerate((AGENT_IDS[:3], AGENT_IDS[3:])):
            expected = {
                "capture": 0.,
                "area": 2 * result.area_after[team] if mask[0] else 0.,
                "gap": (6 if team == 0 else -6) if mask[1] else 0.,
                "terminal": (7 if team == 0 else -4) if mask[2] and tick == 2 else 0.,
            }
            for aid in ids:
                assert dict(result.reward_components[aid]) == pytest.approx(expected)
                assert result.rewards[aid] == pytest.approx(sum(expected.values()))


def test_gap_uses_current_opponent_difference_not_tick_delta():
    """격차를 유지한 틱에도 같은 보상을 주고 동률이면 양 팀 모두 0이다."""
    env = scenario({"red_0": (2, 2)}, experiment=ExperimentConfig(team_signal="gap"))
    assert env.experiment.resolved_reward_mask == (0, 1, 0)
    first = env.step({"red_0": Action.ATTACK_RIGHT})
    second = env.step({})
    assert first.area_after == second.area_after == (19, 17)
    assert first.rewards["red_0"] == second.rewards["red_0"] == pytest.approx(2)
    assert second.rewards["blue_0"] == pytest.approx(-2)
    assert sum(second.rewards.values()) == pytest.approx(0)
    tie = scenario(experiment=ExperimentConfig(reward_signal_mask=(0, 1, 0))).step({})
    assert not any(tie.rewards.values())


def test_gap_and_terminal_both_apply_on_early_elimination():
    """T 이전 마지막 타일 점령 시 격차 항과 마스크로 켠 종료 항을 정확히 한 번 지급한다."""
    env = scenario({"red_0": (2, 2), "blue_0": (2, 3), "blue_1": None, "blue_2": None},
                   owner=np.zeros((6, 6)), delay=0,
                   experiment=ExperimentConfig(reward_signal_mask=(0, 1, 1), win_reward=5, loss_reward=-2))
    result = env.step({"red_0": Action.ATTACK_RIGHT})
    assert result.tick == 1 and result.terminated
    assert result.rewards["red_0"] == 36 + 5
    assert result.rewards["blue_0"] == -36 - 2
    with pytest.raises(RuntimeError):
        env.step({})


def test_gap_shared_with_dead_agents_and_independent_of_personal_credit():
    """사망 팀원도 동일 격차 보상을 받고 개인 모드의 점령 항은 마스크와 독립적이다."""
    env = scenario({"red_0": (2, 2), "red_2": None},
                   experiment=ExperimentConfig(reward_signal_mask=(0, 1, 0)))
    r = env.step({"red_0": Action.ATTACK_RIGHT})
    assert r.rewards["red_2"] == r.rewards["red_0"] == pytest.approx(2)
    for mask, extra in [((0, 0, 0), 0), ((0, 1, 0), 2)]:
        env = scenario({"red_0": (2, 2)}, experiment=ExperimentConfig(
            reward_mode="individual", capture_weight=4, reward_signal_mask=mask))
        r = env.step({"red_0": Action.ATTACK_RIGHT})
        assert r.rewards["red_0"] == pytest.approx(4 + extra)
        assert r.rewards["red_1"] == pytest.approx(extra)
    assert ExperimentConfig(reward_mode="individual").resolved_reward_mask == (0, 0, 0)


@pytest.mark.parametrize("mask", [(), (1, 0), (1, 0, 0, 0), (2, 0, 0), (-1, 0, 0),
                                    (1.0, 0, 0), ("1", 0, 0), "010", {"gap": 1}])
def test_invalid_masks_are_rejected(mask):
    """길이와 이진 값이 잘못된 마스크를 환경 시작 전에 거부한다."""
    with pytest.raises(ValueError, match="mask"):
        ExperimentConfig(reward_signal_mask=mask)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_invalid_gap_weights_are_rejected(weight):
    """격차 가중치는 유한한 0 이상 값만 허용한다."""
    with pytest.raises(ValueError):
        ExperimentConfig(gap_weight=weight)


def test_mask_is_frozen_and_can_change_only_for_new_episode():
    """입력 리스트 수정으로 현재 설정을 바꿀 수 없고 reset에서만 새 마스크를 적용한다."""
    values = [0, 1, 0]
    config = ExperimentConfig(reward_signal_mask=values)
    values[:] = [1, 0, 0]
    assert config.resolved_reward_mask == (0, 1, 0)
    with pytest.raises(FrozenInstanceError):
        config.reward_signal_mask = (1, 0, 0)
    env = TerritoryParallelEnv(experiment=config)
    env.reset(1, {"reward_signal_mask": [0, 0, 1]})
    assert env.core.experiment.resolved_reward_mask == (0, 0, 1)
    before = env.core.snapshot()
    with pytest.raises(ValueError):
        env.reset(2, {"reward_signal_mask": [3, 0, 0]})
    assert env.core.snapshot().agents == before.agents
    assert env.core.experiment.resolved_reward_mask == (0, 0, 1)


def test_reward_masks_do_not_change_transitions():
    """같은 seed·행동에서 마스크를 바꾸어도 맵·사망·부활·RNG 전이는 동일하다."""
    envs = [scenario({"red_0": (2, 2)}, delay=0,
                     experiment=ExperimentConfig(reward_signal_mask=mask))
            for mask in product((0, 1), repeat=3)]
    for actions in ({"red_0": Action.RIGHT}, {}, {"blue_0": Action.ATTACK_LEFT}):
        results = [env.step(actions) for env in envs]
        for env, result in zip(envs[1:], results[1:]):
            assert result.events == results[0].events
            np.testing.assert_array_equal(env.snapshot().owner, envs[0].snapshot().owner)
            assert env.snapshot().agents == envs[0].snapshot().agents


def test_cli_mask_overrides_team_signal_and_roundtrips(tmp_path, capsys):
    """CLI/JSONL에서 마스크·가중치가 보존되고 종료 활성 옵션을 마스크가 덮어쓴다."""
    path = tmp_path / "mask.jsonl"
    main(["--mode", "headless", "--ticks", "3", "--strategy", "feedback",
          "--reward-mask", "0", "1", "0", "--gap-weight", "2",
          "--team-signal", "terminal", "--include-terminal-reward", "--log", str(path)])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["experiment"]["reward_signal_mask"] == [0, 1, 0]
    assert records[0].keys() == {"type", "seed", "game", "experiment", "observation_rotation"}
    for record in records[1:]:
        for parts in record["reward_components"].values():
            assert parts["area"] == parts["terminal"] == 0
    replay(path)
    assert '"verified": true' in capsys.readouterr().out


def test_replay_verifies_capture_rewards_and_credits(tmp_path, capsys):
    """공동 점령 로그를 현재 공식으로 검증하고 다른 보상 값은 거부한다."""
    game = GameConfig(max_ticks=3)
    experiment = ExperimentConfig(reward_mode="individual", capture_weight=4)
    env = TerritoryEnvironment(game, experiment)
    env.reset(seed=7)
    records = [{"type": "reset", "seed": 7, "game": asdict(game), "experiment": asdict(experiment),
                "observation_rotation": env.observation_rotation}]
    for actions in ({"red_1": Action.ATTACK_RIGHT}, {"red_1": Action.RIGHT},
                    {"red_0": Action.ATTACK_RIGHT, "red_1": Action.ATTACK_UP}):
        record = env.step(actions).to_dict()
        record.update(type="step", actions=actions)
        records.append(record)
    shared_capture = records[-1]
    for aid in ("red_0", "red_1"):
        assert shared_capture["credits"][aid] == 1
        assert shared_capture["rewards"][aid] == 4

    path = tmp_path / "shared_capture.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    replay(path)
    assert '"verified": true' in capsys.readouterr().out

    for aid in ("red_0", "red_1"):
        shared_capture["credits"][aid] = 0.5
        shared_capture["reward_components"][aid]["capture"] = 2
        shared_capture["rewards"][aid] = 2
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="replay mismatch at tick 3"):
        replay(path)
