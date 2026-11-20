"""중앙 정렬, 변경 가능한 타일 값, 공통 회전과 실제 실행의 일치를 검사한다."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from territory_rl import Action, ExperimentConfig, GameConfig, TerritoryEnvironment, TileEncoding
from territory_rl.agent import AgentState
from territory_rl.cli import main, replay
from territory_rl.controller import FrontierController
from territory_rl.models import AGENT_IDS
from territory_rl.observation import rotate_action
from territory_rl.parallel_env import TerritoryParallelEnv
from test_rules import agent, scenario


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("position", [(0, 0), (0, 7), (7, 0), (7, 7), (3, 4)])
def test_whole_map_and_agent_identity_at_edges(rotation, position):
    env = TerritoryParallelEnv(experiment=ExperimentConfig(randomize_observation_rotation=True))
    env.reset(7)
    snapshot = env.core.snapshot()
    states = [AgentState(a.agent_id, a.team, a.position) for a in snapshot.agents]
    states[0].position = position
    owner = snapshot.owner.copy()
    owner[position] = 0
    env.reset(7, {"owner": owner, "agents": states, "observation_rotation": rotation})
    encoding = env.core.experiment.tile_encoding
    observations = {aid: env._observe(aid) for aid in AGENT_IDS}
    for aid, obs in observations.items():
        assert env.observation_space(aid).contains(obs)
        assert obs["tilemap"][7, 7] == encoding.friendly_agents[int(aid[-1])]
        assert np.count_nonzero(obs["tilemap"] != encoding.outside) == owner.size
        viewer = next(a for a in states if a.agent_id == aid)
        restored = np.rot90(obs["tilemap"], rotation)
        for target in states:
            location = (7 + target.position[0] - viewer.position[0],
                        7 + target.position[1] - viewer.position[1])
            codes = encoding.friendly_agents if target.team == viewer.team else encoding.enemy_agents
            assert restored[location] == codes[int(target.agent_id[-1])]


def test_custom_tile_values_roundtrip_spaces_and_scripted_policy():
    # 부호/연속성에 의존하는 하드코딩을 잡기 위해 임의의 서로 다른 값을 쓴다.
    encoding = TileEncoding(outside=90, friendly=10, enemy=20,
                            friendly_agents=(11, 13, 17), enemy_agents=(21, 23, 27))
    experiment = ExperimentConfig(tile_encoding=encoding, randomize_observation_rotation=True)
    restored = ExperimentConfig(**json.loads(json.dumps(asdict(experiment))))
    assert restored == experiment
    env = TerritoryParallelEnv(GameConfig(max_ticks=6), restored)
    observations, _ = env.reset(12, {"observation_rotation": 1})
    controllers = {aid: FrontierController(i, tile_encoding=encoding) for i, aid in enumerate(AGENT_IDS)}
    for _ in range(6):
        actions = {}
        for aid, obs in observations.items():
            assert env.observation_space(aid).contains(obs)
            assert set(np.unique(obs["tilemap"])) <= set(encoding.values)
            action = controllers[aid].act(obs)
            assert obs["auxiliary"]["action_mask"][action.index]
            actions[aid] = action.index
        observations, _, terminated, _, _ = env.step(actions)
        if all(terminated.values()):
            break


@pytest.mark.parametrize("values", [
    {"outside": 1}, {"friendly_agents": (2, 2, 4)}, {"friendly": 128},
    {"enemy": -129}, {"outside": True}, {"enemy_agents": [5, 6]},
])
def test_invalid_tile_encoding_is_rejected(values):
    with pytest.raises(ValueError):
        TileEncoding(**values)


@pytest.mark.parametrize("size", [0, -1, 14, True, 15.0])
def test_invalid_window_size_is_rejected(size):
    with pytest.raises(ValueError, match="window"):
        ExperimentConfig(observation_window_size=size)


def test_larger_maps_require_sufficient_window():
    game = GameConfig(height=8, width=10)
    with pytest.raises(ValueError, match="whole map"):
        TerritoryEnvironment(game)
    env = TerritoryEnvironment(game, ExperimentConfig(observation_window_size=19))
    env.reset(1)
    assert env.observe("red", 0)["tilemap"].shape == (19, 19)


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("action", list(Action))
def test_mask_and_action_target_match_the_rotated_grid(rotation, action):
    env = TerritoryParallelEnv(GameConfig(max_ticks=4), ExperimentConfig(randomize_observation_rotation=True))
    env.reset(7, {"observation_rotation": rotation})
    observed_action = env.core.action_to_observation(action)
    assert env.core.action_to_world(observed_action) == action
    obs = env.core.observe("red", 0)
    assert obs["auxiliary"]["action_mask"][observed_action.index] == env.core.action_mask("red_0")[action.index]
    before = env.core.snapshot().agents[0].position
    commands = dict.fromkeys(AGENT_IDS, Action.NOOP.index)
    commands["red_0"] = observed_action.index
    result = env.step(commands)
    if action != Action.NOOP:
        target = action.target(before)
        # 방향 변환의 목표를 격자 회전으로 독립적으로 확인한다.
        world_grid = np.zeros((15, 15), dtype=np.int8)
        delta = action.target((7, 7))
        world_grid[delta] = 1
        observed_target = tuple(np.argwhere(np.rot90(world_grid, -rotation) == 1)[0])
        assert observed_action.target((7, 7)) == observed_target
        matching = [e for e in env.last_result.events if e.agent_id == "red_0" and e.kind in ("move", "attack")]
        if matching:
            assert matching[0].position == target
    assert env.core.snapshot().agents[0].last_action == (
        action if obs["auxiliary"]["action_mask"][observed_action.index] else Action.NOOP)
    assert len(result[0]) == 6


@pytest.mark.parametrize("strategy", ["independent", "sequential", "feedback"])
@pytest.mark.parametrize("rotation", range(4))
def test_rotated_sessions_preserve_world_transition_and_shared_actions(strategy, rotation):
    original = scenario({"red_0": (2, 2)}, delay=0)
    snapshot = original.snapshot()
    states = [AgentState(a.agent_id, a.team, a.position) for a in snapshot.agents]
    env = TerritoryEnvironment(original.config, ExperimentConfig(
        decision_strategy=strategy, randomize_observation_rotation=True))
    env.reset(123, {"owner": snapshot.owner, "agents": states, "observation_rotation": rotation})
    physical_rng = deepcopy(env._rng.bit_generator.state)
    world_actions = dict.fromkeys(AGENT_IDS, Action.NOOP)
    world_actions["red_0"] = Action.ATTACK_RIGHT
    env.begin_decision()
    result = None
    while result is None:
        for aid in AGENT_IDS:
            team, number = aid.rsplit("_", 1)
            session = env.agent_session(team, int(number))
            obs = session.observe()
            assert obs["auxiliary"]["decision_round"] in (1, 2)
            if (strategy == "sequential" and aid == "red_1") or (
                    strategy == "feedback" and env.decision_round == 2 and team == "red"):
                assert obs["team_actions"][0] == rotate_action(Action.ATTACK_RIGHT, rotation)
            session.submit_action(rotate_action(world_actions[aid], rotation))
        assert env._rng.bit_generator.state == physical_rng
        result = env.finish_decision_round()
    assert result.to_dict() == original.step(world_actions).to_dict()
    assert env.snapshot().agents == original.snapshot().agents
    for entry in env.last_decision:
        assert entry["requested"]["red_0"] == Action.ATTACK_RIGHT
        assert entry["observation_requested"]["red_0"] == rotate_action(Action.ATTACK_RIGHT, rotation)


def test_rotation_rng_is_independent_and_reset_is_atomic():
    plain = TerritoryEnvironment()
    rotated = TerritoryEnvironment(experiment=ExperimentConfig(randomize_observation_rotation=True))
    for env in (plain, rotated):
        env.reset(21)
    angle = rotated.observation_rotation
    assert plain.snapshot().agents == rotated.snapshot().agents
    assert plain._rng.bit_generator.state == rotated._rng.bit_generator.state
    before = rotated.snapshot()
    for invalid in (-1, 4, True, "1"):
        with pytest.raises(ValueError, match="rotation"):
            rotated.reset(22, {"observation_rotation": invalid})
        assert rotated.snapshot().agents == before.agents and rotated.observation_rotation == angle
    rng = np.random.default_rng(9)
    for _ in range(30):
        actions = {aid: int(rng.integers(1, 10)) for aid in AGENT_IDS}
        assert plain.step(actions).to_dict() == rotated.step(actions).to_dict()
        assert plain.snapshot().agents == rotated.snapshot().agents
        assert rotated.observation_rotation == angle
    rotated.reset(21)
    assert rotated.observation_rotation == angle


@pytest.mark.parametrize("phase", ["attack", "movement"])
def test_death_retains_position_and_releases_occupancy(phase):
    env = scenario({"red_0": (2, 2), "blue_0": (2, 3), "blue_1": (1, 2)}, delay=2)
    actions = ({"red_0": Action.LEFT, "blue_0": Action.ATTACK_LEFT, "blue_1": Action.DOWN}
               if phase == "attack" else {"red_0": Action.RIGHT, "blue_0": Action.RIGHT})
    env.step(actions)
    expected = (2, 2) if phase == "attack" else (2, 3)
    assert agent(env, "red_0").position == expected and not agent(env, "red_0").alive
    obs = env.observe("red", 0)
    encoding = env.experiment.tile_encoding
    assert encoding.friendly_agents[0] not in obs["tilemap"]
    assert not obs["auxiliary"]["agent_alive"][0]
    if phase == "attack":
        assert agent(env, "blue_1").position == expected
        assert obs["tilemap"][7, 7] == encoding.enemy_agents[1]
    for _ in range(2):
        env.step({})
    assert agent(env, "red_0").alive
    assert env.observe("red", 0)["tilemap"][7, 7] == encoding.friendly_agents[0]


@pytest.mark.parametrize("strategy", ["independent", "sequential", "feedback"])
def test_rotated_cli_logs_replay_both_action_frames(strategy, tmp_path, capsys):
    path = tmp_path / "rotated.jsonl"
    main(["--mode", "headless", "--strategy", strategy, "--ticks", "8", "--seed", "7",
          "--randomize-observation-rotation", "--log", str(path)])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["observation_rotation"] in range(4)
    replay(path)
    assert '"verified": true' in capsys.readouterr().out
    records[1]["decision"][0]["observation_requested"]["red_0"] = 0
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError):
        replay(path)


@pytest.mark.parametrize("rotation", [1, 2, 3])
def test_rotated_keyboard_view_execution_and_log(rotation, tmp_path, capsys):
    import tkinter as tk
    from territory_rl.keyboard_demo import KeyboardDemo

    root = tk.Tk()
    root.withdraw()
    path = tmp_path / "keyboard.jsonl"
    app = KeyboardDemo(GameConfig(height=6, width=8, max_ticks=3),
                       ExperimentConfig(randomize_observation_rotation=True),
                       seed=7, manual_all=True, root=root, log_path=path)
    try:
        app.env.reset(7, {"observation_rotation": rotation})
        app._write_reset()
        app.refresh()
        root.update_idletasks()
        marker = np.zeros((6, 8), dtype=np.int8)
        marker[app.env.snapshot().agents[0].position] = 1
        row, column = np.argwhere(np.rot90(marker, -rotation) == 1)[0]
        size = app.renderer.tile_size
        oval = next(item for item in app.renderer.canvas.find_all() if app.renderer.canvas.type(item) == "oval")
        assert app.renderer.canvas.coords(oval) == [column*size+5, row*size+5,
                                                   (column+1)*size-5, (row+1)*size-5]
        observed_action = app.env.action_to_observation(Action.ATTACK_RIGHT)
        key = next(key for key, action in app.keyboards["red_0"].keymap.items() if action == observed_action)
        app.on_key(SimpleNamespace(keysym=key))
        app.step()
        assert app.last_result.credits["red_0"] == 1
        assert app.env.last_decision[0]["requested"]["red_0"] == Action.ATTACK_RIGHT
    finally:
        app.close()
    replay(path)
    assert '"verified": true' in capsys.readouterr().out
