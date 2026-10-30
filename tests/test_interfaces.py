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
from territory_rl.session import ActionCollector
from territory_rl.visualization import render_rgb
from test_rules import scenario


@pytest.mark.parametrize("obs_mode", ["local", "aware"])
@pytest.mark.parametrize("reward_mode", ["individual", "team"])
def test_spaces_lifecycle_and_invariants(obs_mode, reward_mode):
    env = TerritoryParallelEnv(GameConfig(height=6, width=8, max_ticks=64), ExperimentConfig(obs_mode, reward_mode))
    obs, _ = env.reset(13)
    rng = np.random.default_rng(8)
    for tick in range(1, 65):
        for aid, o in obs.items():
            assert env.observation_space(aid).contains(o)
            if obs_mode == "local":
                assert not o["teammate_features"].any() and not o["teammate_feature_mask"].any()
        actions = {a: int(rng.integers(9)) for a in env.agents}
        obs, rewards, terms, truncs, _ = env.step(actions)
        assert set(obs) == set(rewards) == set(AGENT_IDS)
        assert not any(truncs.values())
        assert all(terms.values()) == (tick == 64)
        assert env.state_space.contains(env.state())
        s = env.core.snapshot()
        alive = [a for a in s.agents if a.alive]
        assert len({a.position for a in alive}) == len(alive)
        assert all(s.owner[a.position] == a.team for a in alive)
        assert sum(s.areas) == 48
    assert not env.agents


def test_official_parallel_contract_and_seed_reproducibility():
    from pettingzoo.test import parallel_api_test, parallel_seed_test
    factory = lambda: TerritoryParallelEnv(GameConfig(height=6, width=8, max_ticks=40))
    parallel_api_test(factory(), num_cycles=80)
    parallel_seed_test(factory, num_cycles=80)


def test_same_trace_ignores_reward_observation_mode_and_submission_order():
    game = GameConfig(height=6, width=8, max_ticks=80)
    envs = [TerritoryEnvironment(game, ExperimentConfig(o, r)) for o in ("local", "aware") for r in ("individual", "team")]
    for e in envs:
        e.reset(77)
    rng = np.random.default_rng(9)
    for _ in range(80):
        actions = {a: int(rng.integers(9)) for a in AGENT_IDS}
        results = [e.step(dict(reversed(list(actions.items()))) if i % 2 else actions) for i, e in enumerate(envs)]
        for e, result in zip(envs[1:], results[1:]):
            np.testing.assert_array_equal(e.snapshot().owner, envs[0].snapshot().owner)
            assert e.snapshot().agents == envs[0].snapshot().agents
            assert result.events == results[0].events


def test_snapshot_and_observation_do_not_alias_internal_state():
    env = scenario()
    s = env.snapshot()
    with pytest.raises(ValueError):
        s.owner[0, 0] = 1
    s.owner.flags.writeable = True
    s.owner[:] = 1
    obs = env.observe("red_0")
    obs["local_grid"][:] = 9
    obs["action_mask"][:] = 0
    assert env.snapshot().owner[0, 0] == 0
    assert env.observe("red_0")["local_grid"].max() <= 1
    assert env.action_mask("red_0")[0] == 1


def test_hidden_changes_do_not_leak_to_local_observation_or_mask():
    first = scenario({"red_0": (0, 0), "red_1": (5, 0)})
    second = scenario({"red_0": (0, 0), "red_1": (5, 1)})
    for key, value in first.observe("red_0").items():
        np.testing.assert_array_equal(value, second.observe("red_0")[key])
    first.experiment = second.experiment = ExperimentConfig(observation_mode="aware")
    assert not np.array_equal(first.observe("red_0")["teammate_features"], second.observe("red_0")["teammate_features"])


def test_aware_previous_actions_and_dead_observation():
    env = scenario({"red_0": (2, 2)}, experiment=ExperimentConfig(observation_mode="aware"))
    env.step({"red_0": Action.RIGHT, "red_1": Action.ATTACK_RIGHT})
    dead = env.observe("red_0")
    assert not dead["local_grid"].any()
    np.testing.assert_array_equal(dead["action_mask"], [1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert dead["teammate_features"][0, 5+Action.ATTACK_RIGHT] == 1


def test_sessions_no_submission_leak_duplicate_and_stale_tokens():
    env = scenario(experiment=ExperimentConfig(observation_mode="aware"))
    collector = ActionCollector(env)
    collector.begin()
    red, teammate = collector.session("red_0"), collector.session("red_1")
    before = teammate.observe()
    red.submit_action(Action.ATTACK_RIGHT)
    for key, value in before.items():
        np.testing.assert_array_equal(value, teammate.observe()[key])
    with pytest.raises(ValueError):
        red.submit_action(Action.NOOP)
    env.step(collector.actions())
    with pytest.raises(RuntimeError):
        red.observe()
    collector.begin()
    stale = collector.session("red_0")
    env.reset(2)
    with pytest.raises(RuntimeError):
        stale.submit_action(Action.NOOP)


@pytest.mark.parametrize("key,expected", [("w", Action.UP), ("S", Action.DOWN), ("a", Action.LEFT),
    ("d", Action.RIGHT), ("Up", Action.ATTACK_UP), ("Down", Action.ATTACK_DOWN),
    ("Left", Action.ATTACK_LEFT), ("Right", Action.ATTACK_RIGHT), ("space", Action.NOOP)])
def test_keyboard_mapping_and_one_shot_consumption(key, expected):
    keyboard = KeyboardController()
    assert keyboard.feed_key(key)
    assert keyboard.act({}) == expected
    assert keyboard.act({}) == Action.NOOP
    assert not keyboard.feed_key("Escape")


def test_keyboard_connects_through_manager_without_advancing_on_input():
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
    a, b = scenario(), scenario()
    frame = render_rgb(a.snapshot())
    assert frame.shape == (144, 144, 3) and frame.dtype == np.uint8
    frame[:] = 0
    for _ in range(10):
        render_rgb(a.snapshot())
        assert a.step({}).to_dict() == b.step({}).to_dict()


@pytest.mark.parametrize("style", ["expand", "aggressive", "defend"])
def test_scripted_controllers_only_use_observations_and_valid_actions(style):
    env = TerritoryEnvironment(GameConfig(height=6, width=8, max_ticks=40))
    env.reset(9)
    bots = {a: FrontierController(i, style) for i, a in enumerate(AGENT_IDS)}
    for _ in range(40):
        obs = {a: env.observe(a) for a in AGENT_IDS}
        actions = {a: bots[a].act(obs[a]) for a in AGENT_IDS}
        assert all(obs[a]["action_mask"][v] for a, v in actions.items())
        env.step(actions)


def test_tk_controls_step_reset_selection_and_cleanup():
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
