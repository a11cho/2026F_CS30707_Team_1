"""실행: python -B -m unittest smoke_test_mc_td.test_smoke -v"""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from territory_rl import ExperimentConfig, GameConfig
from territory_rl.agent import AgentState
from territory_rl.models import AGENT_IDS
from territory_rl.parallel_env import TerritoryParallelEnv
from .config import RunConfig
from .encoding import encode
from .learning import TabularControl, load_models, save_models
from .runner import episode, evaluate, train


class LearningTests(unittest.TestCase):
    def test_mc_first_visit_and_discount(self):
        model = TabularControl("mc", 1.0, 0.5)
        model.finish_episode([((1,), 2, 1), ((1,), 2, 2), ((2,), 3, 4)])
        self.assertEqual(model.q[(1,)][2], 3.0)
        self.assertEqual(model.q[(2,)][3], 4.0)
        self.assertEqual(model.updates, 2)

    def test_sarsa_bootstrap_and_terminal(self):
        model = TabularControl("sarsa", 0.5, 0.9)
        model.q[(2,)] = np.array([100., 4., 0, 0, 0, 0, 0, 0, 0])
        model.sarsa_step((1,), 3, 2, (2,), 1)
        self.assertAlmostEqual(model.q[(1,)][3], 2.8)
        model.sarsa_step((1,), 3, -2, (2,), 0, terminated=True)
        self.assertAlmostEqual(model.q[(1,)][3], 0.4)

    def test_mask_and_unseen_evaluation_does_not_insert(self):
        model = TabularControl("sarsa")
        rng = np.random.default_rng(1)
        for epsilon in (0, 1):
            for _ in range(20):
                self.assertEqual(model.act((1,), [0,0,0,1,0,0,0,0,0], epsilon, rng), 3)
        self.assertEqual(len(model.q), 0)

    def test_grid_removes_individual_duplicates(self):
        config = RunConfig(reward_specs=(("individual", "net", 0), ("individual", "area", 0)))
        self.assertEqual(len(list(config.conditions())), 2)
        self.assertEqual(len(list(RunConfig().conditions())), 18)

    def test_aware_features_are_used(self):
        local = TerritoryParallelEnv(GameConfig(height=6,width=6,observation_radius=2))
        aware = TerritoryParallelEnv(GameConfig(height=6,width=6,observation_radius=2),
                                     ExperimentConfig(observation_mode="aware"))
        a, _ = local.reset(seed=7)
        b, _ = aware.reset(seed=7)
        self.assertNotEqual(encode(a["red_0"]), encode(b["red_0"]))

    def test_serialization_and_reproducible_training(self):
        config = RunConfig(episodes=3,ticks=8,validation_pairs=1,test_pairs=1,eval_opponents=("random",))
        for algorithm in ("mc", "sarsa"):
            a, rows_a = train(config, ExperimentConfig(), algorithm, 7)
            b, rows_b = train(config, ExperimentConfig(), algorithm, 7)
            self.assertEqual(rows_a, rows_b)
            self.assertGreater(a[0].updates, 0)
            for key in a[0].q:
                np.testing.assert_array_equal(a[0].q[key], b[0].q[key])
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                path = Path(directory)/"model.json.gz"
                save_models(path,a,{"seed":7})
                restored, metadata = load_models(path)
                self.assertEqual(metadata["seed"],7)
                before = len(restored[0].q), restored[0].updates
                self.assertEqual(evaluate(config,ExperimentConfig(),a,"test"),
                                 evaluate(config,ExperimentConfig(),restored,"test"))
                self.assertEqual(before,(len(restored[0].q),restored[0].updates))

    def test_sarsa_carries_actual_next_action_and_death_is_not_terminal(self):
        class RecordedModel(TabularControl):
            def __init__(self):
                super().__init__("sarsa")
                self.chosen, self.observed, self.targets = [], [], []
            def act(self,key,mask,epsilon,rng,track=False):
                action = 4 if not self.chosen else 0  # red의 전선 시작 위치에서 적 영역으로 이동합니다.
                self.chosen.append(action)
                return action
            def sarsa_step(self,key,action,reward,next_key=None,next_action=None,terminated=False):
                self.observed.append(action)
                self.targets.append((next_action,terminated))
                super().sarsa_step(key,action,reward,next_key,next_action,terminated)
        class DeathScenario(TerritoryParallelEnv):
            def reset(self, seed=None, options=None):
                # 고정 정책 아군을 목적지와 멀리 배치하여 같은 틱에 목적지를
                # 점령하고 학습자를 살려주는 일이 없게 합니다.
                owner = np.zeros((6,6),dtype=np.int8)
                owner[:,3:] = 1
                positions = [(2,2),(0,0),(5,0),(0,5),(3,5),(5,5)]
                states = [AgentState(aid,i//3,positions[i]) for i,aid in enumerate(AGENT_IDS)]
                return super().reset(seed=seed,options={"owner":owner,"agents":states})
        env = DeathScenario(GameConfig(height=6,width=6,max_ticks=5,respawn_delay=2))
        model = RecordedModel()
        row = episode(env,[model],0,7,19,0,"noop",True)
        self.assertEqual(model.chosen,model.observed)
        self.assertEqual([x[0] for x in model.targets[:-1]],model.observed[1:])
        self.assertEqual([x[1] for x in model.targets],[False]*4+[True])
        self.assertGreater(row["dead_samples"],0)
        self.assertEqual(model.updates,5)

    def test_parallel_contract(self):
        from pettingzoo.test import parallel_api_test, parallel_seed_test
        factory = lambda: TerritoryParallelEnv(GameConfig(height=6,width=6,max_ticks=8))
        parallel_api_test(factory(),num_cycles=16)
        parallel_seed_test(factory,num_cycles=16)

    def test_learns_capture_in_one_tick_scenario(self):
        # 학습 가능성 연결 검사: 고정 시나리오의 완결된 한 틱 에피소드이며
        # 상대는 움직이지 않고 전선에 고정 정책 아군도 없습니다.
        class CaptureScenario(TerritoryParallelEnv):
            def reset(self, seed=None, options=None):
                owner = np.zeros((6,6),dtype=np.int8)
                owner[:,3:] = 1
                positions = [(2,2),(0,0),(5,0),(0,5),(3,5),(5,5)]
                states = [AgentState(aid,i//3,positions[i]) for i,aid in enumerate(AGENT_IDS)]
                return super().reset(seed=seed,options={"owner":owner,"agents":states})
        experiment = ExperimentConfig(reward_mode="individual",terminal_bonus=0)
        for algorithm in ("mc","sarsa"):
            env = CaptureScenario(GameConfig(height=6,width=6,max_ticks=1),experiment)
            model = TabularControl(algorithm,0.1,0.99)
            for i in range(60):
                episode(env,[model],0,7,i,0.5,"noop",True)
            observation, _ = env.reset(seed=7)
            key = encode(observation["red_0"])
            self.assertEqual(int(np.argmax(model.values(key))),8)  # 오른쪽 공격
            row = episode(env,[model],0,7,1000,0,"noop",False)
            self.assertEqual(row["mean_return"],1.0)


if __name__ == "__main__":
    unittest.main()
