"""추론 관전 기능이 Q-table·체크포인트를 변경하지 않는지 독립적으로 검사합니다.

기본 검사는 창을 만들지 않습니다. --gui 옵션은 숨긴 Tk 창으로 UI 제어도 검사합니다.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from territory_rl import ExperimentConfig
from territory_rl.models import AGENT_IDS
from .config import RunConfig
from .learning import save_models, TabularControl
from .runner import train, episode, make_environment
from .viewer import ModelSimulation, SimulationViewer


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.paths = {}
        for count in (1,3,6):
            cfg = RunConfig(height=4,width=4,radius=1,episodes=2,ticks=8,learner_count=count)
            models,_ = train(cfg,ExperimentConfig(),"sarsa",7)
            path=Path(cls.directory.name)/f"models_{count}.json.gz"
            save_models(path,models,dict(config=asdict(cfg),experiment=asdict(ExperimentConfig()),encoding="compact_v1",seed=7))
            cls.paths[count]=path

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_auto_modes_roles_and_legacy_support(self):
        for count in (1,3,6):
            sim=ModelSimulation(self.paths[count])
            self.assertEqual(len(sim.assigned),count)
            self.assertEqual(len(sim.bots),6-count)
            self.assertEqual(sim.mode,"both" if count==6 else "fixed")
        sim=ModelSimulation(self.paths[6],team=1)
        self.assertEqual(sim.roles["blue_0"],0)
        self.assertEqual(sim.roles["red_0"],3)
        sim=ModelSimulation(self.paths[6],mode="fixed",cohort=1,team=1)
        self.assertEqual(sim.roles,{"blue_0":3,"blue_1":4,"blue_2":5})
        with self.assertRaises(ValueError):
            ModelSimulation(self.paths[1],mode="both")
        with self.assertRaises(ValueError):
            ModelSimulation(self.paths[3],cohort=1)

    def test_no_training_calls_or_checkpoint_mutation(self):
        path=self.paths[6]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(TabularControl,"update",side_effect=AssertionError("학습 금지")), \
             patch.object(TabularControl,"sarsa_step",side_effect=AssertionError("학습 금지")), \
             patch.object(TabularControl,"finish_episode",side_effect=AssertionError("학습 금지")), \
             patch.object(TabularControl,"act",side_effect=AssertionError("추론 전용 정책을 사용해야 함")):
            sim=ModelSimulation(path)
            before=[{k:v.copy() for k,v in policy.q.items()} for policy in sim.policies]
            while not sim.env.terminated:
                result=sim.step()
                self.assertFalse(any(e.kind=="rejected" for e in result.events))
            self.assertEqual(sim.env.tick,8)
            self.assertIsNone(sim.step())
            for policy, original in zip(sim.policies,before):
                self.assertEqual(policy.q.keys(),original.keys())
                for key in original:
                    np.testing.assert_array_equal(policy.q[key],original[key])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_frozen_q_and_render_do_not_change_results(self):
        from territory_rl.visualization import render_rgb
        a,b=ModelSimulation(self.paths[6]),ModelSimulation(self.paths[6])
        policy=a.policies[0]
        key=next(iter(policy.q))
        with self.assertRaises(TypeError):
            policy.q[key]=np.zeros(9)
        with self.assertRaises(ValueError):
            policy.q[key][0]=1
        while not a.env.terminated:
            frame=render_rgb(a.env.snapshot())
            self.assertEqual(frame.shape,(96,96,3))
            frame[:]=0
            self.assertEqual(a.step().to_dict(),b.step().to_dict())

    def test_same_seed_reset_replays_actions_and_respawns(self):
        sim=ModelSimulation(self.paths[6],seed=19,policy_seed=123)
        expected=[]
        while not sim.env.terminated:
            expected.append(sim.step().to_dict())
        sim.reset()
        actual=[]
        while not sim.env.terminated:
            actual.append(sim.step().to_dict())
        self.assertEqual(expected,actual)
        self.assertEqual(sim.policy_seed,123)
        with self.assertRaises(ValueError):
            sim.reset(seed=-1)

    def test_matches_existing_greedy_evaluation_transition(self):
        from .learning import load_models
        for count in (1,3,6):
            sim=ModelSimulation(self.paths[count],seed=7,policy_seed=19)
            models,metadata=load_models(self.paths[count])
            cfg=RunConfig(**metadata["config"])
            env=make_environment(cfg,ExperimentConfig(**metadata["experiment"]))
            row=episode(env,models,0,7,19,0,"expand",False)
            while not sim.env.terminated:
                sim.step()
            self.assertEqual(sim.env.snapshot().areas,env.core.snapshot().areas)
            self.assertEqual(sim.env.snapshot().agents,env.core.snapshot().agents)
            np.testing.assert_array_equal(sim.env.snapshot().owner,env.core.snapshot().owner)
            self.assertEqual(sim.returns["red_0"],row["role_0_return"])


@unittest.skipUnless("--gui" in sys.argv,"GUI 검사는 --gui 옵션으로만 실행합니다.")
class ViewerGuiTests(ViewerTests):
    def test_controls_render_reset_apply_and_cleanup(self):
        import tkinter as tk
        root=tk.Tk()
        root.withdraw()
        app=SimulationViewer(ModelSimulation(self.paths[6]),root=root,interval_ms=50)
        try:
            root.update_idletasks()
            self.assertGreater(len(app.renderer.canvas.find_all()),16)
            self.assertIn("role 5",app.roster.get("1.0","end"))
            app.step_once()
            self.assertEqual(app.simulation.env.tick,1)
            app.toggle()
            self.assertTrue(app.running)
            self.assertIsNotNone(app.after_id)
            app.pause()
            self.assertFalse(app.running)
            self.assertIsNone(app.after_id)
            app.interval.set(20)
            app.change_speed()
            app.toggle()
            root.after_cancel(app.after_id)
            app.after_id=None
            app._timer()
            self.assertEqual(app.simulation.env.tick,2)
            app.pause()
            app.reset()
            self.assertEqual(app.simulation.env.tick,0)
            app.mode.set("모델 팀 대 고정 상대")
            app.cohort.set("1")
            app.team.set("파랑")
            app.apply_settings()
            self.assertEqual(app.simulation.roles["blue_0"],3)
            for _ in range(10):
                app.step_once()
            self.assertEqual(app.simulation.env.tick,8)
            self.assertFalse(app.running)
        finally:
            app.close()
        self.assertTrue(app.closed)


if __name__=="__main__":
    # 기본 unittest 실행에서는 GUI 검사를 건너뛰고, 명시적 --gui 때만 실행합니다.
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ViewerTests))
    if "--gui" in sys.argv:
        suite.addTest(ViewerGuiTests("test_controls_render_reset_apply_and_cleanup"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
