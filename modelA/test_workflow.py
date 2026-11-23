"""새 보상, 양 팀 갱신, 재개와 평가·리플레이의 연결을 짧게 검증합니다."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config,parse_opponents
from .env_adapter import ModelAEnv
from .evaluate import evaluate
from .rewards import calculate_learning_rewards
from .rollout import compute_gae
from .train import train


class WorkflowTests(unittest.TestCase):
    def test_condition_seed_run_paths(self):
        from unittest.mock import patch
        from . import __main__ as cli
        config=Path(__file__).parent/'experiment'/'area_selfplay_alpha_grid'/'configs'/'a_t.toml'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='_paths_') as temporary:
            root=Path(temporary)
            with patch.object(cli,'ROOT',root), patch('modelA.train.train') as mocked:
                with contextlib.redirect_stdout(io.StringIO()):
                    for condition in ('A-T','A-S'):
                        for seed in (7,19):
                            self.assertEqual(cli.main(['train','--config',str(config),
                                                      '--condition',condition,'--seed',str(seed)]),0)
                paths=[call.args[1] for call in mocked.call_args_list]
                self.assertEqual(len(set(paths)),4)
                for call in mocked.call_args_list:
                    cfg,path=call.args
                    self.assertEqual(path.parent,root/'experiment'/'area_selfplay_alpha_grid'/'runs')
                    prefix='A-T_' if cfg.reward['mode']=='A-T' else 'A-S_alpha1_'
                    self.assertTrue(path.name.startswith(f'{prefix}seed{cfg.train["seed"]}_'))

    def test_area_share_and_alpha(self):
        teams={'red':['r0','r1','r2'],'blue':['b0','b1','b2']}
        obs={a:{'areas':np.array([16,14] if team=='red' else [14,16])}
             for team,ids in teams.items() for a in ids}
        infos={a:{'capture_credit':int(a in ('r0','r1'))} for a in obs}
        cfg=dict(signal='area_share',mode='A-S',team_weight=1,capture_weight=0.1)
        rewards,parts=calculate_learning_rewards(obs,obs,infos,teams,cfg)
        self.assertAlmostEqual(rewards['r0'],16/30+0.1)
        self.assertAlmostEqual(rewards['r1'],16/30+0.1)
        self.assertAlmostEqual(rewards['r2'],16/30)
        self.assertAlmostEqual(rewards['b0'],14/30)
        self.assertEqual(set(parts['r0']),{'team_area','capture'})
        cfg['capture_weight']=0
        zero,_=calculate_learning_rewards(obs,obs,infos,teams,cfg)
        cfg['mode']='A-T'
        baseline,_=calculate_learning_rewards(obs,obs,infos,teams,cfg)
        self.assertEqual(zero,baseline)

    def test_legacy_reward_configuration(self):
        old=load_config().to_dict()
        del old['reward']['signal']
        del old['evaluation']['mirror_to_training_side']
        cfg=load_config(base=old)
        self.assertEqual(cfg.reward['signal'],'gap_delta')
        self.assertTrue(cfg.evaluation['mirror_to_training_side'])
        for path in ('reward.signal=area_share','reward.capture_weight=0.1'):
            with self.assertRaises(ValueError):
                load_config(base=old,overrides=[path],resume=True)

    def test_horizontal_reflection_observation_mask_and_history(self):
        from territory_rl import Action
        cfg=load_config(overrides=['model.previous_actions=true'])
        env=ModelAEnv(cfg)
        try:
            obs=env.reset(7)
            original=obs['red_0']
            original['agent_alive'][1]=0
            original['agent_positions'][1]=(-1,-1)
            original['action_mask']=np.arange(env.action_count)%2
            original['team_action_codes']=np.array([int(Action.LEFT),int(Action.ATTACK_RIGHT),0])
            original['team_action_features']=np.zeros((3,env.action_count))
            original['team_action_features'][0,Action.LEFT.index]=1
            original['team_action_features'][1,Action.ATTACK_RIGHT.index]=1
            saved={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in original.items()}
            mirrored=env.reflection.observation(original)
            np.testing.assert_array_equal(mirrored['owner'],original['owner'][:,::-1])
            self.assertEqual(mirrored['agent_positions'][1].tolist(),[-1,-1])
            self.assertEqual(mirrored['team_action_codes'].tolist(),[int(Action.RIGHT),int(Action.ATTACK_LEFT),0])
            self.assertEqual(mirrored['team_action_features'][0,Action.RIGHT.index],1)
            self.assertEqual(mirrored['action_mask'][Action.LEFT.index],original['action_mask'][Action.RIGHT.index])
            twice=env.reflection.observation(mirrored)
            for key in original:
                if isinstance(original[key],np.ndarray):
                    np.testing.assert_array_equal(original[key],saved[key])
                    np.testing.assert_array_equal(twice[key],saved[key])
            self.assertEqual(env.reflection.action(Action.UP.index),Action.UP.index)
            env._previous={'red_0':Action.LEFT.index}
            normal,_,_=env.encode(obs,['red_0'])
            reverse,_,_=env.encode(obs,['red_0'],mirror=True)
            length=env.team_size*(env.action_count+1)
            history=reverse[0,-length:].reshape(env.team_size,-1)
            self.assertEqual(history[0,Action.RIGHT.index],1)
            self.assertEqual(history[0,-1],1)
            self.assertFalse(np.array_equal(normal,reverse))
        finally:
            env.close()

    def test_reward_algebra_and_cooperation(self):
        teams={'red':['r0','r1','r2'],'blue':['b0','b1','b2']}
        before={a:{'areas':np.array([15,15])} for ids in teams.values() for a in ids}
        after={a:{'areas':np.array([16,14] if team=='red' else [14,16])}
               for team,ids in teams.items() for a in ids}
        infos={a:{'capture_credit':int(a in ('r0','r1'))} for a in before}
        config=dict(mode='A-T',team_weight=1,capture_weight=1)
        rewards,_=calculate_learning_rewards(before,after,infos,teams,config)
        self.assertAlmostEqual(rewards['r0'],2/3)
        self.assertAlmostEqual(rewards['b0'],-2/3)
        self.assertAlmostEqual(sum(rewards.values()),0)
        config['mode']='A-S'
        rewards,_=calculate_learning_rewards(before,after,infos,teams,config)
        self.assertAlmostEqual(rewards['r0'],5/3)
        self.assertAlmostEqual(rewards['r1'],5/3)
        self.assertAlmostEqual(rewards['r2'],2/3)
        config['mode']='A-T'
        rewards,_=calculate_learning_rewards(after,after,infos,teams,config)
        self.assertTrue(all(value==0 for value in rewards.values()))

    def test_terminal_and_truncation_bootstrap(self):
        reward=np.array([[1.],[2.]],dtype=np.float32)
        value=np.zeros_like(reward)
        advantage,_=compute_gae(reward,value,np.array([0.]),1,1)
        np.testing.assert_allclose(advantage,[[3],[2]])
        advantage,_=compute_gae(reward,value,np.array([5.]),1,1)
        np.testing.assert_allclose(advantage,[[8],[7]])

    def test_opponent_counts(self):
        self.assertEqual(parse_opponents(['expand=4','random=2']),{'expand':4,'random':2})
        for values in (['expand=3'],['expand=0'],['expand=2','expand=4'],['invalid=2']):
            with self.assertRaises(ValueError):
                parse_opponents(values)

    def test_multiple_maps_and_terminal_transition(self):
        for height,width in ((3,10),(8,8)):
            cfg=load_config(overrides=[f'game.height={height}',f'game.width={width}','game.max_ticks=1'])
            env=ModelAEnv(cfg)
            try:
                obs=env.reset(7)
                for team in env.team_names:
                    features,masks,_=env.encode(obs,env.teams[team])
                    self.assertEqual(features.shape,(3,env.input_size))
                    self.assertEqual(masks.shape,(3,env.action_count))
                next_obs,rewards,terminal,cutoff,result,_=env.step({a:env.noop for a in obs})
                self.assertTrue(terminal)
                self.assertFalse(cutoff)
                self.assertEqual(set(next_obs),set(obs))
                self.assertEqual(result.tick,1)
                self.assertTrue(all(value==0.5 for value in rewards.values()))
            finally:
                env.close()

    def test_resume_both_teams_evaluation_replay_and_graphs(self):
        from territory_rl.cli import replay
        from .plot_training import main as plot_training
        from .plot_evaluation import main as plot_evaluation
        from .render_replay import main as render_replay

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='_verify_') as temporary:
            root=Path(temporary)
            cfg=load_config(overrides=[
                'reward.mode=A-S','game.max_ticks=4','train.episodes_per_update=2',
                'train.updates=1','train.checkpoint_every=1','ppo.epochs=1',
                'model.hidden_size=16','model.previous_actions=true'])
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                first_path=train(cfg,root/'resumed')
                _,first=load_checkpoint(first_path)
                extended=load_config(overrides=['train.updates=2'],base=first['config'],resume=True)
                final_path=train(extended,root/'resumed',first)
                _,resumed=load_checkpoint(final_path)
                initial=load_config(overrides=['train.updates=2'],base=cfg.to_dict())
                direct_path=train(initial,root/'direct')
                _,direct=load_checkpoint(direct_path)
                self.assertEqual(resumed['metrics'],direct['metrics'])
                self.assertEqual(resumed['episodes'],4)
                self.assertEqual(len(resumed['metrics']),4)
                for team in ('red','blue'):
                    for part in ('actor','critic'):
                        self.assertTrue(all(torch.equal(value,direct['learners'][team][part][key])
                                            for key,value in resumed['learners'][team][part].items()))
                        self.assertTrue(any(not torch.equal(value,first['learners'][team][part][key])
                                            for key,value in resumed['learners'][team][part].items()))
                evaluation=load_config(base=resumed['config'])
                evaluation.evaluation['opponents']={'expand':4,'random':2}
                folder=evaluate(evaluation,resumed,final_path,root/'eval')
                summary=json.loads((folder/'eval_summary.json').read_text(encoding='utf-8'))
                for team in ('red','blue'):
                    self.assertEqual(summary['results'][team]['expand']['overall']['games'],4)
                    self.assertEqual(summary['results'][team]['random']['overall']['games'],2)
                import csv
                with (folder/'eval_games.csv').open(encoding='utf-8') as file:
                    rows=list(csv.DictReader(file))
                self.assertEqual([r['side'] for r in rows[:4]],['blue','blue','red','red'])
                self.assertEqual(len(rows),12)
                traces=list((folder/'replays').glob('*.jsonl'))
                self.assertEqual(len(traces),8)
                for trace in traces:
                    replay(trace)
                    # 반전 관측에서 선택한 행동이 실제 저장 행동과 일치하는지 검사합니다.
                    from .checkpoint import make_learners,restore
                    from territory_rl import Action
                    records=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
                    header=records[0]
                    meta=header['modelA']
                    self.assertEqual(meta['mirrored'],meta['side']!=meta['policy_team'])
                    env=ModelAEnv(evaluation)
                    try:
                        learners=make_learners(env,evaluation,torch.device('cpu'))
                        restore(resumed,env,learners,torch.device('cpu'))
                        actor=learners[meta['policy_team']]['actor']
                        observations=env.reset(header['seed'])
                        for record in records[1:]:
                            agents=env.teams[meta['side']]
                            inputs,masks,_=env.encode(observations,agents,mirror=meta['mirrored'])
                            with torch.no_grad():
                                chosen=actor(torch.as_tensor(inputs)).masked_fill(
                                    ~torch.as_tensor(masks),float('-inf')).argmax(-1).tolist()
                            for agent,action in zip(agents,chosen):
                                expected=env.reflection.action(action) if meta['mirrored'] else action
                                self.assertEqual(Action.parse(record['actions'][agent]).index,expected)
                            observations,*_=env.step({a:Action.parse(code).index for a,code in record['actions'].items()})
                    finally:
                        env.close()
                plot_training([str(root/'resumed')])
                plot_evaluation([str(folder),'--output',str(root/'figures')])
                render_replay([str(folder),'--output',str(root/'videos')])
                self.assertEqual(len(list((root/'videos').glob('*.gif'))),4)
                self.assertTrue((root/'resumed'/'training_curves.png').is_file())
                self.assertEqual(len(list((root/'figures').glob('*.png'))),3)
                self.assertTrue((root/'figures'/'comparison.html').is_file())
                with (root/'figures'/'summary.csv').open(encoding='utf-8') as summary_file:
                    table_rows=list(csv.DictReader(summary_file))
                self.assertEqual(len(table_rows),2)
                for row in table_rows:
                    self.assertAlmostEqual(float(row['win_rate']),int(row['wins'])/int(row['games']))
                records=json.loads((root/'videos'/'video_manifest.json').read_text(encoding='utf-8'))
                self.assertTrue(all([source['side'] for source in record['sources']]==['blue','red']
                                    for record in records))


if __name__=='__main__':
    unittest.main()

