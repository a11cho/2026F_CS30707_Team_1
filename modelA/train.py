"""고정 진영 자기대전에서 팀별 PPO를 갱신합니다."""

import hashlib
from pathlib import Path
from statistics import mean
import time

import torch
import territory_rl

from .checkpoint import make_learners, restore, save_checkpoint, setup_runtime
from .env_adapter import ModelAEnv
from .opponents import episode_seed
from .ppo import ppo_update
from .rollout import RolloutBuffer, play_episode
from .storage import write_csv, write_json


def source_fingerprint():
    files = list(Path(__file__).parent.glob('*.py'))
    files += list(Path(territory_rl.__file__).parent.glob('*.py'))
    return {f'{p.parent.name}/{p.name}':hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def train(cfg,output,state=None):
    output = Path(output)
    device = setup_runtime(cfg.runtime)
    torch.manual_seed(cfg.train['seed'])
    env = ModelAEnv(cfg)
    try:
        cfg.game = env.game_config
        learners = make_learners(env,cfg,device)
        rows,episodes,start = [],0,0
        if state is not None:
            if cfg.game != state['config']['game']:
                raise ValueError('실효 환경 설정이 체크포인트와 다릅니다.')
            if cfg.train['updates'] < state['update']:
                raise ValueError('총 목표 업데이트는 이미 완료한 수 이상이어야 합니다.')
            restore(state,env,learners,device,training=True)
            rows,episodes,start = list(state['metrics']),state['episodes'],state['update']
        exists = output.exists()
        if state is None and exists:
            raise ValueError(f'새 실행 폴더가 이미 있습니다: {output}')
        if state is not None and exists and not (output/'checkpoints'/'latest.pt').is_file():
            raise ValueError('기존 실행 폴더에는 latest.pt가 필요합니다.')
        output.mkdir(parents=True,exist_ok=state is not None)
        checkpoint = output/'checkpoints'/'latest.pt'
        if state is None or not exists:
            save_checkpoint(checkpoint,cfg,env,learners,device,start,episodes,rows)
        write_json(output/'config.json',cfg.to_dict())
        write_json(output/'env_signature.json',env.signature)
        write_json(output/f'sources_update{start:05d}.json',source_fingerprint())
        if rows:
            write_csv(output/'train_metrics.csv',rows)
        for update in range(start+1,cfg.train['updates']+1):
            started = time.perf_counter()
            buffers = {team:RolloutBuffer(cfg.ppo['gamma'],cfg.ppo['gae_lambda']) for team in learners}
            games = {team:[] for team in learners}
            for _ in range(cfg.train['episodes_per_update']):
                seed = episode_seed(cfg.train['seed'],10,episodes)
                result = play_episode(env,{t:p['actor'] for t,p in learners.items()},seed,device,
                                      critics={t:p['critic'] for t,p in learners.items()},buffers=buffers)
                for team in learners:
                    games[team].append(result[team])
                episodes += 1
            # 양 팀 모두 갱신 전에 같은 정책 버전으로 수집된 경험을 사용합니다.
            for team,parts in learners.items():
                losses = ppo_update(parts['actor'],parts['critic'],parts['optimizer'],
                                    buffers[team],cfg.ppo,device)
                keys = tuple(key for key in games[team][0] if key != 'termination_reasons')
                row = dict(update=update,episodes=episodes,team=team,
                           **{key:mean(game[key] for game in games[team]) for key in keys},**losses)
                rows.append(row)
            write_csv(output/'train_metrics.csv',rows)
            if update % cfg.train['checkpoint_every']==0 or update==cfg.train['updates']:
                save_checkpoint(checkpoint,cfg,env,learners,device,update,episodes,rows)
            scores = ' / '.join(f'{team} {mean(g["score"] for g in games[team]):.3f}' for team in learners)
            print(f'업데이트 {update}/{cfg.train["updates"]} · 누적 경기 {episodes} · '
                  f'{scores} · {time.perf_counter()-started:.1f}초',flush=True)
        return checkpoint
    finally:
        env.close()
