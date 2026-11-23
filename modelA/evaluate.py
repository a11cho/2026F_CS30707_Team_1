"""각 팀 정책을 고정 상대와 파랑 진영 먼저, 빨강 진영 나중 순서로 평가합니다."""

from pathlib import Path

import torch

from .checkpoint import make_learners, restore, setup_runtime
from .env_adapter import ModelAEnv
from .metrics import summarize_games
from .opponents import episode_seed
from .rollout import play_episode
from .storage import write_csv, write_json


def evaluate(cfg,state,checkpoint_path,output):
    output = Path(output)
    if output.exists():
        raise ValueError(f'새 평가 폴더가 이미 있습니다: {output}')
    device = setup_runtime(cfg.runtime)
    env = ModelAEnv(cfg)
    try:
        learners = make_learners(env,cfg,device)
        restore(state,env,learners,device)
        for parts in learners.values():
            parts['actor'].eval()
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'status.json',dict(status='running'))
        write_json(output/'config.json',cfg.to_dict())
        rows,replays = [],[]
        for policy_team,parts in learners.items():
            for opponent,count in cfg.evaluation['opponents'].items():
                pairs = count//2
                # 환경 ID의 빨강→파랑 순서를 뒤집어 파랑부터 평가합니다.
                for side in reversed(env.team_names):
                    for pair in range(pairs):
                        seed = episode_seed(cfg.evaluation['seed'],20,pair)
                        torch.manual_seed(episode_seed(cfg.evaluation['seed'],21,pair,env.team_names.index(side)))
                        relative = Path('replays')/f'policy_{policy_team}_{opponent}_{side}.jsonl'
                        replay = output/relative if pair==0 else None
                        mirror=cfg.evaluation['mirror_to_training_side'] and side!=policy_team
                        metadata = dict(policy_team=policy_team,side=side,opponent=opponent,
                                        pair=pair,checkpoint=str(checkpoint_path),update=state['update'],
                                        mirror_to_training_side=cfg.evaluation['mirror_to_training_side'],
                                        mirrored=mirror)
                        result = play_episode(env,{side:parts['actor']},seed,device,opponent=opponent,
                                              deterministic=cfg.evaluation['deterministic'],
                                              replay_path=replay,metadata=metadata,
                                              mirrored_teams=(side,) if mirror else ())[side]
                        rows.append(dict(policy_team=policy_team,opponent=opponent,side=side,
                                         pair=pair,seed=seed,mirrored=mirror,**result))
                        if replay:
                            replays.append(dict(**metadata,path=relative.as_posix()))
                print(f'학습팀 {policy_team} · 상대 {opponent} · {count}경기 완료',flush=True)
        results = {}
        for team in learners:
            results[team] = {}
            for opponent in cfg.evaluation['opponents']:
                subset = [row for row in rows if row['policy_team']==team and row['opponent']==opponent]
                results[team][opponent] = dict(overall=summarize_games(subset),by_side={
                    side:summarize_games([row for row in subset if row['side']==side])
                    for side in reversed(env.team_names)})
        write_csv(output/'eval_games.csv',rows)
        write_json(output/'eval_summary.json',dict(
            checkpoint=str(checkpoint_path),update=state['update'],episodes=state['episodes'],
            reward=cfg.reward,settings=cfg.evaluation,results=results,
            note='opponents의 경기 수는 정책 하나당 양 진영 합계입니다. 서로 다른 상대의 평균은 집계하지 않습니다.'))
        write_json(output/'replay_index.json',replays)
        write_json(output/'status.json',dict(status='complete',games=len(rows),replays=len(replays)))
        return output
    finally:
        env.close()
