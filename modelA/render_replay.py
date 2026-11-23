"""평가의 첫 파랑 경기와 첫 빨강 경기를 이어 붙여 GIF를 만듭니다."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from territory_rl import TerritoryEnvironment, GameConfig, ExperimentConfig
from territory_rl.visualization import render_rgb

from .rewards import calculate_learning_rewards
from .storage import atomic_write,write_json


def render_game(path,fps):
    lines=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    reset=lines[0]
    meta=reset['modelA']
    env=TerritoryEnvironment(GameConfig(**reset['game']),ExperimentConfig(**reset['experiment']))
    env.reset(reset['seed'])
    teams={}
    for agent in reset.get('agents', [a.agent_id for a in env.snapshot().agents]):
        teams.setdefault(agent.rsplit('_',1)[0],[]).append(agent)
    before={agent:env.observe(*_identity(agent)) for group in teams.values() for agent in group}
    frames=[]
    durations=[]
    font_path=Path('C:/Windows/Fonts/malgun.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()

    def add_frame(duration):
        snapshot=env.snapshot()
        board=Image.fromarray(render_rgb(snapshot,tile_size=48))
        image=Image.new('RGB',(max(680,board.width),board.height+130),'#121826')
        image.paste(board,(0,125))
        draw=ImageDraw.Draw(image)
        text=(f"Policy {meta['policy_team']} | playing {meta['side']} | opponent {meta['opponent']}\n"
              f"Tick {snapshot.tick}/{snapshot.max_ticks} | red {snapshot.areas[0]} / blue {snapshot.areas[1]}\n"
              f"Pair {meta['pair']} | seed {reset['seed']} | {fps} ticks/sec\n"
              f"Horizontal alignment: {'ON' if meta.get('mirrored',False) else 'OFF'} | "
              f"reward: {reset['learning_reward'].get('signal','gap_delta')} | "
              f"alpha: {reset['learning_reward']['capture_weight']:g}")
        draw.multiline_text((8,8),text,font=font,fill='white',spacing=5)
        for agent in snapshot.agents:
            if agent.alive:
                row,col=agent.position
                draw.text((col*48+24,row*48+149),str(int(agent.agent_id.rsplit('_',1)[1])+1),
                          font=font,fill='black',anchor='mm')
        frames.append(image)
        durations.append(duration)

    interval=max(10,round(1000/fps/10)*10)
    add_frame(interval)
    for item in lines[1:]:
        result=env.step(item['actions'])
        actual=json.loads(json.dumps(result.to_dict()))
        if actual != {key:item[key] for key in actual}:
            raise ValueError(f'원본 환경 전이가 다릅니다: {path}, tick={result.tick}')
        after={agent:env.observe(*_identity(agent)) for group in teams.values() for agent in group}
        infos={agent:dict(capture_credit=result.credits[agent]) for agent in after}
        rewards,components=calculate_learning_rewards(before,after,infos,teams,reset['learning_reward'])
        if rewards != item['learning_rewards'] or components != item['learning_components']:
            raise ValueError(f'학습 보상이 다릅니다: {path}, tick={result.tick}')
        before=after
        add_frame(interval)
    if not env.terminated:
        raise ValueError(f'완료되지 않은 리플레이입니다: {path}')
    durations[-1]=2000
    return frames,durations,meta


def _identity(agent):
    team,number=agent.rsplit('_',1)
    return team,int(number)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation',type=Path,help='replay_index.json이 있는 평가 폴더')
    parser.add_argument('--output',type=Path,help='실험 result/ 아래 폴더 권장')
    parser.add_argument('--fps',type=float,default=5)
    args=parser.parse_args(argv)
    if not 0<args.fps<=50:
        parser.error('--fps는 0보다 크고 50 이하여야 합니다.')
    evaluation=args.evaluation.resolve()
    output=args.output or evaluation.parents[3]/'result'/f'replays_{evaluation.parent.parent.name}_{evaluation.name}'
    output.mkdir(parents=True,exist_ok=True)
    records=json.loads((evaluation/'replay_index.json').read_text(encoding='utf-8'))
    groups={}
    for row in records:
        groups.setdefault((row['policy_team'],row['opponent']),{})[row['side']]=row
    manifest=[]
    for (policy,opponent),sides in groups.items():
        if set(sides)!={'blue','red'}:
            raise ValueError('양 진영의 첫 경기 리플레이가 필요합니다.')
        frames,durations,sources=[],[],[]
        for side in ('blue','red'):
            path=(evaluation/sides[side]['path']).resolve()
            if not path.is_relative_to(evaluation):
                raise ValueError('평가 폴더 밖의 리플레이 경로입니다.')
            part,times,meta=render_game(path,args.fps)
            frames.extend(part)
            durations.extend(times)
            sources.append(dict(path=str(path),side=side,mirrored=meta.get('mirrored',False),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        target=output/f'policy_{policy}_vs_{opponent}_blue_then_red.gif'
        def save(tmp):
            frames[0].save(tmp,format='GIF',save_all=True,append_images=frames[1:],
                           duration=durations,loop=0,optimize=False)
        atomic_write(target,save)
        with Image.open(target) as image:
            total=0
            for index in range(image.n_frames):
                image.seek(index)
                total+=image.info['duration']
            if total!=sum(durations):
                raise ValueError('저장된 GIF 기간이 다릅니다.')
        manifest.append(dict(gif=target.name,sources=sources,duration_ms=total,
                             verified_environment=True,verified_learning_rewards=True))
        print(f'영상 저장: {target.resolve()}')
    write_json(output/'video_manifest.json',manifest)


if __name__=='__main__':
    main()
