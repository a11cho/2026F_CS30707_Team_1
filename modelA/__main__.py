"""실험별 새 학습·이어 학습·상대별 평가 명령."""

import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

from .config import load_config,parse_opponents,validate


ROOT = Path(__file__).resolve().parent


def new_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    trainer = commands.add_parser('train',help='고정 진영 양 팀 자기대전')
    trainer.add_argument('--experiment',default='area_selfplay_alpha_grid',help='experiment/ 아래 실험 이름')
    trainer.add_argument('--condition',choices=('A-T','A-S'),help='새 학습의 보상 조건')
    source = trainer.add_mutually_exclusive_group()
    source.add_argument('--config',type=Path)
    source.add_argument('--resume',type=Path,help='실행 폴더 또는 latest.pt')
    trainer.add_argument('--seed',type=int)
    evaluator = commands.add_parser('evaluate',help='양 정책의 고정 상대 평가')
    evaluator.add_argument('--checkpoint',type=Path,required=True)
    evaluator.add_argument('--config',type=Path)
    evaluator.add_argument('--opponent',action='append',metavar='이름=짝수경기수')
    for command in (trainer,evaluator):
        command.add_argument('--set',action='append',default=[],metavar='구역.항목=값')
        command.add_argument('--output',type=Path,help='새 결과 폴더. 생략하면 자동 생성')
    args = parser.parse_args(argv)
    try:
        from .checkpoint import load_checkpoint
        if args.command=='train':
            from .train import train
            if args.resume:
                if args.seed is not None or args.condition:
                    parser.error('이어 학습에서는 저장된 시드와 조건을 사용합니다.')
                path,state = load_checkpoint(args.resume)
                if not args.output and (path.name!='latest.pt' or path.parent.name!='checkpoints'):
                    parser.error('별도 보관한 체크포인트는 --output으로 새 실행 폴더를 지정하세요.')
                if args.output and args.output.exists():
                    parser.error('--output에는 새 실행 폴더를 지정하세요. 같은 폴더에서 재개하려면 --output을 생략하세요.')
                cfg = load_config(overrides=args.set,base=state['config'],resume=True)
                output = args.output.resolve() if args.output else path.parent.parent
                result = train(cfg,output,state)
            else:
                if Path(args.experiment).name != args.experiment or args.experiment in ('.','..'):
                    parser.error('--experiment에는 폴더 이름 하나를 지정하세요.')
                experiment = ROOT/'experiment'/args.experiment
                condition = args.condition or 'A-T'
                config = args.config or experiment/'configs'/('a_t.toml' if condition=='A-T' else 'a_s.toml')
                overrides = list(args.set)
                if args.seed is not None:
                    overrides.append(f'train.seed={args.seed}')
                if args.condition:
                    overrides.append(f'reward.mode={args.condition}')
                cfg = load_config(config,overrides)
                alpha=f'_alpha{cfg.reward["capture_weight"]:g}' if cfg.reward['mode']=='A-S' else ''
                name = f'{cfg.reward["mode"]}{alpha}_seed{cfg.train["seed"]}_{new_id()}'
                output = args.output or experiment/'runs'/name
                result = train(cfg,output.resolve())
            print(f'체크포인트: {result}')
        else:
            from .evaluate import evaluate
            path,state = load_checkpoint(args.checkpoint)
            cfg = load_config(args.config,args.set,base=state['config'])
            if args.opponent:
                cfg.evaluation['opponents'] = parse_opponents(args.opponent)
            validate(cfg)
            saved_config=load_config(base=state['config']).to_dict()
            for key in ('game','reward','model','train','ppo'):
                if cfg.to_dict()[key] != saved_config[key]:
                    parser.error('평가에서는 evaluation과 runtime만 변경할 수 있습니다.')
            run = path.parent.parent
            output = args.output or run/'evaluations'/f'eval_{new_id()}'
            result = evaluate(cfg,state,path,output.resolve())
            print(f'평가 결과: {result}')
    except KeyboardInterrupt:
        print('중단했습니다. 마지막으로 저장된 latest.pt에서 --resume으로 재개할 수 있습니다.',file=sys.stderr)
        return 130
    except (ValueError,OSError,TypeError,KeyError) as exc:
        parser.exit(2,f'실행 오류: {exc}\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
