"""양 팀의 학습 CSV에서 성능·손실·보상 구성 추이를 PNG로 저장합니다."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,help='실행 폴더 또는 train_metrics.csv')
    parser.add_argument('--window',type=int,default=10,help='최근 업데이트 평균. 1은 원본')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--show',action='store_true')
    args = parser.parse_args(argv)
    if args.window<1:
        parser.error('--window는 1 이상이어야 합니다.')
    path = args.run/'train_metrics.csv' if args.run.is_dir() else args.run
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('완료된 학습 기록이 없습니다.')
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    panels = [
        ('자기대전 점수','score'),('최종 점유율','final_area_share'),
        ('경기 중 평균 점유율','mean_area_share'),('Critic 손실','value_loss'),
        ('Actor 손실','policy_loss'),('행동 엔트로피','entropy'),
        ('KL 근사','approx_kl'),('팀 보상 합계 / 에이전트','team_return'),
        ('개인 점령 보상 합계 / 에이전트','capture_return')]
    fig,axes = plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for team,color in (('red','#b94958'),('blue','#2466a7')):
        selected = [row for row in rows if row['team']==team]
        steps = [int(row['episodes']) for row in selected]
        for ax,(title,key) in zip(axes.flat,panels):
            vals = [float(row[key]) for row in selected]
            smooth = [np.mean(vals[max(0,i-args.window+1):i+1]) for i in range(len(vals))]
            ax.plot(steps,vals,color=color,alpha=0.2,linewidth=0.7)
            ax.plot(steps,smooth,color=color,label=team)
            ax.set(title=title,xlabel='누적 경기 수')
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend()
    config_path=path.parent/'config.json'
    info=''
    if config_path.exists():
        cfg=json.loads(config_path.read_text(encoding='utf-8'))
        reward=cfg['reward']
        info=f" · {reward.get('signal','gap_delta')} · alpha={reward['capture_weight']:g}"
    fig.suptitle(f'{path.parent.name}\n굵은 선은 최근 {args.window}업데이트 평균 · 학습 진영 고정{info}')
    output = args.output or path.with_name('training_curves.png')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    print(f'학습 그래프 저장: {output.resolve()}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__=='__main__':
    main()
