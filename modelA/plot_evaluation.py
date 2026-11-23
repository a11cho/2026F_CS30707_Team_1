"""모델·상대별 평가를 색상 표로 저장합니다."""
import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from .storage import write_csv

HEADERS=['모델','보상 종류','α','학습 시드','학습 정책','상대','경기 수','승 / 무 / 패','최종 점유율','평균 점유율','경기 점수','승률']

def load_records(folders):
    """원본 평가 경로와 실제 승률을 보존합니다."""
    records=[]
    for folder in folders:
        data=json.loads((folder/'eval_summary.json').read_text(encoding='utf-8'))
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        for team,opponents in data['results'].items():
            for opponent,result in opponents.items():
                row=dict(run=folder.parent.parent.name,evaluation=folder.name,source=str(folder.resolve()),
                    condition=cfg['reward']['mode'],seed=cfg['train']['seed'],policy_team=team,opponent=opponent,
                    reward_signal=cfg['reward'].get('signal','gap_delta'),alpha=cfg['reward']['capture_weight'],
                    mirror_to_training_side=data['settings'].get('mirror_to_training_side',False),**result['overall'])
                row['win_rate']=row['wins']/row['games']
                records.append(row)
    return records

def sort_key(r):
    return (r['opponent'],r['condition']!='A-T',r['alpha'] if r['condition']!='A-T' else 0,str(r['seed']),r['policy_team'])

def aggregate_records(records):
    """같은 보상·상대·방향 정렬 조건만 경기 수로 가중 집계합니다."""
    groups=defaultdict(list)
    for r in records:
        groups[(r['condition'],r['reward_signal'],r['alpha'] if r['condition']!='A-T' else 0,r['opponent'],r['mirror_to_training_side'])].append(r)
    output=[]
    for (condition,signal,alpha,opponent,mirror),rows in groups.items():
        n=sum(r['games'] for r in rows)
        row=dict(condition=condition,reward_signal=signal,alpha=alpha,opponent=opponent,
            mirror_to_training_side=mirror,seed=','.join(map(str,sorted({r['seed'] for r in rows}))),
            policy_team='양 팀',runs=len({r['run'] for r in rows}),games=n)
        for key in ('wins','draws','losses'):
            row[key]=sum(r[key] for r in rows)
        for key in ('score','final_area_share','mean_area_share','deaths','captures'):
            row[key]=sum(r[key]*r['games'] for r in rows)/n
        row['win_rate']=row['wins']/n
        output.append(row)
    return sorted(output,key=sort_key)

def display_row(r):
    return [r['condition'],('점유율+점령' if r['condition']=='A-S' else '점유율') if r['reward_signal']=='area_share' else ('변화량+점령' if r['condition']=='A-S' else '면적 차이 변화'),
        '—' if r['condition']=='A-T' else f"{r['alpha']:g}",str(r['seed']),r['policy_team'],r['opponent'],str(r['games']),
        f"{r['wins']} / {r['draws']} / {r['losses']}",f"{r['final_area_share']:.1%}",
        f"{r['mean_area_share']:.1%}",f"{r['score']:.1%}",f"{r['win_rate']:.1%}"]

def render_table(rows,path,title):
    """맨 오른쪽 실제 승률을 0~100% 고정 척도로 색칠합니다."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(18,1.7+.34*len(rows)))
    ax.axis('off')
    ax.set_title(title,fontsize=16,pad=24)
    table=ax.table(cellText=[display_row(r) for r in rows],colLabels=HEADERS,cellLoc='center',loc='center',
        colWidths=[.055,.095,.04,.09,.07,.07,.06,.12,.085,.085,.085,.075])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,1.6)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee8')
        if i==0:
            cell.set_facecolor('#24354b')
            cell.set_text_props(color='white',weight='bold')
        else:
            cell.set_facecolor('#f0f4f8' if i%2==0 else 'white')
            if j==len(HEADERS)-1:
                cell.set_facecolor(colormaps['RdYlGn'](rows[i-1]['win_rate']))
                cell.set_text_props(weight='bold',color='#101820')
    fig.text(.02,.015,'승률: 0% 빨강 → 50% 노랑 → 100% 초록 | 경기 점수 = (승 + 0.5 × 무) / 경기 수 | 양 진영 합계',fontsize=10)
    fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.3)
    plt.close(fig)

def render_html(summary,records,path):
    """고정 헤더와 원본 평가 상대 링크를 포함합니다."""
    from matplotlib import colormaps,colors
    def table(rows,details=False):
        parts=['<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in HEADERS)+'</tr></thead><tbody>']
        for r in rows:
            cells=display_row(r)
            first=html.escape(cells[0])
            if details:
                link=Path(os.path.relpath(r['source'],path.parent)).as_posix()+'/eval_summary.json'
                first='<a href="'+html.escape(link,quote=True)+'" title="'+html.escape(r['run'],quote=True)+'">'+first+'</a>'
            parts.append('<tr><td>'+first+'</td>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells[1:-1]))
            color=colors.to_hex(colormaps['RdYlGn'](r['win_rate']))
            parts.append('<td style="background:'+color+';font-weight:bold">'+cells[-1]+'</td></tr>')
        return ''.join(parts)+'</tbody></table>'
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Model A 평가 비교</title>
<style>body{font-family:Malgun Gothic,sans-serif;margin:24px;color:#172536}table{border-collapse:collapse;width:100%;margin:20px 0 40px;font-size:14px}th,td{padding:9px;border:1px solid #dae1e8;text-align:center}th{background:#24354b;color:white;position:sticky;top:0}tr:nth-child(even){background:#f0f4f8}a{color:#205b9a}</style>
<h1>Model A 평가 비교</h1><p>A-T: 팀 점유율 보상. A-S: 팀 점유율 + α × 개인 점령 보상.<br>승률 = 승 / 경기 수. 경기 점수 = (승 + 0.5 × 무) / 경기 수.<br>승률: 0% 빨강 → 50% 노랑 → 100% 초록. 경기 수는 양 진영 합계입니다.<br>요약은 같은 조건의 시드·두 정책을 합산합니다. 상세 모델 이름에 마우스를 올리면 실행 폴더 이름이 표시되며, 클릭하면 원본 평가 JSON을 엽니다.</p>'''
    text+='<h2>조건별 요약</h2>'+table(summary)
    for opponent in dict.fromkeys(r['opponent'] for r in records):
        text+='<h2>'+html.escape(opponent)+' · 시드·정책별 상세</h2>'+table([r for r in records if r['opponent']==opponent],True)
    path.write_text(text+'</html>',encoding='utf-8')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    records=sorted(load_records(args.evaluations),key=sort_key)
    summary=aggregate_records(records)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'comparison.csv',records)
    write_csv(args.output/'summary.csv',summary)
    render_table(summary,args.output/'evaluation_summary.png','Model A · 조건별 평가 요약 (세 시드·양 팀 합산)')
    for opponent in dict.fromkeys(r['opponent'] for r in records):
        render_table([r for r in records if r['opponent']==opponent],args.output/f'evaluation_{opponent}.png',f'상대: {opponent} · 시드·정책별 평가')
    render_html(summary,records,args.output/'comparison.html')
    print(f'평가 비교 표 저장: {args.output.resolve()}')

if __name__=='__main__':
    main()
