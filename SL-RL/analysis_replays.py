"""각 비교 조건에서 한 경기를 선정해 점령·개체·행동을 애니메이션으로 표시합니다."""
from collections import Counter
from functools import lru_cache
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from arena import SIDES
from evaluation import Player, match
from storage import write_csv
from territory_rl.models import Action
from territory_rl.visualization import render_rgb

ACTION_TEXT = {Action.NOOP:"대기", Action.UP:"이동↑", Action.DOWN:"이동↓", Action.LEFT:"이동←",
               Action.RIGHT:"이동→", Action.ATTACK_UP:"공격↑", Action.ATTACK_DOWN:"공격↓",
               Action.ATTACK_LEFT:"공격←", Action.ATTACK_RIGHT:"공격→"}


def rank(row):
    """승·무·패 → 종료 점령 격차 → 에이전트당 보상 순으로 대표 경기를 정합니다."""
    return (int(row["win"]), int(row["draw"]), int(row["final_gap"]), float(row["per_agent_return"]))


@lru_cache(maxsize=8)
def font(size, bold=False):
    family = font_manager.FontProperties(family=["Malgun Gothic","AppleGothic","Noto Sans CJK KR","DejaVu Sans"],
                                         weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(family),size)


def gif_frame(state, last, row, title, rival):
    board = np.asarray(state["owner"])
    h,w = board.shape
    tile = min(52,max(20,640//w))
    width = max(680,w*tile+40)
    origin = ((width-w*tile)//2,100)
    image = Image.new("RGB",(width,origin[1]+h*tile+82),"#F5F4F0")
    draw = ImageDraw.Draw(image)
    draw.text((20,9),title,font=font(21,True),fill="#182F3C")
    team = row["field"]
    labels = {team:f"{row['model'].split('/')[0]} · {row['trained_team']} 학습",
              SIDES[1-SIDES.index(team)]:rival}
    text_y = 46
    for side,name,color,x in [(0,"빨강","#983740",20),(1,"파랑","#145988",width//2)]:
        draw.text((x,text_y),f"{name}   {labels[SIDES[side]]}   |   {int((board==side).sum())}칸",
                  font=font(16,True),fill=color)
    draw.text((20,74),f"틱 {state['tick']:02d} / {last:02d}     경기 시드 {row['seed']}",font=font(15),fill="#414F55")
    agents=[SimpleNamespace(team=SIDES.index(a["id"].split("_")[0]),position=a["position"],alive=a["alive"])
            for a in state["agents"]]
    rgb=render_rgb(SimpleNamespace(owner=board,agents=agents),tile)
    image.paste(Image.fromarray(rgb),origin)
    for a in state["agents"]:
        if a["alive"]:
            r,c=a["position"]
            number=int(a["id"].split("_")[1])+1
            draw.text((origin[0]+(c+.5)*tile,origin[1]+(r+.5)*tile),str(number),anchor="mm",font=font(17,True),fill="#18232B")
    actions=state.get("actions",{})
    for i,side in enumerate(SIDES):
        line="   ".join(f"{int(a.split('_')[1])+1} {ACTION_TEXT[Action(v)]}" for a,v in sorted(actions.items()) if a.startswith(side+"_"))
        draw.text((20,origin[1]+h*tile+7+30*i),("빨강   " if i==0 else "파랑   ")+(line or "경기 시작"),
                  font=font(16),fill="#983740" if i==0 else "#145988")
    return image


def save_gif(path, frames, fps):
    # 저장 중 중단되어도 이전에 생성한 완전한 GIF를 보존합니다.
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".gif",delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        frames[0].save(temp,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def render_selection(out, update, data, fps):
    chosen,counts={},Counter()
    for e in data:
        for row in e.games:
            group="duel" if e.rival else f"{e.run.size}v{e.run.size}_vs_{row['opponent']}"
            counts[group]+=1
            if group not in chosen or rank(row)>rank(chosen[group][1]):
                chosen[group]=(e,row)
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=f"update_{update:04d}.pt"
    @lru_cache(maxsize=8)
    def player(run_path):
        return Player(run_path,checkpoint)
    index=[]
    for group,(e,row) in chosen.items():
        rival=player(e.rival.path) if e.rival else row["opponent"]
        foe=SIDES.index(row["opponent_team"]) if e.rival else None
        field=SIDES.index(row["field"])
        result,states=match(player(e.run.path),SIDES.index(row["trained_team"]),field,rival,foe,int(row["seed"]),True)
        if any(not math.isclose(float(row[k]),v,abs_tol=1e-6,rel_tol=1e-7) for k,v in result.items()):
            raise RuntimeError(f"{group}: 재생 결과와 원본 평가 행이 일치하지 않습니다.")
        other=f"{e.rival.name} · {row['opponent_team']} 학습" if e.rival else row["opponent"]
        title="1vs1 복제 팀 vs 3vs3 직접 학습 팀" if e.rival else f"{e.run.size}vs{e.run.size} 학습 팀  vs  {other}"
        frames=[gif_frame(s,result["ticks"],row,title,other) for s in states]
        save_gif(out/(group+".gif"),frames,fps)
        actions=[(aid,Action(v)) for s in states[1:] for aid,v in s["actions"].items() if aid.startswith(row["field"]+"_")]
        forward=(Action.RIGHT,Action.ATTACK_RIGHT) if field==0 else (Action.LEFT,Action.ATTACK_LEFT)
        index.append(dict(file=group+".gif",candidates=counts[group],source=os.path.relpath(e.path/"games.csv",out),
                          **row, forward_action_fraction=sum(v in forward for _,v in actions)/len(actions)))
    write_csv(out/"index.csv",index)
    print(f"대표 리플레이 {len(index)}개 · {out}")
