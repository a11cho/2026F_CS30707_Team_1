"""실제 맞대결의 매 틱을 복제해 세 정책을 동일한 상태·상대 행동에서 비교합니다."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import tomllib

import run
from arena import SIDES, seed_for
from evaluation import Player
from model import Action, MESSAGE, STATUS, SLOTS, arena_for, play_tick
from ppo.cooperation_scenes import SceneTick, capture_entry, yield_moves
from storage import allocate, load_pt, write_csv, write_json
import torch

MODES = ("공개", "비공개", "독립")


def hide_codes(net, args):
    # 순번과 앞선 송신자의 상태는 보존하고 행동 원핫만 지웁니다.
    board, context, peers, mask = args
    context = context.clone()
    context[:, STATUS+SLOTS:].reshape(-1,SLOTS,MESSAGE)[:,:,:len(Action)] = 0
    return board, context, peers, mask


def assess(before, after, step, field, actions):
    team = before.teams[field]
    t = SceneTick(before.state, step, field, team, [actions[a].index for a in team], actions)
    enemy = [a for a in before.state.agents if a.team != field and a.alive
             and step.tick-a.last_attack_tick >= 2]
    threatened = {aid for aid,pos in t.allies.items() if any(
        abs(pos[0]-a.position[0])+abs(pos[1]-a.position[1])==1 for a in enemy)}
    entry = capture_entry(t)
    chain, _ = yield_moves(t)
    count = Counter(alive=len(t.allies), exposed=len(threatened), exposed_deaths=len(threatened & t.deaths),
                    deaths=len(set(team) & t.deaths), cover_opp=len(t.cover_targets), saves=0,
                    entry_opp=entry.total, entries=entry.completed, chain_opp=chain.total, chains=chain.completed)
    for cell,victim in t.successful_covers.items():
        defenders = [aid for aid,target in t.ally_attacks.items() if target==cell]
        if defenders:
            revised = dict(actions)
            revised.update(dict.fromkeys(defenders,Action.NOOP))
            alt = deepcopy(before).env.step(revised)
            count["saves"] += any(e.kind=="death" and e.agent_id==victim for e in alt.events)
    return count


def board(state):
    cells = [[("🟥" if v==0 else "🟦") for v in row] for row in state.owner]
    for a in state.agents:
        if a.alive:
            r,c=a.position
            cells[r][c]=("🔴" if a.team==0 else "🔵")+a.agent_id[-1]
    return "\n".join(" ".join(row) for row in cells)


def evaluate(cfg,seed,pairs,counts,examples):
    state=load_pt(run.run_path(run.HERE,seed)/"latest.pt")
    cfg_game=state["config"]["game"]
    nets=run.policies_for(state["config"])
    for net,weights in zip(nets,state["policies"]):
        net.load_state_dict(weights)
        net.eval()
    player=Player(run.ROOT/"SL-RL"/"experiment"/"exp2"/cfg["baselines"][str(seed)][1],"latest.pt")
    if player.size!=3 or player.seed!=seed or player.cfg["game"]!=cfg_game:
        raise ValueError("비교 체크포인트 설정 불일치")
    for trained in range(2):
      for other in range(2):
       for field in range(2):
        foe=1-field
        for pair in range(pairs):
            game_seed=seed_for(cfg["evaluation"]["seed"],pair,71)
            arena=arena_for(cfg_game,game_seed)
            while not arena.state.terminated:
                before=deepcopy(arena)
                external=player.actions(before,foe,other)
                policies=[None,None]
                policies[field]=nets[trained]
                _,_,actual=play_tick(arena,policies,0,greedy=True,external_actions=external)
                fixed={k:Action(v) for k,v in arena.env.last_decision[-1]["applied"].items()}
                for side,weight in [(field,trained),(foe,other)]:
                    source="공개" if side==field else "독립"
                    outcomes={}
                    for mode in MODES:
                        if mode==source:
                            branch,step=arena,actual
                            actions=fixed
                        else:
                            branch=deepcopy(before)
                            policies=[None,None]
                            actions=dict(fixed)
                            hook=None
                            if mode=="독립":
                                actions.update(player.actions(branch,side,weight))
                            else:
                                policies[side]=nets[weight]
                                if mode=="비공개":
                                    hook=nets[weight].register_forward_pre_hook(hide_codes)
                            try:
                                _,_,step=play_tick(branch,policies,0,greedy=True,external_actions=actions)
                            finally:
                                if hook: hook.remove()
                            actions={k:Action(v) for k,v in branch.env.last_decision[-1]["applied"].items()}
                        metric=assess(before,branch,step,side,actions)
                        counts[seed,source,mode].update(metric)
                        outcomes[mode]=(metric, actions, branch.state)
                    team=before.teams[side]
                    assert outcomes["공개"][1][team[0]]==outcomes["비공개"][1][team[0]]
                    for key in ("saves","entries","chains","deaths"):
                        vals=[item[0][key] for item in outcomes.values()]
                        score=(max(vals)-min(vals),max(vals))
                        slot=(seed,key)
                        if slot in examples and examples[slot][0]>=score: continue
                        detail=[f"장면 출처 {source} · 경기 시드 {game_seed} · 틱 {before.state.tick} · 진영 {SIDES[side]} · 학습 팀 {SIDES[weight]}",
                                "행동 전\n"+board(before.state)]
                        for mode,(metric,acts,st) in outcomes.items():
                            detail.append(f"{mode} · {key}={metric[key]}\n"+board(st)+"\n"+
                                          " ".join(f"{a}={acts[a].name}" for a in team))
                        examples[slot]=(score,"\n\n".join(detail))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",type=int,default=20,help="학습 팀 조합·진영당 경기 수")
    parser.add_argument("--seeds",type=int,nargs="+",default=[7,19,31])
    args=parser.parse_args()
    if args.pairs<1: parser.error("--pairs는 양수입니다.")
    torch.set_num_threads(1)
    cfg=tomllib.loads((run.HERE/"config.toml").read_text(encoding="utf-8-sig"))
    out=allocate(run.HERE/"result","paired")
    write_json(out/"settings.json",dict(pairs=args.pairs,seeds=args.seeds,checkpoint="latest.pt",evaluation_seed=cfg["evaluation"]["seed"]))
    counts=defaultdict(Counter)
    examples={}
    for seed in args.seeds:
        evaluate(cfg,seed,args.pairs,counts,examples)
        rows=[]
        for (s,source,mode),c in counts.items():
            row=dict(seed=s,source=source,mode=mode,natural=int(source==mode),**c)
            for label,a,b,scale in [("exposure_per_1000","exposed","alive",1000),
                    ("death_if_exposed","exposed_deaths","exposed",1),("cover_rate","saves","cover_opp",1),
                    ("entry_rate","entries","entry_opp",1),("chain_rate","chains","chain_opp",1)]:
                row[label]=scale*c[a]/c[b] if c[b] else None
            rows.append(row)
        write_csv(out/"summary.csv",rows)
        html='<meta charset="utf-8"><title>동일 틱 비교</title><h1>동일 장면 전후 리플레이</h1><p>각 지표에서 세 조건의 차이가 가장 큰 틱. 동률이면 성공 건수가 큰 첫 장면. 빨강·파랑 원은 생존 개체입니다.</p>'
        for (s,k),(_,detail) in examples.items():
            html+=f'<details><summary>시드 {s} · {k}</summary><pre>{escape(detail)}</pre></details>'
        (out/"replay.html").write_text(html,encoding="utf-8")
        print(seed,out,flush=True)


if __name__=="__main__":
    main()

