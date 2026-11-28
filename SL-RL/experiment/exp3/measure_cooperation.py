"""exp3의 기존 평가 경기를 재생해 협력 지표를 집계하고 연구 요약을 갱신합니다."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import torch
from analysis import collect, find_runs, read_rows
from arena import SIDES
from evaluation import Player, cooperation, match
from experiment.exp3.grid import conditions, copied_source, key_of, label
from storage import experiment, write_csv, run_lock

COUNTS = ("attacked_targets","shared_targets","captures","joint_captures","deaths","failed_moves")
REPORT = ROOT/"SL_RL_RESEARCH_SUMMARY.md"
OUT = HERE/"result"/"cooperation_u0400"


@lru_cache(maxsize=8)
def player(path):
    return Player(path, "update_0400.pt")


def one(condition, evaluation, row):
    rival = evaluation.rival
    field = SIDES.index(row["field"])
    events = []
    game, _ = match(player(evaluation.run.path), SIDES.index(row["trained_team"]), field,
                    player(rival.path) if rival else row["opponent"],
                    SIDES.index(row["opponent_team"]) if rival else None,
                    int(row["seed"]), False, event_log=events)
    if any(not math.isclose(float(row[k]), v, rel_tol=1e-7, abs_tol=1e-6) for k,v in game.items()):
        raise ValueError(f"원래 평가와 재생 결과가 다릅니다: {evaluation.path}, {row}")
    for side in (field, 1-field):
        primary = side == field
        model = evaluation.run if primary else rival
        yield dict(condition=condition, source=(evaluation.path/"games.csv").relative_to(HERE).as_posix(),
                   model=model.path.relative_to(HERE).as_posix() if model else row["opponent"],
                   train_size=model.size if model else "bot", train_seed=evaluation.run.seed,
                   trained_team=row["trained_team"] if primary else row["opponent_team"],
                   primary_trained_team=row["trained_team"], opponent_trained_team=row["opponent_team"],
                   matchup="duel" if rival else row["opponent"], field=SIDES[side], seed=row["seed"],
                   ticks=game["ticks"], win=game["win" if primary else "loss"],
                   draw=game["draw"], final_gap=game["final_gap"]*(1 if primary else -1),
                   **cooperation(events,side))


def save_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    write_csv(tmp, rows)
    os.replace(tmp,path)


def measure(task):
    torch.set_num_threads(1)
    key, cfg, check = task
    condition = label(key)
    origin = copied_source(cfg,key)
    source = HERE/"copied"/origin if origin else HERE
    runs = [r for r in find_runs(source,None) if key_of(r.settings["config"])==key]
    expected = {(size,seed) for size in (1,3) for seed in cfg["ae_grid"]["seeds"]}
    if len(runs)!=len(expected) or {(r.size,r.seed) for r in runs}!=expected:
        raise ValueError(f"{condition}의 체크포인트 구성을 확인하세요.")
    groups = collect(source,cfg,runs,400,cfg["evaluation"]["games"])
    paths = []
    for e in groups:
        # 원본 경기 CSV 및 선택한 체크포인트가 바뀌면 캐시를 새로 생성합니다.
        digest = hashlib.sha256((e.path.relative_to(HERE).as_posix()).encode())
        digest.update((e.path/"games.csv").read_bytes())
        for r in (e.run,e.rival):
            if r:
                p = player(r.path)
                if p.update != 400:
                    raise ValueError("평가에는 업데이트 400이 필요합니다.")
                for team in p.teams:
                    for tensor in team.state_dict().values():
                        digest.update(tensor.cpu().numpy().tobytes())
                if hasattr(p.encoder, "state_dict"):
                    for tensor in p.encoder.state_dict().values():
                        digest.update(tensor.cpu().numpy().tobytes())
        target = OUT/"games"/f"{condition}_{e.run.name}_{e.rival.name if e.rival else 'bots'}_{digest.hexdigest()[:16]}.csv"
        if check:
            for row in (e.games[0], e.games[-1]):
                list(one(condition,e,row))
        elif not target.exists():
            rows = [record for row in e.games for record in one(condition,e,row)]
            save_csv(target,rows)
            print(f"{condition} · {e.run.name} · {len(e.games)}경기 저장",flush=True)
        paths.append(target)
    return paths


def ratio(num,den):
    return num/den if den else None


def summarize(files):
    groups = defaultdict(Counter)
    for file in files:
        for row in read_rows(file):
            if row["train_size"]=="bot":
                continue
            for seed in (row["train_seed"],"all"):
                total = groups[row["condition"],row["matchup"],row["train_size"],seed]
                total.update({k:int(row[k]) for k in (*COUNTS,"win","ticks")})
                total["games"]+=1
    results=[]
    for (condition,matchup,size,seed),g in sorted(groups.items()):
        results.append(dict(condition=condition, matchup=matchup, train_size=size, train_seed=seed,
                            **g, shared_target_rate=ratio(g["shared_targets"],g["attacked_targets"]),
                            joint_capture_rate=ratio(g["joint_captures"],g["captures"]),
                            deaths_per_game=g["deaths"]/g["games"], failed_moves_per_game=g["failed_moves"]/g["games"],
                            win_rate=g["win"]/g["games"]))
    return results


def pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def update_report(rows,report=REPORT):
    duel = [r for r in rows if r["matchup"]=="duel" and r["train_seed"]=="all"]
    data = ["### exp3 협력 지표 측정", "",
            "업데이트 400 정책을 기존 평가와 같은 경기 시드·진영·상대로 재생했다. 아래 표는 복제 팀과 직접 팀이 서로 대전한 경기의 팀별 값이다. 공동 공격률은 공격한 (틱, 목표 칸) 중 같은 팀 2명 이상이 공격한 비율, 공동 점령률은 소유권이 바뀐 점령 칸 중 기여자가 2명 이상인 비율이다. 비율은 경기별 분자·분모를 합산했고 분모가 0이면 —로 표시했다.", "",
            "| 지도 조건 | 학습 팀 | 경기 | 공동 공격률 | 공동 점령률 | 사망/경기 | 이동 실패/경기 | 승률 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    names={"1":"1대1 복제","3":"3대3 직접"}
    for r in duel:
        data.append(f"| {r['condition']} | {names[r['train_size']]} | {r['games']} | {pct(r['shared_target_rate'])} | {pct(r['joint_capture_rate'])} | {r['deaths_per_game']:.2f} | {r['failed_moves_per_game']:.2f} | {pct(r['win_rate'])} |")
    data+=["","[고정 봇 5종과 맞대결의 시드별 집계 및 분자·분모](experiment/exp3/result/cooperation_u0400/summary.csv) · [경기별 양 팀 측정 파일 목록](experiment/exp3/result/cooperation_u0400/files.csv)"]
    pairs=defaultdict(dict)
    for r in duel:
        pairs[r["condition"]][r["train_size"]]=r
    interp=["### exp3에서 관찰된 공동 공격과 공동 점령",""]
    for k,title in (("shared_target_rate","공동 공격률"),("joint_capture_rate","공동 점령률")):
        comparable=[p for p in pairs.values() if all(p[s][k] is not None for s in ("1","3"))]
        higher=sum(p["3"][k]>p["1"][k] for p in comparable)
        lower=sum(p["3"][k]<p["1"][k] for p in comparable)
        interp.append(f"맞대결에서 직접 팀의 {title}이 복제 팀보다 높은 조건은 {higher}개, 낮은 조건은 {lower}개, 같은 조건은 {len(comparable)-higher-lower}개다.")
    interp+=["","두 비율은 화력을 한 칸에 모으는 빈도를 측정한다. 불필요한 중복 공격도 같은 방식으로 집계되므로, 이 수치만으로 협력이 승리를 일으켰다고 해석하기는 어렵다. 상대별 승률·점령과 함께 읽어야 한다."]
    original=report.read_bytes()
    text=original.decode("utf-8").replace("\r\n","\n")
    for tag,anchor,lines in (("DATA","## Ⅱ.",data),("INTERPRETATION","### 3. Sequential",interp)):
        start=f"<!-- EXP3_COOP_{tag}_BEGIN -->"
        end=f"<!-- EXP3_COOP_{tag}_END -->"
        block="\n".join([start,*lines,end])
        if start in text:
            i=text.index(start)
            text=text[:i]+block+text[text.index(end,i)+len(end):]
        else:
            i=text.index(anchor)
            text=text[:i]+block+"\n\n"+text[i:]
    if report.read_bytes()!=original:
        raise RuntimeError("보고서가 변경되었습니다. 명령을 다시 실행하세요.")
    tmp=report.with_suffix(".coop.tmp")
    tmp.write_bytes(text.replace("\n","\r\n" if b"\r\n" in original else "\n").encode("utf-8"))
    os.replace(tmp,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--check",action="store_true",help="모델·원자료를 확인하고 대진별 두 경기를 재생합니다.")
    args=parser.parse_args()
    if args.workers<1:
        parser.error("작업자 수는 양수여야 합니다.")
    _,cfg=experiment("exp3")
    tasks=[(key,cfg,args.check) for key in conditions(cfg)]
    def execute():
        with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
            return [p for group in pool.map(measure,tasks) for p in group]
    if args.check:
        execute()
        print("체크포인트·기존 대진·표본 재생 확인 완료")
        return
    (OUT/"games").mkdir(parents=True,exist_ok=True)
    with run_lock(OUT):
        files=execute()
        rows=summarize(files)
        save_csv(OUT/"summary.csv",rows)
        save_csv(OUT/"files.csv",[{"file":p.relative_to(OUT).as_posix()} for p in files])
        update_report(rows)
    print(REPORT)


if __name__=="__main__":
    main()
