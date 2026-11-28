"""세 시드의 구조 비교 작업을 배치하고 기존 실험과 동일한 평가기를 사용합니다."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import torch
from fixed_ppo import reference, train_stage
from storage import allocate, load_pt, run_lock, write_json


def config():
    with (HERE/"config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    shapes, seeds = cfg["architectures"], cfg["seeds"]
    if not shapes or any(not s or any(type(n)!=int or n<1 for n in s) for s in shapes):
        raise ValueError("구조마다 양의 정수 은닉층 폭이 필요합니다.")
    if not seeds or any(type(s)!=int or s<0 for s in seeds) or len(set(seeds))!=len(seeds) or len(set(map(tuple,shapes)))!=len(shapes):
        raise ValueError("시드와 구조의 중복 및 범위를 확인하세요.")
    for section, keys in (("ppo",("updates","episodes_per_update","batch_size","epochs","checkpoint_every")),
                          ("finetune",("updates","episodes_per_update","validation_every","validation_games")),
                          ("evaluation",("games",))):
        if any(type(cfg[section][k])!=int or cfg[section][k]<=0 for k in keys):
            raise ValueError(f"{section} 예산은 양의 정수여야 합니다.")
    if cfg["finetune"]["episodes_per_update"] % 4:
        raise ValueError("고정 봇 업데이트당 경기 수를 4의 배수로 지정하세요.")
    if cfg["evaluation"]["games"] % 2 or cfg["finetune"]["validation_games"] % 2:
        raise ValueError("양 진영 평가를 위해 경기 수를 짝수로 지정하세요.")
    if cfg["evaluation"]["seed"] == cfg["finetune"]["validation_seed"]:
        raise ValueError("검증과 최종 평가 시드는 분리하세요.")
    return cfg


def variants(cfg):
    for seed in cfg["seeds"]:
        for shape in cfg["architectures"]:
            entry = copy.deepcopy({k:v for k,v in cfg.items() if k not in ("evaluation","architectures","seeds")})
            entry["ppo"]["hidden_layers"] = shape
            yield dict(config=entry, seed=seed, size=3)


def jobs(folder, cfg, create):
    existing = [(p.parent, json.loads(p.read_text(encoding="utf-8"))) for p in folder.glob("run[0-9]*/settings.json")]
    result = {}
    for meta in variants(cfg):
        matches = [path for path, other in existing if other==meta]
        if len(matches)>1 or (not matches and not create):
            raise ValueError(f"설정과 일치하는 run 수를 확인하세요: {meta['seed']}, {meta['config']['ppo']['hidden_layers']}, {len(matches)}개")
        run = matches[0] if matches else allocate(folder,"run")
        if not matches:
            write_json(run/"settings.json", meta)
        result.setdefault(meta["seed"], []).append(run)
    return result


def main():
    parser = argparse.ArgumentParser(description="exp5 · 3대3 PPO 네트워크 크기 비교")
    parser.add_argument("stage", choices=("plan","train","plots","evaluate"))
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", choices=("best.pt","latest.pt"), default="best.pt")
    parser.add_argument("--trace-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        for name in args.worker:
            if args.stage!="train" or not name.startswith("run") or not name[3:].isdigit():
                parser.error("잘못된 작업 창 인자입니다.")
            run = HERE/name
            with run_lock(run):
                meta = json.loads((run/"settings.json").read_text(encoding="utf-8"))
                for fixed in (False, True):
                    train_stage(run, meta["config"], meta["seed"], fixed)
        return
    cfg = config()
    if args.stage=="plan":
        for meta in variants(cfg):
            src = reference(HERE,meta["config"],meta["seed"])
            print(meta["seed"], meta["config"]["ppo"]["hidden_layers"], src or "신규 자기대전 사전학습")
    elif args.stage=="train":
        lock = HERE/".batch"
        lock.mkdir(exist_ok=True)
        with run_lock(lock):
            workers = []
            for seed, runs in jobs(HERE,cfg,True).items():
                cmd = [sys.executable,"-B",str(Path(__file__).resolve()),"train","--worker",*[r.name for r in runs]]
                workers.append((seed,subprocess.Popen(cmd,cwd=ROOT,
                                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name=="nt" else 0)))
            failed = [s for s,p in workers if p.wait()!=0]
            if failed:
                raise SystemExit(f"실패한 시드 {failed}. 같은 명령으로 재개하세요.")
    else:
        from compare import compare, plots
        runs = [r for group in jobs(HERE,cfg,False).values() for r in group]
        if any(not (r/"latest.pt").is_file() or load_pt(r/"latest.pt")["update"]!=cfg["finetune"]["updates"] for r in runs):
            raise ValueError("현재 설정의 학습이 아직 완료되지 않았습니다.")
        if args.stage=="plots":
            plots(HERE,runs)
        else:
            compare(HERE,runs,cfg["evaluation"],args.checkpoint,args.trace_all)


if __name__ == "__main__":
    main()
