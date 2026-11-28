"""혼합 교사 행동 모방 실험의 병렬 학습·평가·그래프 명령입니다."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import torch
from imitation import fit
from storage import allocate, load_pt, run_lock, save_pt, write_json


def config():
    with (HERE/"config.toml").open("rb") as stream:
        cfg = tomllib.load(stream)
    opt = cfg["imitation"]
    for key in ("train_games", "validation_games", "epochs", "batch_size"):
        if type(opt[key]) is not int or opt[key] < 1:
            raise ValueError(f"{key}는 양의 정수여야 합니다.")
    for key in ("train_games", "validation_games"):
        if opt[key] % 2:
            raise ValueError(f"{key}는 양 진영 균형을 위해 짝수로 설정하세요.")
    if not math.isfinite(opt["learning_rate"]) or opt["learning_rate"] <= 0:
        raise ValueError("학습률은 양의 유한한 수여야 합니다.")
    games = cfg["evaluation"]["games"]
    if type(games) is not int or games < 2 or games % 2:
        raise ValueError("평가 경기 수는 2 이상의 짝수여야 합니다.")
    if not cfg["baseline"]["runs"]:
        raise ValueError("기준 run을 하나 이상 지정하세요.")
    return cfg


def source(cfg, seed, name):
    folder = ROOT/"experiment"/"exp2"/name
    update = cfg["baseline"]["update"]
    for filename in (f"update_{update:04d}.pt", "latest.pt", "best.pt"):
        path = folder/filename
        if not path.is_file():
            continue
        data = load_pt(path)
        if data["update"] != update:
            continue
        if data["config"].get("representation") != "raw" or data["size"] != 3 or data["seed"] != seed:
            raise ValueError(f"{path}의 입력 방식, 인원 또는 시드가 다릅니다.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        settings = dict(seed=seed, size=3, teachers=["expand","aggressive"], imitation=cfg["imitation"],
                        source=path.relative_to(ROOT).as_posix(), source_update=update,
                        source_sha256=digest, config=data["config"])
        return settings, data
    raise FileNotFoundError(f"{folder}에 업데이트 {update}의 체크포인트가 없습니다.")


def jobs(folder, cfg, prepare):
    result = []
    for seed, name in cfg["baseline"]["runs"].items():
        settings, base = source(cfg, int(seed), name)
        matches = [p.parent for p in folder.glob("run[0-9]*/settings.json")
                   if json.loads(p.read_text(encoding="utf-8")) == settings]
        if len(matches) > 1:
            raise ValueError(f"시드 {seed}에 동일한 설정의 run이 여러 개 있습니다.")
        if not matches and not prepare:
            raise ValueError(f"시드 {seed}의 현재 설정과 일치하는 학습 결과가 없습니다.")
        run = matches[0] if matches else allocate(folder, "run")
        if prepare:
            if not matches:
                write_json(run/"settings.json", settings)
            if not (run/"before.pt").exists():
                save_pt(run/"before.pt", dict(config=base["config"], size=3, seed=int(seed),
                        update=0, fit=None, policies=base["policies"]))
        result.append(run)
    return result


def main():
    parser = argparse.ArgumentParser(description="exp4 · expand/aggressive 혼합 교사학습")
    parser.add_argument("stage", choices=("train", "plots", "evaluate", "plan"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--trace-all", action="store_true", help="평가 전 경기의 틱별 JSON 저장")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = config()
    if args.worker:
        if args.stage != "train" or not args.worker.startswith("run") or not args.worker[3:].isdigit():
            parser.error("작업 창의 run 이름을 확인하세요.")
        run = HERE/args.worker
        with run_lock(run):
            fit(run, json.loads((run/"settings.json").read_text(encoding="utf-8")))
        return
    if args.stage == "plan":
        for seed, name in cfg["baseline"]["runs"].items():
            info, _ = source(cfg, int(seed), name)
            print(f"시드 {seed} · {info['source']} → 두 교사 혼합")
        return
    if args.stage == "train":
        batch = HERE/".batch"
        batch.mkdir(exist_ok=True)
        with run_lock(batch):
            runs = jobs(HERE, cfg, True)
            workers = [(r, subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                        "train", "--worker", r.name], cwd=ROOT,
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0)) for r in runs]
            failed = [r.name for r, proc in workers if proc.wait() != 0]
            if failed:
                raise SystemExit(f"실패한 실행 {failed}. 같은 명령으로 재개하세요.")
        return
    from results import learning_plots, compare
    runs = jobs(HERE, cfg, False)
    for run in runs:
        if not (run/"latest.pt").exists() or load_pt(run/"latest.pt")["update"] != cfg["imitation"]["epochs"]:
            raise ValueError(f"{run.name}의 학습이 아직 완료되지 않았습니다.")
    if args.stage == "plots":
        learning_plots(HERE, runs)
    else:
        compare(HERE, cfg["evaluation"], runs, args.trace_all)


if __name__ == "__main__":
    main()
