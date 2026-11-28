"""rlTeam의 Python으로 인원별 AE 두 개와 세 시드의 PPO 비교를 실행합니다."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import run  # 저장소의 원본 환경을 같은 경로에서 불러옵니다.
from storage import experiment, load_pt, run_lock

SEEDS = (7, 19, 31)
CONDITIONS = (("ae", 1), ("ae", 3), ("raw", 1), ("raw", 3))


def command(name, stage, *options):
    return [sys.executable, "-B", str(ROOT/"run.py"), stage, "--experiment", name, *map(str, options)]


def training_command(name, mode, size, seed, fit=None):
    args = ["--representation", mode, "--size", size, "--seed", seed]
    if fit:
        args.extend(["--fit", fit])
    return command(name, "train", *args)


def execute(args):
    print(subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def completed_fit(root, cfg, size):
    parent = root/"pretrain"/f"{size}v{size}"
    candidates = sorted((p for p in parent.glob("fit*") if p.name[3:].isdigit()),
                        key=lambda p: int(p.name[3:]), reverse=True)
    for p in candidates:
        if not (p/"encoder.pt").is_file() or not (p/"metrics.csv").is_file():
            continue
        with (p/"metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            epochs = [row["epoch"] for row in csv.DictReader(stream)]
        if not epochs or epochs[-1] != str(cfg["ssl"]["epochs"]):
            continue
        state = load_pt(p/"encoder.pt")
        if state["size"] == size and all(state["config"][k] == cfg[k] for k in ("game", "ssl")):
            return p
    return None


def prepare(root, cfg):
    fits = {}
    for size in (1, 3):
        fit = completed_fit(root, cfg, size)
        if fit is None:
            execute(command(root.name, "pretrain", "--size", size))
            fit = completed_fit(root, cfg, size)
            if fit is None:
                raise RuntimeError(f"{size}vs{size} 인코더 사전학습 기록을 확인하세요.")
        fits[size] = fit
        print(f"{size}vs{size} AE · {fit.relative_to(root)}", flush=True)
    return fits


def existing_run(root, cfg, size, seed, encoder):
    matches = []
    for file in root.glob("run[0-9]*/settings.json"):
        s = json.loads(file.read_text(encoding="utf-8"))
        if (s["config"], s["size"], s["seed"], s["encoder"]) == (cfg, size, seed, encoder):
            matches.append(file.parent)
    if len(matches) > 1:
        raise ValueError(f"같은 조건의 run이 여러 개입니다. 개별 resume 명령을 사용하세요: {matches}")
    return matches[0] if matches else None


def worker(root, cfg, seed, fit_names):
    locks = root/".batch"/f"seed{seed}"
    locks.mkdir(parents=True, exist_ok=True)
    with run_lock(locks):
        for mode, size in CONDITIONS:
            fit = fit_names[size] if mode == "ae" else None
            source = f"pretrain/{size}v{size}/{fit}/encoder.pt" if fit else None
            previous = existing_run(root, dict(cfg, representation=mode), size, seed, source)
            if previous:
                print(f"{mode} · {size}vs{size} · 시드 {seed} · {previous.name} 이어서 확인", flush=True)
                execute(command(root.name, "resume", "--run", previous.name))
            else:
                execute(training_command(root.name, mode, size, seed, fit))


def main():
    parser = argparse.ArgumentParser(description="인원별 AE 준비 후 3개 시드 병렬 실행")
    parser.add_argument("--experiment", default=HERE.parent.name)
    parser.add_argument("--plan", action="store_true", help="명령만 출력")
    parser.add_argument("--worker", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--fits", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, cfg = experiment(args.experiment)
    if cfg.get("representation") != "ae" or "online_validation" not in cfg:
        parser.error("마스크 없는 AE와 학습 중 검증을 설정한 실험을 지정하세요.")
    if args.plan:
        for size in (1, 3):
            print(subprocess.list2cmdline(command(root.name, "pretrain", "--size", size)))
        for seed in SEEDS:
            for mode, size in CONDITIONS:
                print(subprocess.list2cmdline(training_command(root.name, mode, size, seed)))
        return
    if args.worker:
        if not args.fits:
            parser.error("작업 창에는 준비된 인코더 두 개가 필요합니다.")
        worker(root, cfg, args.worker, dict(zip((1, 3), args.fits)))
        return
    if os.name != "nt":
        parser.error("세 개의 새 터미널을 여는 일괄 실행은 Windows에서 지원합니다.")
    lock = root/".batch"/"launcher"
    lock.mkdir(parents=True, exist_ok=True)
    with run_lock(lock):
        fits = prepare(root, cfg)
        procs = []
        for seed in SEEDS:
            args = [sys.executable, "-B", str(HERE), "--experiment", root.name, "--worker", str(seed),
                    "--fits", fits[1].name, fits[3].name]
            procs.append((seed, subprocess.Popen(args, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)))
        failed = [seed for seed, proc in procs if proc.wait() != 0]
        if failed:
            raise SystemExit(f"실패한 시드 {failed}. 같은 명령을 다시 실행하면 마지막 저장부터 이어갑니다.")
    print("세 시드의 학습을 완료했습니다.", flush=True)


if __name__ == "__main__":
    main()
