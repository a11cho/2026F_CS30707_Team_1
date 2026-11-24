"""16차원 복사본을 확인하고 나머지 AE 그리드를 시드별 Windows 콘솔에서 실행합니다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import run  # 저장소의 원본 환경을 불러오는 경로를 설정합니다.
import torch
from experiment.exp2.launch import completed_fit, existing_run
from experiment.exp3.copied_results import import_baseline
from experiment.exp3.grid import SIZES, conditions, copied_source, label
from pretrain import pretrain
from storage import experiment, run_lock
from training import train


def prepare(root, cfg, variants, *, import_only=False):
    fits = {}
    for key, variant in variants.items():
        source = copied_source(cfg, key)
        if source:
            import_baseline(root, source, key, variant, cfg["ae_grid"]["seeds"])
        elif not import_only:
            for size in SIZES:
                fit = completed_fit(root, variant, size)
                if fit is None:
                    fit = pretrain(root, variant, size)
                fits[f"{label(key)}_{size}"] = fit.name
                print(f"{label(key)} · {size}대{size} · {fit.relative_to(root)}", flush=True)
    return fits


def worker(root, cfg, variants, seed, fits):
    lock = root/".batch"/f"seed{seed}"
    lock.mkdir(parents=True, exist_ok=True)
    with run_lock(lock):
        for key, variant in variants.items():
            if copied_source(cfg, key):
                continue
            for size in SIZES:
                fit = fits[f"{label(key)}_{size}"]
                source = f"pretrain/{size}v{size}/{fit}/encoder.pt"
                previous = existing_run(root, variant, size, seed, source)
                print(f"{label(key)} · {size}대{size} · 시드 {seed} · {previous.name if previous else '새 run'}", flush=True)
                if previous:
                    train(root, variant, size, seed, resume=previous.name)
                else:
                    train(root, variant, size, seed, fit_name=fit)


def main():
    parser = argparse.ArgumentParser(description="AE 그리드 준비와 시드별 병렬 PPO 학습")
    parser.add_argument("--experiment", default=HERE.parent.name)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true", help="조건·재사용 여부·학습 예산 표시")
    actions.add_argument("--import-only", action="store_true", help="기준 결과 복사 및 검증만 실행")
    actions.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fits", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, cfg = experiment(args.experiment)
    variants = conditions(cfg)
    seeds = cfg["ae_grid"]["seeds"]
    torch.set_num_threads(1)
    if args.plan:
        games = cfg["ppo"]["updates"] * cfg["ppo"]["episodes_per_update"]
        new_runs = 0
        for key in variants:
            source = copied_source(cfg, key)
            count = len(seeds)*len(SIZES)
            if not source:
                new_runs += count
            print(f"{label(key)} · {source+' 복사' if source else '학습/재개'} · {count} runs")
        print(f"터미널 시드 {seeds} · 신규 조건 {new_runs} runs × {games}경기 = {new_runs*games:,}경기")
        return
    if args.worker is not None:
        if args.worker not in seeds or not args.fits:
            parser.error("시드 또는 준비된 인코더 정보가 없습니다.")
        worker(root, cfg, variants, args.worker, json.loads(args.fits))
        return
    if not args.import_only and os.name != "nt":
        parser.error("새 콘솔을 여는 일괄 실행은 Windows에서 사용하세요.")
    lock = root/".batch"/"launcher"
    lock.mkdir(parents=True, exist_ok=True)
    with run_lock(lock):
        fits = prepare(root, cfg, variants, import_only=args.import_only)
        if args.import_only or not fits:
            return
        procs = []
        for seed in seeds:
            cmd = [sys.executable, "-B", str(HERE), "--experiment", root.name, "--worker", str(seed),
                   "--fits", json.dumps(fits)]
            procs.append((seed, subprocess.Popen(cmd, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)))
        failed = [seed for seed, proc in procs if proc.wait() != 0]
        if failed:
            raise SystemExit(f"중단된 시드 {failed}. 같은 명령으로 재개할 수 있습니다.")
    print("AE 그리드 학습을 완료했습니다.", flush=True)


if __name__ == "__main__":
    main()
