"""학습·평가 실행: python -B -m smoke_test_mc_td.run --preset quick|full"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from .config import RunConfig, condition_id
from .learning import load_models, save_models
from .runner import evaluate, summarize, train

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def selection_key(candidate):
    """보상 크기 대신 validation 점수와 최종 면적 비율로 후보 순위를 매깁니다."""
    return candidate["validation"]["score"], candidate["validation"]["area_fraction"]


def run_grid(config, output=None):
    """모든 조건·알고리즘·seed를 학습하고 validation 선정과 test 평가를 저장합니다."""
    config.validate()
    output = Path(output).resolve() if output else ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("출력 폴더는 smoke_test_mc_td 내부의 하위 폴더여야 합니다.")
    output.mkdir(parents=True, exist_ok=False)
    sources = [*ROOT.glob("*.py"), * (ROOT.parent / "territory_rl").glob("*.py")]
    hashes = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output / "manifest.json", dict(config=asdict(config), python=sys.version, numpy=np.__version__,
               platform=platform.platform(), encoding="compact_v1", source_sha256=hashes,
               selection="validation 점수, 최종 면적 비율 순으로 선택하며 완전 동률은 첫 seed를 유지합니다.",
               policy="학습자 역할별 독립 Q-table, 진영 교대, 파라미터 공유 없음",
               evaluation="greedy 정책과 무작위 동률 처리, 고정된 별도 평가 seed와 양쪽 진영 경기",
               status="running"))
    groups, selected = [], []
    total = len(list(config.conditions())) * len(config.algorithms) * len(config.seeds)
    completed = 0
    for experiment in config.conditions():
        cid = condition_id(experiment)
        for algorithm in config.algorithms:
            candidates = []
            for seed in config.seeds:
                directory = output / cid / algorithm / f"seed_{seed}"
                directory.mkdir(parents=True)
                models, training = train(config, experiment, algorithm, seed)
                checkpoint = directory / "model.json.gz"
                save_models(checkpoint, models, dict(config=asdict(config), experiment=asdict(experiment),
                            algorithm=algorithm, seed=seed, encoding="compact_v1"))
                # 저장된 체크포인트를 다시 읽어 평가합니다.
                models, _ = load_models(checkpoint)
                validation_rows = evaluate(config, experiment, models, "validation")
                baseline = [type(m)(algorithm, config.alpha, config.gamma) for m in models]
                baseline_rows = evaluate(config, experiment, baseline, "validation")
                candidate = dict(condition=cid, algorithm=algorithm, seed=seed,
                                 checkpoint=str(checkpoint.relative_to(output)), validation=summarize(validation_rows),
                                 untrained_validation=summarize(baseline_rows),
                                 updates=sum(m.updates for m in models), q_states=sum(len(m.q) for m in models))
                write_csv(directory / "training.csv", training)
                write_csv(directory / "validation.csv", validation_rows)
                write_csv(directory / "untrained_validation.csv", baseline_rows)
                write_json(directory / "summary.json", candidate)
                candidates.append(candidate)
                completed += 1
                print(f"[{completed}/{total}] {cid} {algorithm} seed={seed}: 검증점수={candidate['validation']['score']:.3f}", flush=True)
            best = max(candidates, key=selection_key)
            models, _ = load_models(output / best["checkpoint"])
            test_rows = evaluate(config, experiment, models, "test")
            write_csv(output / cid / algorithm / "selected_test.csv", test_rows)
            scores = [c["validation"]["score"] for c in candidates]
            groups.append(dict(condition=cid, algorithm=algorithm, validation_score_mean=float(np.mean(scores)),
                               validation_score_std=float(np.std(scores, ddof=1)) if len(scores)>1 else 0.0,
                               candidates=candidates))
            selected.append(dict(**best, test=summarize(test_rows)))
            write_json(output / "results.json", dict(groups=groups, selected=selected))
    # 전체 대표 모델도 test 점수가 아닌 validation만으로 선택합니다.
    representatives = {a: max((s for s in selected if s["algorithm"] == a), key=selection_key)
                       for a in config.algorithms}
    results = dict(groups=groups, selected=selected, representatives=representatives,
                   note="최고 seed의 test는 선택 모델의 성능이며 알고리즘의 5-seed 평균 성능이 아닙니다.")
    write_json(output / "results.json", results)
    flat = [dict(condition=s["condition"], algorithm=s["algorithm"], seed=s["seed"],
                 validation_score=s["validation"]["score"], test_score=s["test"]["score"],
                 test_area=s["test"]["area_fraction"], known_state_fraction=s["test"]["known_state_fraction"])
            for s in selected]
    write_csv(output / "selected.csv", flat)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(f"결과 폴더: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--algorithm", choices=("mc", "sarsa", "both"), default="both")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--learners", type=int, choices=(1, 3), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = RunConfig(learner_count=args.learners)
    if args.preset == "quick":
        cfg = replace(cfg, episodes=4, ticks=12, validation_pairs=1, test_pairs=1,
                      reward_specs=(("individual", "gross", 0.0),), eval_opponents=("random",))
    if args.episodes is not None:
        cfg = replace(cfg, episodes=args.episodes)
    if args.algorithm != "both":
        cfg = replace(cfg, algorithms=(args.algorithm,))
    run_grid(cfg, args.output)


if __name__ == "__main__":
    main()
