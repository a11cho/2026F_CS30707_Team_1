"""Re-evaluate existing PPO comparisons and export the seven-scene grid."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from .cooperation_scenes import SCENES, summarize_scenes


MODELS = {
    "baseline": "baseline_eval_files",
    "prediction_only": "prediction_eval_files",
    "receiver_backprop": "backprop_eval_files",
}
MODES = {"on": "training_mode", "off": "without_teammate_proposals"}


@dataclass(frozen=True)
class Case:
    trained: str
    board: str
    respawn: str
    model: str
    seed: int
    reference: Path

    @property
    def condition(self):
        return self.trained, self.board, self.respawn

    @property
    def filename(self):
        return f"{self.trained}__{self.board}-{self.respawn}__{self.model}__seed{self.seed}.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(report, old, case):
    for key in ("checkpoint", "seed", "update", "evaluation_game", "training_game", "pairs_per_opponent"):
        if report[key] != old[key]:
            raise ValueError(f"{case.filename}: {key} differs from the reference")
    if report["training_seed"] != case.seed:
        raise ValueError(f"{case.filename}: training seed differs from the reference")
    for key in MODES.values():
        actual, expected = report[key]["overall"], old[key]["overall"]
        for field in ("games", "wins", "draws", "losses", "respawns", "deaths", "cooperation"):
            if actual[field] != expected[field]:
                raise ValueError(f"{case.filename}: {key}.{field} changed during evaluation")
        for scene in actual["cooperation_scenes"].values():
            if not 0 <= scene["count"] <= scene["opportunities"]:
                raise ValueError(f"{case.filename}: invalid cooperation count")


def run_case(case, raw):
    old = read_json(case.reference)
    path = raw / case.filename
    if path.exists():
        report = read_json(path)
        verify(report, old, case)
        return case, report
    temporary = path.with_suffix(".tmp")
    command = [
        sys.executable, "-m", "ppo", "evaluate", old["checkpoint"], "--cooperation-scenes",
        "--map-size", *case.board.split("x"), "--respawn", case.respawn,
        "--seed", str(old["seed"]), "--pairs", str(old["pairs_per_opponent"]),
        "--opponents", *old["training_mode"]["by_opponent"],
    ]
    with temporary.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, check=True)
    report = read_json(temporary)
    verify(report, old, case)
    temporary.replace(path)
    print(case.filename, flush=True)
    return case, report


def aggregate(results):
    by_condition = {}
    totals = {model: {mode: Counter() for mode in MODES} for model in MODELS}
    for case, report in results:
        group = by_condition.setdefault(case.condition, {})
        model = group.setdefault(case.model, {"raw": [], **{mode: Counter() for mode in MODES}})
        model["raw"].append(case.filename)
        for mode, key in MODES.items():
            for name, metric in report[key]["overall"]["cooperation_scenes"].items():
                for field in ("count", "opportunities"):
                    model[mode][name, field] += metric[field]
                    totals[case.model][mode][name, field] += metric[field]
    return {
        "scenes": SCENES,
        "conditions": [
            {
                "trained": trained, "test_board": board, "test_respawn": respawn,
                "models": {name: {"raw": model["raw"], **{mode: summarize_scenes(model[mode]) for mode in MODES}}
                           for name, model in group.items()},
            }
            for (trained, board, respawn), group in by_condition.items()
        ],
        "pooled": {model: {mode: summarize_scenes(counts) for mode, counts in modes.items()}
                   for model, modes in totals.items()},
    }


def write_csv(path, summary):
    fields = ("trained", "test_board", "test_respawn", "model", "sharing", "scene", "opportunities",
              "successes", "success_rate")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for case in summary["conditions"]:
            for model, results in case["models"].items():
                for mode in MODES:
                    for name, description in SCENES.items():
                        metric = results[mode][name]
                        writer.writerow((case["trained"], case["test_board"], case["test_respawn"],
                                         model, mode, description, metric["opportunities"],
                                         metric["count"], metric["rate"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    reference = read_json(args.reference)
    seeds = reference["training_seeds"]
    cases = [
        Case(row["trained"], row["test_board"], row["test_respawn"], name, seed, Path(path))
        for row in reference["scenarios"]
        for name, files in MODELS.items()
        for seed, path in zip(seeds, row[files], strict=True)
    ]
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        results = list(workers.map(lambda case: run_case(case, raw), cases))
    summary = aggregate(results)
    summary["training_seeds"] = seeds
    summary["evaluation_reports"] = len(results)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    write_csv(args.output / "comparison.csv", summary)


if __name__ == "__main__":
    main()
