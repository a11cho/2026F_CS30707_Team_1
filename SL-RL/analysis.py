"""학습 그래프, 일괄 평가, 보고서, 대표 리플레이의 명령행 진입점입니다."""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arena import BOTS, SIDES, seed_for
from evaluation import evaluate
from storage import experiment


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Run:
    path: Path
    size: int
    seed: int
    settings: dict

    @property
    def name(self):
        return self.path.name


@dataclass
class Evaluation:
    run: Run
    rival: Run | None
    path: Path
    games: list


def find_runs(root, requested):
    if requested:
        if any(not re.fullmatch(r"run[1-9][0-9]*", name) for name in requested):
            raise ValueError("run1 형식으로 실행 이름을 지정하세요.")
        paths = [root / name for name in dict.fromkeys(requested)]
    else:
        paths = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"run[1-9][0-9]*", p.name)]
    runs = []
    for p in sorted(paths, key=lambda d: int(d.name[3:])):
        s = metadata(p / "settings.json")
        runs.append(Run(p, s["size"], s["seed"], s))
    if not runs:
        raise ValueError("학습 run을 찾지 못했습니다.")
    return runs


def complete_rows(folder, run, rival, checkpoint, update, options, games):
    """진영·학습 팀·상대·시드 조합이 정확히 한 번씩 있는 완료 결과를 찾습니다."""
    try:
        s = metadata(folder / "settings.json")
        model = s["model"]
        if (s["update"] != update or s["model_seed"] != run.seed or s["model_training_size"] != run.size
                or s["games_per_pair"] != games or s["evaluation_seed"] != options["seed"]
                or s["evaluation_size"] != 3 or not s["greedy"]):
            return None
        # 기존 맞대결 결과에는 상대 업데이트 번호가 별도로 기록되지 않아 고정 파일명으로 대조합니다.
        aliases = [checkpoint] if rival else [checkpoint, "latest.pt", "best.pt"]
        if model not in {f"{run.name}/{name}" for name in aliases}:
            return None
        opponents = [f"{rival.name}/{checkpoint}"] if rival else options["opponents"]
        if sorted(s["opponents"]) != sorted(opponents) or s["other_seed"] != (rival.seed if rival else None):
            return None
        rows = read_rows(folder / "games.csv")
        other_teams = SIDES if rival else ("",)
        seeds = [seed_for(options["seed"], i, 71) for i in range(games // 2)]
        expected = Counter((team, opp, other, side, seed) for team in SIDES for opp in opponents
                           for other in other_teams for side in SIDES for seed in seeds)
        actual = Counter()
        for row in rows:
            if row["model"] != model or int(row["size"]) != run.size or int(row["train_seed"]) != run.seed:
                return None
            if int(row["ticks"]) <= 0 or not all(math.isfinite(float(row[k])) for k in ("mean_area_share", "per_agent_return")):
                return None
            gap = int(row["final_gap"])
            if tuple(int(row[k]) for k in ("win", "draw", "loss")) != (int(gap>0), int(gap==0), int(gap<0)):
                return None
            actual[row["trained_team"], row["opponent"], row["opponent_team"], row["field"], int(row["seed"])] += 1
        return rows if actual == expected else None
    except (FileNotFoundError, KeyError, ValueError, csv.Error):
        return None


def collect(root, cfg, runs, update, games, execute=False, refresh=False):
    checkpoint = f"update_{update:04d}.pt"
    pairs = [(run, None) for run in runs]
    pairs.extend((a,b) for a in runs for b in runs if a.size == 1 and b.size == 3 and a.seed == b.seed
                 and a.settings["config"].get("representation") == b.settings["config"].get("representation"))
    data = []
    for run, rival in pairs:
        parent = root/"comparison" if rival else run.path/"evaluation"
        folders = [p for p in parent.glob("eval*") if p.is_dir() and re.fullmatch(r"eval[1-9][0-9]*",p.name)]
        rows = None
        for folder in sorted(folders, key=lambda p: int(p.name[4:]), reverse=True):
            if refresh:
                break
            rows = complete_rows(folder, run, rival, checkpoint, update, cfg["evaluation"], games)
            if rows is not None:
                break
        if rows is None:
            if not execute:
                raise ValueError(f"{run.name} / {rival.name if rival else '고정 상대'} 결과가 없습니다. 같은 옵션으로 evaluate를 실행하세요.")
            folder = evaluate(root, cfg, run.name, checkpoint, rival.name if rival else None, games,
                              cfg["evaluation"]["opponents"], False)
            rows = complete_rows(folder, run, rival, checkpoint, update, cfg["evaluation"], games)
            if rows is None:
                raise RuntimeError(f"완료된 경기 목록이 예상과 다릅니다: {folder}")
        print(f"{folder.relative_to(root)} · {len(rows)}경기", flush=True)
        data.append(Evaluation(run, rival, folder, rows))
    return data


def main():
    p = argparse.ArgumentParser(description="SL-RL 학습 및 3vs3 평가 분석")
    subs = p.add_subparsers(dest="stage", required=True)
    for name in ("plots","evaluate","report","replays"):
        sub = subs.add_parser(name)
        sub.add_argument("--experiment", default="exp1")
        sub.add_argument("--condition", help="AE 그리드 조건 이름 (예: mask09_z16)")
        sub.add_argument("--runs", nargs="+", help="생략하면 해당 실험의 모든 run")
        sub.add_argument("--update", type=int, help="분석 종료 시점 및 평가 체크포인트")
        if name != "plots":
            sub.add_argument("--games", type=int, help="정책·상대 조합당 양 진영 합계")
        if name == "evaluate":
            sub.add_argument("--refresh", action="store_true", help="조건이 같은 완료 결과도 새로 평가")
        if name == "replays":
            sub.add_argument("--fps", type=int, default=5)
    args = p.parse_args()
    root, cfg = experiment(args.experiment)
    update = cfg["ppo"]["updates"] if args.update is None else args.update
    bots = cfg["evaluation"]["opponents"]
    if not bots or len(set(bots)) != len(bots) or any(b not in BOTS for b in bots):
        p.error("config.toml에서 평가 상대 목록을 확인하세요.")
    if update < 1:
        p.error("--update는 양수여야 합니다.")
    torch.set_num_threads(1)
    if "ae_grid" in cfg:
        from experiment.exp3.grid_analysis import grid_analysis
        return grid_analysis(p, args, root, cfg, update)
    if args.condition:
        p.error("--condition은 AE 그리드 실험에서 사용하세요.")
    runs = find_runs(root, args.runs)
    out = root/"result"/f"analysis_u{update:04d}"
    if args.runs:
        out /= "_".join(r.name for r in runs)
    groups = {}
    for run in runs:
        mode = run.settings["config"].get("representation")
        groups.setdefault(mode, []).append(run)
    for mode, group in groups.items():
        analyze_group(p, args, root, cfg, group, update, out/mode if mode else out)


def analyze_group(p, args, root, cfg, runs, update, out):
    if args.stage == "plots":
        from analysis_plots import training_plots
        training_plots(root, out, runs, update)
    else:
        games = cfg["evaluation"]["games"] if args.games is None else args.games
        if games < 2 or games % 2:
            p.error("--games는 2 이상의 짝수여야 합니다.")
        if args.stage == "replays" and not 1 <= args.fps <= 30:
            p.error("--fps는 1~30이어야 합니다.")
        data = collect(root, cfg, runs, update, games, execute=args.stage=="evaluate",
                       refresh=getattr(args, "refresh", False))
        if args.stage == "evaluate":
            return
        opts = cfg["evaluation"]
        dest = out / f"eval_{opts['seed']}_{games}_{'-'.join(opts['opponents'])}"
        dest.mkdir(parents=True, exist_ok=True)
        if args.stage == "report":
            from analysis_plots import report
            report(root, out, dest, runs, update, data, opts, games)
        else:
            from analysis_replays import render_selection
            render_selection(dest/"replays", update, data, args.fps)
        out = dest
    print(out, flush=True)
    if args.stage == "report":
        return data, dest


if __name__ == "__main__":
    main()
