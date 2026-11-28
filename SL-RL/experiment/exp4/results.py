"""학습 곡선 및 같은 평가 경기에서의 교사학습 전후 승률을 저장합니다."""
import json
from collections import defaultdict

import numpy as np
from analysis import read_rows
from analysis_plots import plt, save, aggregate
from evaluation import evaluate
from imitation import TEACHERS
from storage import allocate, write_csv, write_json


def learning_plots(folder, runs):
    dest = folder/"result"/("learning_" + "_".join(run.name for run in runs))
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 2.8*len(runs)), squeeze=False, layout="constrained")
    for run, (left, right) in zip(runs, axes):
        settings = json.loads((run/"settings.json").read_text(encoding="utf-8"))
        data = read_rows(run/"training.csv")
        for side, color in (("red", "firebrick"), ("blue", "steelblue")):
            rows = [row for row in data if row["team"] == side]
            x = [int(row["epoch"]) for row in rows]
            left.plot(x, [float(r["val_loss"]) for r in rows], color=color, label=f"{side} 검증")
            left.plot(x[1:], [float(r["train_loss"]) for r in rows[1:]], "--", color=color, label=f"{side} 학습")
            right.plot(x, [100*float(r["val_accuracy"]) for r in rows], color=color, label=side)
        left.set(ylabel=f"시드 {settings['seed']} · 손실", xlabel="epoch")
        right.set(xlabel="epoch", ylabel="검증 행동 일치율 (%)", ylim=(0,100))
        left.legend(fontsize=8, ncol=2)
        right.legend(fontsize=8)
    fig.suptitle("두 교사 행동 모방 · epoch 0은 기존 PPO")
    save(fig, dest/"learning.png")
    print(dest/"learning.png")


def compare(folder, options, runs, trace_all=False):
    dest = allocate(folder/"result", "evaluation")
    raw, summary, sources = defaultdict(list), [], []
    for run in runs:
        meta = json.loads((run/"settings.json").read_text(encoding="utf-8"))
        for stage, filename in (("before", "before.pt"), ("after", "best.pt")):
            path = evaluate(folder, {"evaluation":options}, run.name, filename, None,
                            options["games"], list(TEACHERS), False, trace_all=trace_all)
            rows = read_rows(path/"games.csv")
            sources.append(dict(run=run.name, seed=meta["seed"], stage=stage,
                                path=path.relative_to(folder).as_posix()))
            for row in rows:
                raw[meta["seed"],row["opponent"],stage].append(row)
    seeds = sorted({key[0] for key in raw})
    for seed in [*seeds, "all"]:
        members = seeds if seed == "all" else [seed]
        for opponent in TEACHERS:
            scores = []
            for stage in ("before", "after"):
                rows = [row for member in members for row in raw[member, opponent, stage]]
                scores.append(aggregate(rows))
            old, new = scores
            summary.append(dict(seed=seed, opponent=opponent, games_per_stage=old["games"],
                                before_wins=old["wins"], before_draws=old["draws"], before_losses=old["losses"],
                                after_wins=new["wins"], after_draws=new["draws"], after_losses=new["losses"],
                                before_win_rate=old["win_rate"], after_win_rate=new["win_rate"],
                                delta_percentage_points=100*(new["win_rate"]-old["win_rate"]),
                                before_gap=old["final_gap"], after_gap=new["final_gap"]))
    write_csv(dest/"win_rates.csv", summary)
    write_json(dest/"sources.json", dict(evaluation=options, checkpoints=sources))
    fig, axes = plt.subplots(1, 2, figsize=(10,4), layout="constrained")
    for opponent, ax in zip(TEACHERS, axes):
        rows = [r for r in summary if r["opponent"] == opponent]
        x = np.arange(len(rows))
        for stage, offset, label in (("before", -.19, "기존 PPO"), ("after", .19, "교사학습 후")):
            bars = ax.bar(x+offset, [100*r[f"{stage}_win_rate"] for r in rows], .36, label=label)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set(title=f"{opponent} 상대", ylabel="승률 (%)", ylim=(0,109),
               xticks=x, xticklabels=[f"시드 {r['seed']}" if r["seed"]!="all" else "전체" for r in rows])
        ax.legend(fontsize=8)
    save(fig, dest/"win_rates.png")
    print(dest)
