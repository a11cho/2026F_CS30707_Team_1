"""구조별 훈련 곡선과 같은 경기 시드의 사전·사후 승률을 기록합니다."""
import collections
import json
import numpy as np
from analysis import read_rows
from analysis_plots import aggregate, plt, save
from evaluation import evaluate
from fixed_ppo import OPPONENTS
from storage import allocate, write_csv, write_json


def description(run):
    cfg = json.loads((run/"settings.json").read_text(encoding="utf-8"))
    return str(cfg["config"]["ppo"]["hidden_layers"]), cfg["seed"]


def plots(folder, runs):
    out = allocate(folder/"result", "learning")
    groups = collections.defaultdict(list)
    for run in runs:
        shape, seed = description(run)
        groups[shape].append((seed, run))
    fig, axes = plt.subplots(len(groups), 4, figsize=(15,3*len(groups)), squeeze=False, layout="constrained")
    for (shape, members), row in zip(groups.items(), axes):
        for seed, run in members:
            train = read_rows(run/"training.csv")
            by_step = collections.defaultdict(list)
            for item in train:
                by_step[int(item["update"])].append(item)
            for ax, metric in zip(row, ("actor_loss","critic_loss")):
                ax.plot(sorted(by_step), [np.mean([float(r[metric]) for r in by_step[u]])
                                         for u in sorted(by_step)], label=str(seed), linewidth=1)
                ax.set_title(f"{shape} · {metric}")
            val = read_rows(run/"validation.csv")
            for ax, opp in zip(row[2:], OPPONENTS):
                ax.plot([int(r["update"]) for r in val], [100*float(r[opp]) for r in val], label=str(seed))
                ax.set(title=f"{shape} · {opp} 검증 승률 (%)", ylim=(0,100))
        for ax in row:
            ax.set_xlabel("고정 봇 PPO 업데이트")
            ax.legend(fontsize=8)
    save(fig, out/"training.png")
    print(out)


def compare(folder, runs, options, checkpoint, trace):
    out = allocate(folder/"result", "evaluation")
    groups = collections.defaultdict(list)
    sources = []
    shapes, seeds = [], set()
    for run in runs:
        shape, seed = description(run)
        if shape not in shapes:
            shapes.append(shape)
        seeds.add(seed)
        for stage, file in (("before","pretrained.pt"), ("after",checkpoint)):
            path = evaluate(folder, {"evaluation":options}, run.name, file, None, options["games"],
                            list(OPPONENTS), False, trace_all=trace)
            sources.append(dict(run=run.name, seed=seed, architecture=shape, stage=stage,
                                checkpoint=file, evaluation=path.relative_to(folder).as_posix()))
            for row in read_rows(path/"games.csv"):
                groups[shape,seed,row["opponent"],stage].append(row)
    table = []
    for shape in shapes:
        for seed in [*sorted(seeds), "all"]:
            members = sorted(seeds) if seed=="all" else [seed]
            for opp in OPPONENTS:
                result = dict(architecture=shape, seed=seed, opponent=opp)
                for stage in ("before","after"):
                    stats = aggregate([r for s in members for r in groups[shape,s,opp,stage]])
                    for name in ("games","wins","draws","losses","win_rate","final_gap"):
                        result[f"{stage}_{name}"] = stats[name]
                result["change_pp"] = 100*(result["after_win_rate"] - result["before_win_rate"])
                table.append(result)
    write_csv(out/"win_rates.csv", table)
    write_json(out/"sources.json", dict(evaluation=options, runs=sources))
    fig, axes = plt.subplots(1,2, figsize=(11,4), layout="constrained")
    x = np.arange(len(shapes))
    for ax, opp in zip(axes, OPPONENTS):
        data = [r for r in table if r["seed"]=="all" and r["opponent"]==opp]
        for stage, shift, label in (("before",-.19,"자기대전 후"),("after",.19,"고정 봇 학습 후")):
            bars = ax.bar(x+shift,[100*r[f"{stage}_win_rate"] for r in data],.36,label=label)
            ax.bar_label(bars,fmt="%.1f")
        ax.set(title=opp, ylabel="승률 (%)", xticks=x, xticklabels=shapes, ylim=(0,109))
        ax.axhline(80,color="gray",ls=":",lw=1)
        ax.legend()
    save(fig, out/"win_rates.png")
    print(out)
