"""exp1의 시드별 학습, 병렬 실행, 평가 및 그래프 저장 진입점입니다."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from model import ROOT, SequentialPolicy, ppo_update
from matches import evaluate, validate
from arena import SIDES, seed_for
from storage import load_pt, run_lock, save_pt, write_csv, write_json
from training import sync_validation
import torch


def policies_for(cfg):
    return [SequentialPolicy(cfg["game"], cfg["ppo"]) for _ in SIDES]


def run_path(exp, seed):
    return exp/"runs"/f"seed{seed}"


def train(exp, cfg, seed):
    run = run_path(exp, seed)
    run.mkdir(parents=True, exist_ok=True)
    opt = cfg["ppo"]
    with run_lock(run):
        torch.manual_seed(seed)
        nets = policies_for(cfg)
        opts = [torch.optim.Adam(net.parameters(), lr=opt["learning_rate"]) for net in nets]
        start, history, validation = 0, [], []
        latest = run/"latest.pt"
        if latest.exists():
            state = load_pt(latest)
            if state["config"] != cfg or state["seed"] != seed:
                raise ValueError("저장된 run과 설정이 다릅니다. 별도 실험 경로를 사용하세요.")
            start, history, validation = state["update"], state["history"], state["validation_history"]
            for net, optimizer, weights, moments in zip(nets, opts, state["policies"], state["optimizers"]):
                net.load_state_dict(weights)
                optimizer.load_state_dict(moments)
            torch.set_rng_state(state["rng"])
            sync_validation(run, state)
            write_csv(run/"training.csv", history)
        write_json(run/"settings.json", dict(config=cfg, seed=seed))
        for step in range(start+1, opt["updates"]+1):
            begin = time.perf_counter()
            n = opt["episodes_per_update"]
            seeds = [seed_for(seed_for(seed,91), i) for i in range((step-1)*n, step*n)]
            metrics = ppo_update(cfg["game"], seeds, nets, opts, opt)
            seconds = time.perf_counter()-begin
            history.extend(dict(update=step, episodes=step*n, team=side, seconds=seconds, **row)
                           for side,row in zip(SIDES, metrics))
            write_csv(run/"training.csv", history)
            if step % opt["checkpoint_every"] == 0 or step == opt["updates"]:
                score = validate(cfg, nets)
                validation.append(dict(update=step, score=score))
                state = dict(config=cfg, seed=seed, update=step, history=history, validation_history=validation,
                             policies=[net.state_dict() for net in nets], optimizers=[o.state_dict() for o in opts],
                             rng=torch.get_rng_state())
                save_pt(latest, state)
                sync_validation(run, state)
                print(f"시드 {seed} · {step*n}경기 · 검증 점수 {score:.3f}", flush=True)
        print(f"완료 {run}", flush=True)


def plot(exp, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ("per_agent_return", "final_gap", "actor_loss", "critic_loss", "entropy", "approx_kl")
    out = exp/"result"/"training"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15,8), constrained_layout=True)
    valfig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
    for seed in cfg["seeds"]:
        run = run_path(exp, seed)
        with (run/"training.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for side in SIDES:
            data = [r for r in rows if r["team"] == side]
            for key, panel in zip(metrics, axes.flat):
                panel.plot([int(r["episodes"]) for r in data], [float(r[key]) for r in data],
                           lw=.8, alpha=.8, label=f"{seed} {side}")
                panel.set(xlabel="Episodes", title=key)
        with (run/"validation.csv").open(encoding="utf-8-sig") as f:
            vals = list(csv.DictReader(f))
        ax.plot([int(r["update"])*cfg["ppo"]["episodes_per_update"] for r in vals],
                [float(r["score"]) for r in vals], marker=".", label=f"Seed {seed}")
    axes.flat[0].legend(fontsize=7)
    ax.set(xlabel="Episodes", ylabel="Validation score", ylim=(0,1))
    ax.legend()
    fig.savefig(out/"training.png", dpi=160)
    valfig.savefig(out/"validation.png", dpi=160)
    plt.close("all")
    print(out)


def evaluate_runs(exp, cfg, checkpoint):
    entries = []
    for seed in cfg["seeds"]:
        state = load_pt(run_path(exp, seed)/checkpoint)
        if state["seed"] != seed or any(state["config"][k] != cfg[k] for k in ("ppo","game")):
            raise ValueError(f"설정과 체크포인트가 일치하지 않습니다: {seed}")
        nets = policies_for(cfg)
        for net, weights in zip(nets, state["policies"]):
            net.load_state_dict(weights)
            net.eval()
        entries.append((seed, nets, state["update"]))
    return evaluate(ROOT, exp, cfg, entries, checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch","train","plot","evaluate"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", choices=("latest.pt","best.pt"), default="latest.pt")
    args = parser.parse_args()
    cfg = tomllib.loads((HERE/"config.toml").read_text(encoding="utf-8-sig"))
    torch.set_num_threads(1)
    if args.action == "launch":
        if os.name != "nt":
            parser.error("멀티 터미널 실행은 Windows 전용입니다. train --seed를 사용하세요.")
        for seed in cfg["seeds"]:
            subprocess.Popen([sys.executable,"-B",__file__,"train","--seed",str(seed)],
                             cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif args.action == "train":
        if args.seed not in cfg["seeds"]:
            parser.error("config.toml에 있는 --seed를 선택하세요.")
        train(HERE, cfg, args.seed)
    elif args.action == "plot":
        plot(HERE, cfg)
    else:
        evaluate_runs(HERE, cfg, args.checkpoint)


if __name__ == "__main__":
    main()
