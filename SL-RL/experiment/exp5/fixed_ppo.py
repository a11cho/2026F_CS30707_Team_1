"""구조별 자기대전 사전학습과 고정 봇 상대 승리 최적화."""
import copy
import json

import numpy as np
import torch
from arena import Arena, SIDES, bot, seed_for
from evaluation import PolicyPair, match
from storage import load_pt, save_pt, write_csv
from training import decide, make_policy, restore_encoder, rollout, trajectory, update

OPPONENTS = ("expand", "aggressive")


def reference(root, cfg, seed):
    """같은 초기화·규칙·자기대전 예산의 단층 최종 가중치만 재사용합니다."""
    opt = cfg["ppo"].copy()
    widths = opt.pop("hidden_layers")
    if not cfg["reuse_matching_exp2"] or widths != [128]:
        return None
    matches = []
    for file in (root.parent/"exp2").glob("run*/settings.json"):
        info = json.loads(file.read_text(encoding="utf-8"))
        original = info["config"]
        if (info["seed"] == seed and info["size"] == 3 and original.get("representation") == "raw"
                and original["game"] == cfg["game"] and original["ppo"] == opt):
            candidate = file.parent/"latest.pt"
            if candidate.is_file():
                saved = load_pt(candidate)
                if (saved["update"] == opt["updates"] and saved["config"] == original
                        and saved["seed"] == seed and saved["size"] == 3):
                    matches.append(candidate)
    if len(matches) > 1:
        raise ValueError(f"같은 사전학습 체크포인트가 여러 개입니다: {matches}")
    return matches[0] if matches else None


def weights(cfg, seed, policies, step):
    return dict(config=cfg, size=3, seed=seed, fit=None, update=step,
                policies=[p.state_dict() for p in policies])


def fixed_game(cfg, seed, episode, encoder, policies):
    # 4경기마다 빨강·파랑 학습 정책이 두 봇을 한 번씩 만납니다.
    side = episode % 2
    rival = OPPONENTS[(episode // 2) % 2]
    key = seed_for(seed, 551, episode)
    arena = Arena(cfg["game"], 3, key)
    foes = [bot(rival, seed_for(key, 31, i)) for i in range(3)]
    records, rewards = [], []
    area = arena.state.areas
    potential = (area[side] - area[1-side]) / arena.state.owner.size
    while not arena.state.terminated:
        acts, record = decide(arena, side, encoder, policies[side], False)
        acts.update({aid: controller.act(arena.observation(1-side, i))
                     for i,(aid,controller) in enumerate(zip(arena.teams[1-side], foes))})
        arena.step(acts, 0)
        s = arena.state
        gap = s.areas[side] - s.areas[1-side]
        next_potential = 0.0 if s.terminated else gap / s.owner.size
        reward = (float(np.sign(gap)) if s.terminated else 0.0)
        reward += cfg["finetune"]["gap_shaping"] * (cfg["ppo"]["gamma"]*next_potential - potential)
        records.append(record)
        rewards.append([reward]*3)
        potential = next_potential
    batch = trajectory(records, torch.tensor(rewards, dtype=torch.float32), cfg["ppo"])
    return side, batch, dict(win=int(gap>0), draw=int(gap==0), final_gap=gap)


def validate_bots(cfg, encoder, policies, step):
    opts = cfg["finetune"]
    pair = PolicyPair(cfg, step, encoder, policies)
    row = {"update":step}
    for opponent in OPPONENTS:
        wins = 0
        for trained in range(2):
            for i in range(opts["validation_games"]//2):
                key = seed_for(opts["validation_seed"], i, 71)
                for field in range(2):
                    game, _ = match(pair, trained, field, opponent, None, key, False)
                    wins += game["win"]
        row[opponent] = wins / (2*opts["validation_games"])
    return row


def train_stage(run, cfg, seed, fixed):
    path = run/("latest.pt" if fixed else "pretrained.pt")
    log = run/("training.csv" if fixed else "pretraining.csv")
    opt = cfg["ppo"]
    stage = cfg["finetune"] if fixed else opt
    end = stage["updates"]
    state = load_pt(path) if path.exists() else None
    if state is None and not fixed:
        src = reference(run.parent, cfg, seed)
        if src:
            state = load_pt(src)
            state = dict(state, config=cfg, source=str(src.relative_to(run.parent.parent)))
            save_pt(path, state)
    if state:
        write_csv(log, state["history"])
        if fixed:
            save_pt(run/"best.pt", state["best"])
            write_csv(run/"validation.csv", state["validation"])
        if state["update"] == end:
            return
    torch.manual_seed(seed_for(seed, 552) if fixed else seed)
    encoder = restore_encoder(None, cfg)
    nets = [make_policy(cfg) for _ in SIDES]
    opts = [torch.optim.Adam(p.parameters(), lr=stage["learning_rate"]) for p in nets]
    initial = state or (load_pt(run/"pretrained.pt") if fixed else None)
    if initial:
        for net, params in zip(nets, initial["policies"]):
            net.load_state_dict(params)
    if state:
        for optimizer, saved in zip(opts, state["optimizers"]):
            optimizer.load_state_dict(saved)
        torch.set_rng_state(state["rng"])
    start = state["update"] if state else 0
    rows = state["history"] if state else []
    validations = state.get("validation", []) if state else []
    best = state.get("best") if state else None
    best_key = tuple(state.get("best_key", (-1,-1))) if state else (-1,-1)
    for step in range(start+1, end+1):
        buffers, metrics = [[], []], [[], []]
        for i in range(stage["episodes_per_update"]):
            ep = (step-1)*stage["episodes_per_update"]+i
            if fixed:
                side, batch, info = fixed_game(cfg, seed, ep, encoder, nets)
                buffers[side].append(batch)
                metrics[side].append(info)
            else:
                batches, stats = rollout(cfg["game"], 3, seed, ep, encoder, nets, opt)
                for side in range(2):
                    buffers[side].append(batches[side])
                    metrics[side].append(stats[side])
        for side in range(2):
            group = metrics[side]
            rows.append(dict(team=SIDES[side], update=step, episodes=ep+1,
                             win=float(np.mean([r["win"] for r in group])),
                             final_gap=float(np.mean([r["final_gap"] for r in group])),
                             **update(nets[side], opts[side], buffers[side], opt)))
        interval = stage["validation_every"] if fixed else opt["checkpoint_every"]
        if step % interval == 0 or step == end:
            current = weights(cfg, seed, nets, step)
            if fixed:
                val = validate_bots(cfg, encoder, nets, step)
                validations.append(val)
                key = (min(val[o] for o in OPPONENTS), sum(val[o] for o in OPPONENTS))
                if key > best_key:
                    best_key, best = key, copy.deepcopy(current)
            save_pt(path, dict(**current, source="selfplay" if not fixed else "pretrained.pt",
                               history=rows, rng=torch.get_rng_state(), best=best, best_key=best_key,
                               validation=validations, optimizers=[o.state_dict() for o in opts]))
            if fixed:
                save_pt(run/"best.pt", best)
                write_csv(run/"validation.csv", validations)
            write_csv(log, rows)
            label = "고정 봇" if fixed else "자기대전"
            print(f"{run.name} · {label} {step}/{end}", flush=True)
