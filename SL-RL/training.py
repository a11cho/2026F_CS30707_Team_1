"""팀별 rollout, 에이전트별 GAE 및 PPO 업데이트를 담당합니다."""
import json
import time
import numpy as np
import torch
from torch import nn

from arena import Arena, SIDES, inputs, opening_row, reflect, seed_for
from models import MapEncoder, TeamPolicy
from storage import allocate, load_pt, run_lock, save_pt, write_csv, write_json
from territory_rl.models import Action


def restore_encoder(fit, cfg=None):
    if cfg is not None and cfg.get("representation") == "raw":
        return nn.Identity()
    cfg = fit["config"]
    opt, game = cfg["ssl"], cfg["game"]
    encoder = MapEncoder(game["width"]*game["height"], opt["hidden"], opt["latent"])
    encoder.load_state_dict(fit["weights"])
    return encoder.eval().requires_grad_(False)


def make_policy(cfg):
    width = cfg["ssl"]["latent"]
    if cfg.get("representation") == "raw":
        width = cfg["game"]["width"] * cfg["game"]["height"]
    return TeamPolicy(width, cfg["ppo"])


@torch.no_grad()
def decide(arena, side, encoder, policy, greedy):
    board, own, entities, mask = map(torch.from_numpy, inputs(arena, side))
    z = encoder(board)
    distribution, values = policy(z, own, entities, mask)
    action = distribution.probs.argmax(1) if greedy else distribution.sample()
    world = {aid: reflect(Action.from_index(int(a)), side)
             for aid, a in zip(arena.teams[side], action)}
    return world, (z, own, entities, mask, action, distribution.log_prob(action), values)


def trajectory(records, rewards, opt):
    # 형태는 (틱, 팀원, ...). 역방향 합산 동안 에이전트 축을 유지합니다.
    columns = [torch.stack(items) for items in zip(*records)]
    values = columns[-1]
    next_value = torch.zeros_like(values[0])
    running = torch.zeros_like(next_value)
    advantages = torch.empty_like(values)
    for t in reversed(range(len(records))):
        delta = rewards[t] + opt["gamma"] * next_value - values[t]
        running = delta + opt["gamma"] * opt["gae_lambda"] * running
        advantages[t] = running
        next_value = values[t]
    return [x.flatten(0, 1) for x in [*columns[:-1], advantages, advantages+values]]


def rollout(game, size, seed, episode, encoder, policies, opt):
    base = seed_for(seed, 91)
    arena = Arena(game, size, seed_for(base, episode), opening_row(base, episode, game["height"]))
    records, rewards = [[], []], [[], []]
    area_sums = np.zeros(2)
    while not arena.state.terminated:
        actions = {}
        for side in range(2):
            chosen, data = decide(arena, side, encoder, policies[side], False)
            actions.update(chosen)
            records[side].append(data)
        earned = arena.step(actions, opt["alpha"])
        area_sums += np.asarray(arena.state.areas) / arena.state.owner.size
        for side, team in enumerate(arena.teams):
            rewards[side].append([earned[a] for a in team])
    batches, stats = [], []
    for side in range(2):
        r = torch.tensor(rewards[side], dtype=torch.float32)
        batches.append(trajectory(records[side], r, opt))
        diff = arena.state.areas[side] - arena.state.areas[1-side]
        stats.append(dict(ticks=arena.state.tick, decisions=int((batches[-1][3].sum(1)>1).sum()),
                          per_agent_return=r.sum(0).mean().item(), mean_area_share=area_sums[side]/arena.state.tick,
                          final_gap=diff, win=int(diff>0), draw=int(diff==0)))
    return batches, stats


def update(policy, optim, rollouts, opt):
    z, own, entities, mask, acts, old, adv, targets = [torch.cat(x) for x in zip(*rollouts)]
    active = mask.sum(1)>1
    if active.any():
        valid = adv[active]
        adv = (adv-valid.mean())/(valid.std(unbiased=False)+1e-8)
    totals = np.zeros(4)
    samples = np.zeros(4)
    for _ in range(opt["epochs"]):
        for idx in torch.randperm(len(z)).split(opt["batch_size"]):
            dist, value = policy(z[idx], own[idx], entities[idx], mask[idx])
            usable = active[idx]
            log_ratio = dist.log_prob(acts[idx]) - old[idx]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio*adv[idx], ratio.clamp(1-opt["clip"],1+opt["clip"])*adv[idx])
            actor = -surrogate[usable].mean() if usable.any() else value.new_zeros(())
            entropy = dist.entropy()[usable].mean() if usable.any() else value.new_zeros(())
            critic = (value-targets[idx]).square().mean()
            loss = actor + opt["value_weight"]*critic - opt["entropy"]*entropy
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), opt["max_grad_norm"])
            optim.step()
            with torch.no_grad():
                kl = ((ratio-1)-log_ratio)[usable].sum().item()
                n = int(usable.sum())
                totals += [actor.item()*n, critic.item()*len(idx), entropy.item()*n, kl]
                samples += [n, len(idx), n, n]
    return dict(zip(("actor_loss","critic_loss","entropy","approx_kl"), map(float, totals/np.maximum(samples, 1))))


def encoder_file(path, cfg, size, fit_name):
    parent = path/"pretrain"/f"{size}v{size}"
    paths = [parent/fit_name/"encoder.pt"] if fit_name else sorted(parent.glob("fit*/encoder.pt"),
                key=lambda p: int(p.parent.name[3:]), reverse=True)
    for p in paths:
        fit = load_pt(p)
        if fit["size"] == size and all(fit["config"][k]==cfg[k] for k in ("game","ssl")):
            return p, fit
    raise ValueError("현재 설정과 일치하는 사전학습 결과가 없습니다. 먼저 pretrain을 실행하세요.")


def train(path, cfg, size, seed, fit_name=None, resume=None, updates=None, validation_games=None):
    if resume:
        run = path/resume
        fit = None
    else:
        if validation_games is not None and "online_validation" in cfg:
            raise ValueError("이 실험은 학습 중 검증이 설정되어 있습니다.")
        if cfg.get("representation") == "raw" and fit_name:
            raise ValueError("원본 지도 입력에는 인코더를 지정할 수 없습니다.")
        source, fit = (None, None) if cfg.get("representation") == "raw" else encoder_file(path, cfg, size, fit_name)
        run = allocate(path, "run")
        write_json(run/"settings.json", dict(size=size, seed=seed, config=cfg, encoder=source.relative_to(path).as_posix() if source else None,
                                              requested_updates=updates if updates is not None else cfg["ppo"]["updates"]))
    with run_lock(run):
        if resume:
            settings = json.loads((run/"settings.json").read_text(encoding="utf-8"))
            cfg = settings["config"]
            if validation_games is not None and "online_validation" in cfg:
                raise ValueError("이 실험은 학습 중 검증이 설정되어 있습니다.")
            if "online_validation" in cfg and not (run/"latest.pt").exists():
                # 첫 검증 전에 종료한 실행도 원래 시드와 인코더로 같은 폴더에서 다시 시작합니다.
                size, seed = settings["size"], settings["seed"]
                fit = load_pt(path/settings["encoder"]) if settings["encoder"] else None
                resume = None
        result = train_in_run(run, cfg, size, seed, fit, resume, updates)
        if validation_games is not None:
            from checkpoint_selection import keep_last_and_best
            target = updates if updates is not None else load_pt(run/"latest.pt")["config"]["ppo"]["updates"]
            keep_last_and_best(run, target, validation_games)
        return result


def sync_validation(run, state):
    """latest에 기록된 점수 이력을 기준으로 best와 CSV를 복구할 수 있게 합니다."""
    rows = state["validation_history"]
    winner = max(rows, key=lambda row: (row["score"], row["update"]))
    if winner["update"] == state["update"]:
        save_pt(run/"best.pt", state)
    elif not (run/"best.pt").is_file():
        raise FileNotFoundError("이전 최고 체크포인트 best.pt가 없습니다.")
    write_csv(run/"validation.csv", rows)


def train_in_run(run, cfg, size, seed, fit, resume, updates):
    if resume:
        state = load_pt(run/"latest.pt")
        cfg, size, seed, fit = state["config"], state["size"], state["seed"], state["fit"]
        start, history = state["update"], state["history"]
    else:
        start, history = 0, []
    online = cfg.get("online_validation")
    validations = list(state["validation_history"]) if resume and online else []
    opt = cfg["ppo"]
    end = updates if updates is not None else opt["updates"]
    if end < start:
        raise ValueError("목표 업데이트가 체크포인트보다 작습니다.")
    torch.manual_seed(seed)
    encoder = restore_encoder(fit, cfg)
    policies = [make_policy(cfg) for _ in SIDES]
    optimizers = [torch.optim.Adam(net.parameters(), lr=opt["learning_rate"]) for net in policies]
    if resume:
        for side in range(2):
            policies[side].load_state_dict(state["policies"][side])
            optimizers[side].load_state_dict(state["optimizers"][side])
        torch.set_rng_state(state["rng"])
        write_csv(run/"training.csv", history)
        if online:
            sync_validation(run, state)
    print(f"{run} | {size}vs{size} | seed={seed}", flush=True)
    for step in range(start+1, end+1):
        begin = time.perf_counter()
        buffers, stats = [[], []], [[], []]
        for i in range(opt["episodes_per_update"]):
            ep = (step-1)*opt["episodes_per_update"]+i
            batch, info = rollout(cfg["game"], size, seed, ep, encoder, policies, opt)
            for side in range(2):
                buffers[side].append(batch[side])
                stats[side].append(info[side])
        rows = []
        for side in range(2):
            info = stats[side]
            metrics = {k:float(np.mean([x[k] for x in info])) for k in info[0]}
            rows.append(dict(update=step, episodes=ep+1, team=SIDES[side],
                             environment_ticks=sum(x["ticks"] for x in info),
                             action_decisions=sum(x["decisions"] for x in info),
                             **metrics, **update(policies[side], optimizers[side], buffers[side], opt)))
        for row in rows:
            row["update_seconds"] = time.perf_counter()-begin
        history.extend(rows)
        if step % opt["checkpoint_every"] == 0 or step == end:
            state = dict(config=cfg, size=size, seed=seed, fit=fit, update=step, history=history,
                         rng=torch.get_rng_state(), policies=[p.state_dict() for p in policies],
                         optimizers=[o.state_dict() for o in optimizers])
            if online:
                from checkpoint_selection import validation_score
                from evaluation import PolicyPair
                pair = PolicyPair(cfg, step, encoder, policies)
                score = validation_score(pair, online["games"], size=size)
                validations.append(dict(size=size, **score))
                state["validation_history"] = validations
                save_pt(run/"latest.pt", state)
                sync_validation(run, state)
                print(f"{size}vs{size} 검증 {score['wins']}승 {score['draws']}무 {score['losses']}패 · {score['score']:.3f}", flush=True)
            else:
                save_pt(run/f"update_{step:04d}.pt", state)
                save_pt(run/"latest.pt", state)
            print(f"체크포인트 {step}/{end} · {ep+1}경기", flush=True)
        write_csv(run/"training.csv", history)
    return run
