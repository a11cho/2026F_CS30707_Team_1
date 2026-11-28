"""3대3 봇 시범을 수집하고 기존 양 진영의 actor를 행동 모방으로 미세조정합니다."""
import copy
import numpy as np
import torch

from arena import Arena, SIDES, bot, inputs, reflect, seed_for
from storage import load_pt, save_pt, write_csv
from training import make_policy

TEACHERS = ("expand", "aggressive")


def demonstrations(game, seed, split, games):
    """게임 단위 분할. 교사가 실제 제출한 행동을 학생의 진영 정규화 좌표계로 기록합니다."""
    columns = [[] for _ in range(6)]
    for episode in range(games):
        key = seed_for(seed, split, episode)
        arena = Arena(game, 3, key)
        controllers = [[bot(TEACHERS[(side + episode) % 2], seed_for(key, side, i))
                        for i in range(3)] for side in range(2)]
        while not arena.state.terminated:
            actions = {}
            for side, team in enumerate(arena.teams):
                chosen = [c.act(arena.observation(side, i)) for i, c in enumerate(controllers[side])]
                actions.update(zip(team, chosen))
                board, own, peers, mask = inputs(arena, side)
                labels = np.array([reflect(a, side).index for a in chosen], dtype=np.int64)
                if not mask[np.arange(len(team)), labels].all():
                    raise ValueError("교사 행동과 학생 행동 마스크가 일치하지 않습니다.")
                free = mask.sum(1) > 1
                if free.any():
                    teachers = np.full(len(team), (side + episode) % 2, dtype=np.int64)
                    for parts, array in zip(columns, (board, own, peers, mask, labels, teachers)):
                        parts.append(array[free])
            arena.step(actions, 0)
    return tuple(torch.from_numpy(np.concatenate(parts)) for parts in columns)


def epoch_pass(policy, data, batch_size, optimizer=None, generator=None):
    """두 교사의 손실을 동일 비중으로 반영하고 교사별 정확도를 기록합니다."""
    board, own, peers, mask, actions, teachers = data
    counts = torch.bincount(teachers, minlength=len(TEACHERS))
    if (counts == 0).any():
        raise ValueError("두 교사 모두 유효한 시범이 필요합니다.")
    weights = len(actions) / (len(TEACHERS) * counts.float())
    totals = torch.zeros(2, dtype=torch.float64)
    correct = torch.zeros(2, dtype=torch.int64)
    order = (torch.randperm(len(actions), generator=generator) if optimizer is not None
             else torch.arange(len(actions)))
    policy.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for idx in order.split(batch_size):
            dist, _ = policy(board[idx], own[idx], peers[idx], mask[idx])
            loss = -dist.log_prob(actions[idx])
            types = teachers[idx]
            if optimizer is not None:
                optimizer.zero_grad()
                (loss * weights[types]).mean().backward()
                optimizer.step()
            totals.scatter_add_(0, types, loss.detach().double())
            hits = (dist.probs.argmax(1) == actions[idx]).long()
            correct.scatter_add_(0, types, hits)
    accuracy = correct / counts
    return dict(loss=(totals / counts).mean().item(), accuracy=accuracy.mean().item(),
                **{f"{name}_accuracy":accuracy[i].item() for i, name in enumerate(TEACHERS)})


def model_state(base, policies, epoch):
    return dict(config=base["config"], size=3, seed=base["seed"], fit=None, update=epoch,
                policies=[p.state_dict() for p in policies])


def fit(run, settings):
    base = load_pt(run/"before.pt")
    opt = settings["imitation"]
    networks = [make_policy(base["config"]) for _ in SIDES]
    for net, weights in zip(networks, base["policies"]):
        net.load_state_dict(weights)
        net.critic.requires_grad_(False)
    optimizers = [torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                   lr=opt["learning_rate"]) for net in networks]
    history, start, best_loss, best = [], 0, float("inf"), None
    if (run/"latest.pt").exists():
        saved = load_pt(run/"latest.pt")
        for net, optimizer, weights, state in zip(networks, optimizers, saved["policies"], saved["optimizers"]):
            net.load_state_dict(weights)
            optimizer.load_state_dict(state)
        history, start = saved["history"], saved["update"]
        best_loss, best = saved["best_loss"], saved["best"]
        save_pt(run/"best.pt", best)
        write_csv(run/"training.csv", history)
        if start >= opt["epochs"]:
            return
    game, seed = base["config"]["game"], base["seed"]
    train = demonstrations(game, seed, 801, opt["train_games"])
    valid = demonstrations(game, seed, 802, opt["validation_games"])
    if not history:
        for side, net in enumerate(networks):
            check = epoch_pass(net, valid, opt["batch_size"])
            history.append(dict(epoch=0, team=SIDES[side], train_loss="", train_accuracy="",
                                **{f"val_{k}":v for k,v in check.items()}))
    for epoch in range(start + 1, opt["epochs"] + 1):
        scores = []
        for side, (net, optimizer) in enumerate(zip(networks, optimizers)):
            rng = torch.Generator().manual_seed(seed_for(seed, 803, side, epoch))
            tr = epoch_pass(net, train, opt["batch_size"], optimizer, rng)
            val = epoch_pass(net, valid, opt["batch_size"])
            scores.append(val["loss"])
            history.append(dict(epoch=epoch, team=SIDES[side], train_loss=tr["loss"],
                                train_accuracy=tr["accuracy"], **{f"val_{k}":v for k,v in val.items()}))
        current = model_state(base, networks, epoch)
        score = sum(scores) / len(scores)
        improved = score < best_loss
        if improved:
            best_loss, best = score, copy.deepcopy(current)
        save_pt(run/"latest.pt", dict(**current, history=history, best=best, best_loss=best_loss,
                                     optimizers=[optimizer.state_dict() for optimizer in optimizers]))
        if improved:
            save_pt(run/"best.pt", best)
        write_csv(run/"training.csv", history)
        print(f"{run.name} · epoch {epoch}/{opt['epochs']} · 검증 손실 {score:.4f}", flush=True)
