"""동일한 시드·진영에서 순차 정책과 고정 봇 또는 기존 독립 정책을 평가합니다."""
from collections import defaultdict

from model import arena_for, play_tick
from arena import SIDES, bot, seed_for
from evaluation import Player, cooperation
from checkpoint_selection import OPPONENTS
from storage import allocate, write_csv, write_json


def match(cfg, policy, field, rival, rival_team, seed, measure):
    arena = arena_for(cfg["game"], seed)
    foe = 1 - field
    nets = [None, None]
    nets[field] = policy
    if isinstance(rival, str):
        bots = [bot(rival, seed_for(seed, i, 66)) for i in range(arena.size)]
    events = [{}]
    total_return = 0.
    while not arena.state.terminated:
        if isinstance(rival, str):
            actions = {aid: arena.env.action_to_world(bots[i].act(arena.env.observe(SIDES[foe], i)))
                       for i, aid in enumerate(arena.teams[foe])}
        else:
            actions = rival.actions(arena, foe, rival_team)
        _, rewards, step = play_tick(arena, nets, cfg["ppo"]["alpha"], greedy=True, external_actions=actions)
        total_return += sum(rewards[field]) / arena.size
        if measure:
            events.append({"events": [vars(e) for e in step.events]})
    gap = int(arena.state.areas[field] - arena.state.areas[foe])
    row = dict(win=int(gap > 0), draw=int(gap == 0), loss=int(gap < 0), final_gap=gap,
               ticks=arena.state.tick, per_agent_return=total_return)
    if measure:
        for side, prefix in [(field, "model"), (foe, "opponent")]:
            row.update({f"{prefix}_{k}": v for k,v in cooperation(events, side).items()})
    return row


def games(cfg, policies, rivals, n, validation=False):
    if n < 2 or n % 2:
        raise ValueError("경기 수는 양 진영에 나눌 양의 짝수여야 합니다.")
    for name, rival, other in rivals:
        for trained, policy in enumerate(policies):
            for i in range(n):
                field = i // (n // 2)
                seed = seed_for(cfg["evaluation"]["seed"], i % (n // 2), 72 if validation else 71)
                row = match(cfg, policy, field, rival, other, seed, not validation)
                yield dict(opponent=name, opponent_team=SIDES[other] if other is not None else "",
                           trained_team=SIDES[trained], field=SIDES[field], seed=seed, **row)


def validate(cfg, policies):
    rows = list(games(cfg, policies, [(s,s,None) for s in OPPONENTS],
                      cfg["evaluation"]["validation_games"], True))
    return sum(r["win"] + .5*r["draw"] for r in rows) / len(rows)


def evaluate(root, exp, cfg, entries, checkpoint):
    prepared = []
    for seed, policies, step in entries:
        rivals = [(s,s,None) for s in cfg["evaluation"]["opponents"]]
        for size, name in zip((1,3), cfg["baselines"][str(seed)], strict=True):
            player = Player(root/"SL-RL"/"experiment"/"exp2"/name, checkpoint)
            if (player.seed != seed or player.size != size or player.cfg["representation"] != "raw"
                    or player.cfg["game"] != cfg["game"] or player.cfg["ppo"]["alpha"] != cfg["ppo"]["alpha"]):
                raise ValueError(f"비교 모델 설정 불일치: {name}")
            rivals.extend((f"exp2/{name}",player,s) for s in range(2))
        prepared.append((seed, policies, step, rivals))
    out = allocate(exp/"result", "evaluation")
    write_json(out/"settings.json", dict(config=cfg, checkpoint=checkpoint, greedy=True))
    rows, summary = [], []
    for seed, policies, step, rivals in prepared:
        groups = defaultdict(list)
        for row in games(cfg, policies, rivals, cfg["evaluation"]["games"]):
            row.update(train_seed=seed, update=step)
            rows.append(row)
            groups[(row["opponent"],row["opponent_team"],row["trained_team"])].append(row)
        for (opponent, other, team), group in groups.items():
            wins, draws, losses = (sum(r[k] for r in group) for k in ("win","draw","loss"))
            item = dict(train_seed=seed, update=step, trained_team=team, opponent=opponent, opponent_team=other,
                        games=len(group), wins=wins, draws=draws, losses=losses, win_rate=wins/len(group))
            for k in ("final_gap","ticks","per_agent_return"):
                item[k] = sum(r[k] for r in group)/len(group)
            for prefix in ("model","opponent"):
                for k in ("shared_targets","attacked_targets","joint_captures","captures","deaths","failed_moves"):
                    key = f"{prefix}_{k}"
                    item[key] = sum(r[key] for r in group)
                for rate, a, b in [("shared_target_rate","shared_targets","attacked_targets"),
                                   ("joint_capture_rate","joint_captures","captures")]:
                    denom = item[f"{prefix}_{b}"]
                    item[f"{prefix}_{rate}"] = item[f"{prefix}_{a}"]/denom if denom else None
            summary.append(item)
        write_csv(out/"games.csv", rows)
        write_csv(out/"summary.csv", summary)
        print(f"평가 저장 {out} · 시드 {seed}", flush=True)
    return out
