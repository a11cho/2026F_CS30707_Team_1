"""학습된 양 팀을 고정 상대 전체와 평가하고 두 실행의 정책끼리 맞붙입니다."""
import numpy as np

from arena import Arena, BOTS, SIDES, bot, seed_for
from storage import allocate, load_pt, write_csv, write_json
from training import decide, make_policy, restore_encoder


class PolicyPair:
    def __init__(self, cfg, update, encoder, teams):
        self.cfg, self.update = cfg, update
        self.encoder, self.teams = encoder, teams

    def actions(self, arena, field_side, trained_side):
        # 양 진영 입력과 행동을 공통 방향으로 변환하므로 반대 진영도 같은 경로입니다.
        return decide(arena, field_side, self.encoder, self.teams[trained_side], True)[0]


class Player(PolicyPair):
    def __init__(self, run, checkpoint):
        try:
            state = load_pt(run/checkpoint)
        except FileNotFoundError:
            # 정리 전 사용하던 번호 명령도 보존된 시점이면 계속 동작합니다.
            if not checkpoint.startswith("update_"):
                raise
            for alias in ("latest.pt", "best.pt"):
                if (run/alias).is_file():
                    state = load_pt(run/alias)
                    if checkpoint == f"update_{state['update']:04d}.pt":
                        break
            else:
                raise
        self.cfg, self.size = state["config"], state["size"]
        self.label = f"{run.name}/{checkpoint}"
        self.seed, self.update = state["seed"], state["update"]
        self.encoder = restore_encoder(state["fit"], self.cfg)
        self.teams = [make_policy(self.cfg) for _ in SIDES]
        for policy, weights in zip(self.teams, state["policies"]):
            policy.load_state_dict(weights)
            policy.eval()



def frame(arena):
    s = arena.state
    return {"tick":s.tick, "owner":s.owner.tolist(), "agents":[
        {"id":a.agent_id, "position":a.position, "alive":a.alive} for a in s.agents]}


def match(player, team, field, opponent, other_team, seed, trace, *, size=3, row=0):
    arena = Arena(player.cfg["game"], size, seed, row)
    foe = 1-field
    if isinstance(opponent, str):
        bots = [bot(opponent, seed_for(seed,i,66)) for i in range(size)]
    returns = np.zeros(size)
    share = 0.
    frames = [frame(arena)] if trace else []
    while not arena.state.terminated:
        acts = player.actions(arena, field, team)
        if isinstance(opponent, str):
            acts.update({aid:bots[i].act(arena.observation(foe, i)) for i,aid in enumerate(arena.teams[foe])})
        else:
            acts.update(opponent.actions(arena, foe, other_team))
        rewards = arena.step(acts, player.cfg["ppo"]["alpha"])
        share += arena.state.areas[field] / arena.state.owner.size
        returns += [rewards[aid] for aid in arena.teams[field]]
        if trace:
            frames.append(dict(**frame(arena), actions={k:int(v) for k,v in acts.items()}))
    s = arena.state
    gap = s.areas[field]-s.areas[foe]
    return dict(ticks=s.tick, win=int(gap>0), draw=int(gap==0), loss=int(gap<0),
                final_gap=gap, per_agent_return=float(returns.mean()), mean_area_share=share/s.tick), frames


def evaluate(path, cfg, run, checkpoint, versus, games, opponents, replay):
    if games < 2 or games % 2:
        raise ValueError("경기 수는 진영 교대를 위해 양의 짝수여야 합니다.")
    if not opponents or any(x not in BOTS for x in opponents) or len(set(opponents)) != len(opponents):
        raise ValueError("고정 상대 목록과 중복을 확인하세요.")
    player = Player(path/run, checkpoint)
    rival = Player(path/versus, checkpoint) if versus else None
    if rival and (player.cfg["game"] != rival.cfg["game"] or player.cfg["ppo"]["alpha"] != rival.cfg["ppo"]["alpha"]):
        raise ValueError("맞대결은 같은 경기 설정과 보상 계수를 사용하세요.")
    if rival and rival.size == player.size:
        raise ValueError("복제 비교에는 1인 학습 run과 3인 학습 run이 필요합니다.")
    out = allocate(path/"comparison" if rival else path/run/"evaluation", "eval")
    settings = dict(model=player.label, model_training_size=player.size, model_seed=player.seed,
                    update=player.update, evaluation_seed=cfg["evaluation"]["seed"],
                    games_per_pair=games, opponents=opponents if not rival else [rival.label],
                    other_seed=rival.seed if rival else None, evaluation_size=3, greedy=True)
    write_json(out/"settings.json", settings)
    pairs = [(rival,i) for i in range(2)] if rival else [(s,None) for s in opponents]
    rows, summaries = [], []
    for trained in range(2):
        for opponent, other in pairs:
            label = rival.label if rival else opponent
            group = []
            for game in range(games):
                field = game//(games//2)
                seed = seed_for(cfg["evaluation"]["seed"], game%(games//2), 71)
                trace = replay and game%(games//2)==0
                result, frames = match(player, trained, field, opponent, other, seed, trace)
                ids = dict(model=player.label, size=player.size, train_seed=player.seed,
                           trained_team=SIDES[trained], opponent=label, opponent_team=SIDES[other] if rival else "")
                row = dict(**ids, field=SIDES[field], seed=seed, **result)
                group.append(row)
                if trace:
                    name = f"{trained}_{other if rival else opponent}_{field}.json"
                    write_json(out/name, dict(game=row, frames=frames))
            rows.extend(group)
            summaries.append(dict(**ids, games=games, wins=sum(x["win"] for x in group),
                                  draws=sum(x["draw"] for x in group), losses=sum(x["loss"] for x in group),
                                  final_gap=float(np.mean([x["final_gap"] for x in group])),
                                  per_agent_return=float(np.mean([x["per_agent_return"] for x in group])),
                                  mean_area_share=float(np.mean([x["mean_area_share"] for x in group])),
                                  win_rate=sum(x["win"] for x in group)/games))
            write_csv(out/"games.csv", rows)
            write_csv(out/"summary.csv", summaries)
    print(out, flush=True)
    return out
