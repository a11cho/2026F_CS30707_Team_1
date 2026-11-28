"""같은 틱의 앞선 아군 행동에 조건화한 3대3 팀 정책과 PPO 자기대전."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "SL-RL"), str(ROOT)]

import torch
from torch import nn
from torch.distributions import Categorical
from arena import SIDES, inputs, reflect
from models import TeamPolicy
from training import trajectory, update
from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.environment import TerritoryEnvironment
from territory_rl.models import Action

SLOTS = 3
STATUS = 6
MESSAGE = STATUS + len(Action)


class SequentialPolicy(TeamPolicy):
    """기존 raw critic 입력을 유지하고 actor에만 순번·송신자별 행동과 상태를 추가합니다."""
    def __init__(self, game, ppo):
        super().__init__(game["height"] * game["width"], ppo)
        first = self.actor[0]
        self.actor[0] = nn.Linear(first.in_features + SLOTS + SLOTS * MESSAGE, first.out_features)

    def forward(self, board, own_context, peers, mask):
        own, public = own_context[:, :STATUS], own_context[:, STATUS:]
        count = own.new_full((len(own), 1), peers.shape[1])
        base = torch.cat((board, own, self.entity(peers).mean(1), count), dim=1)
        logits = self.actor(torch.cat((base, public), dim=1))
        dist = Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))
        return dist, self.critic(base).squeeze(-1)


def arena_for(game, seed):
    cfg = GameConfig(**game, initial_placement="frontier")
    env = TerritoryEnvironment(cfg, ExperimentConfig(decision_strategy="sequential",
                               observation_window_size=2 * max(cfg.width, cfg.height) - 1))
    env.reset(seed)
    teams = [[f"{side}_{i}" for i in range(SLOTS)] for side in SIDES]
    return SimpleNamespace(env=env, state=env.snapshot(), size=SLOTS, teams=teams)


@torch.no_grad()
def play_tick(arena, policies, alpha, *, greedy=False, hide_actions=False, external_actions=None):
    """제출 당시의 정책 입력을 보존하고 한 번의 실제 전이에 대한 보상을 반환합니다."""
    env = arena.env
    env.begin_decision()
    records = []
    for side, team in enumerate(arena.teams):
        if policies[side] is None:
            for i, aid in enumerate(team):
                env.agent_session(SIDES[side], i).submit_action(env.action_to_observation(external_actions[aid]))
            records.append(None)
            continue
        board, own, peers, masks = map(torch.from_numpy, inputs(arena, side))
        samples = []
        for i, aid in enumerate(team):
            session = env.agent_session(SIDES[side], i)
            codes = session.observe()["team_actions"]
            slots = own.new_zeros(SLOTS, MESSAGE)
            peer_ids = [a.agent_id for a in arena.state.agents if a.agent_id != aid]
            if not hide_actions:
                for j, code in enumerate(codes):
                    if code:
                        world = env.action_to_world(Action(int(code)))
                        slots[j, reflect(world, side).index] = 1
                        slots[j, len(Action):] = peers[i, peer_ids.index(team[j])]
            rank = own.new_zeros(SLOTS)
            rank[i] = 1
            context = torch.cat((own[i], rank, slots.flatten())).unsqueeze(0)
            features = (board[i:i+1], context, peers[i:i+1], masks[i:i+1])
            dist, value = policies[side](*features)
            act = dist.probs.argmax(1) if greedy else dist.sample()
            samples.append((*features, act, dist.log_prob(act), value))
            world = reflect(Action.from_index(act.item()), side)
            session.submit_action(env.action_to_observation(world))
        records.append(tuple(torch.cat(column) for column in zip(*samples)))
    result = env.finish_decision_round()
    arena.state = env.snapshot()
    rewards = [[arena.state.areas[s] / arena.state.owner.size + alpha * result.credits[aid]
                for aid in team] for s, team in enumerate(arena.teams)]
    return records, rewards, result


def episode(game, seed, policies, opt, stats=None):
    """양 팀의 에이전트별 궤적을 틱 단위 GAE로 변환합니다."""
    arena = arena_for(game, seed)
    records, rewards = [[], []], [[], []]
    while not arena.state.terminated:
        tick, earned, _ = play_tick(arena, policies, opt["alpha"])
        for side in range(2):
            records[side].append(tick[side])
            rewards[side].append(earned[side])
    if stats is not None:
        for side in range(2):
            gap = arena.state.areas[side] - arena.state.areas[1-side]
            stats[side].append(dict(per_agent_return=float(torch.tensor(rewards[side]).sum(0).mean()),
                                    final_gap=int(gap), ticks=arena.state.tick, win=int(gap > 0)))
    return [trajectory(records[s], torch.tensor(rewards[s], dtype=torch.float32), opt) for s in range(2)]


def ppo_update(game, episode_seeds, policies, optimizers, opt):
    """같은 정책 버전으로 경기들을 수집한 뒤 각 팀을 자기 경험으로 업데이트합니다."""
    buffers = [[], []]
    stats = [[], []]
    for seed in episode_seeds:
        batches = episode(game, seed, policies, opt, stats)
        for side in range(2):
            buffers[side].append(batches[side])
    if not buffers[0]:
        raise ValueError("한 업데이트에는 한 경기 이상이 필요합니다.")
    metrics = [update(policies[s], optimizers[s], buffers[s], opt) for s in range(2)]
    for row, games in zip(metrics, stats):
        row.update({key: sum(g[key] for g in games) / len(games) for key in games[0]})
    return metrics
