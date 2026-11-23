"""한 경기에서 양 팀의 경험을 수집합니다."""

import numpy as np
import torch

from .metrics import GameMetrics
from .networks import sample_actions
from .opponents import make_opponents
from .replay import ReplayWriter


def compute_gae(rewards, values, next_value, gamma, gae_lambda):
    """한 경기 안에서 팀원별 GAE를 계산합니다. 사망으로 궤적을 끊지 않습니다."""
    advantage = np.empty_like(values)
    following = np.zeros_like(next_value)
    for tick in range(len(rewards)-1, -1, -1):
        delta = rewards[tick] + gamma*next_value - values[tick]
        following = delta + gamma*gae_lambda*following
        advantage[tick] = following
        next_value = values[tick]
    return advantage, advantage + values


class RolloutBuffer:
    def __init__(self, gamma, gae_lambda):
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.games = []

    def add_episode(self, steps, next_value):
        batch = {key:np.stack([row[key] for row in steps]) for key in steps[0]}
        advantage, target = compute_gae(batch.pop('rewards'), batch.pop('values'), next_value,
                                        self.gamma, self.gae_lambda)
        batch.update(advantage=advantage, target=target)
        self.games.append({key:array.reshape((-1, *array.shape[2:])) for key,array in batch.items()})

    def tensors(self, device):
        return {key:torch.as_tensor(np.concatenate([game[key] for game in self.games]),device=device)
                for key in self.games[0]}


@torch.no_grad()
def play_episode(env, actors, seed, device, *, critics=None, buffers=None,
                 opponent=None, deterministic=False, replay_path=None, metadata=None,
                 mirrored_teams=()):
    """actors는 실제 진영→정책입니다. 학습 시 양 진영, 평가 시 한 진영을 전달합니다."""
    observations = env.reset(seed)
    controllers = {}
    for team in env.team_names:
        if team not in actors:
            controllers.update(make_opponents(opponent, env.teams[team], seed))
    metrics = {team:GameMetrics(env.teams[team], env.team_names.index(team)) for team in actors}
    trajectories = {team:[] for team in actors}
    writer = ReplayWriter(replay_path,env,seed,metadata or {}) if replay_path else None
    while True:
        submitted, tick_data = {}, {}
        for team, actor in actors.items():
            agents = env.teams[team]
            mirror=team in mirrored_teams
            features, masks, alive = env.encode(observations, agents,mirror=mirror)
            inputs = torch.as_tensor(features,device=device)
            actions, log_prob = sample_actions(actor,inputs,torch.as_tensor(masks,device=device),deterministic)
            selected = actions.cpu().numpy()
            submitted.update({agent:env.reflection.action(index) if mirror else int(index)
                              for agent,index in zip(agents,selected)})
            if buffers is not None:
                tick_data[team] = dict(features=features,masks=masks,alive=alive,
                                       actions=selected,old_log_prob=log_prob.cpu().numpy(),
                                       values=critics[team](inputs).cpu().numpy())
        submitted.update({agent:int(controller.act(observations[agent]).index)
                          for agent,controller in controllers.items()})
        observations,rewards,terminal,cutoff,result,components = env.step(submitted)
        for team in actors:
            metrics[team].record(result,rewards,components)
            if buffers is not None:
                tick_data[team]['rewards'] = np.array([rewards[a] for a in env.teams[team]],dtype=np.float32)
                trajectories[team].append(tick_data[team])
        if writer:
            writer.record(env.replay_actions(submitted),result,rewards,components)
        if terminal or cutoff:
            break
    if buffers is not None:
        for team in actors:
            next_value = np.zeros(len(env.teams[team]),dtype=np.float32)
            if cutoff and not terminal:
                features,_,_ = env.encode(observations,env.teams[team],mirror=team in mirrored_teams)
                next_value = critics[team](torch.as_tensor(features,device=device)).cpu().numpy()
            buffers[team].add_episode(trajectories[team],next_value)
    if writer:
        writer.finish()
    return {team:tracker.finish(result) for team,tracker in metrics.items()}
