"""양 팀 모델·옵티마이저와 재개에 필요한 난수 상태를 한 번에 저장합니다."""

from pathlib import Path

import torch

from .networks import make_networks
from .storage import atomic_write


VERSION = 2


def setup_runtime(cfg):
    torch.set_num_threads(cfg['threads'])
    device = torch.device(cfg['device'])
    if device.type not in ('cpu','cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('사용 가능한 cpu 또는 cuda 장치를 지정하세요.')
    return device


def make_learners(env,cfg,device):
    learners = {}
    for team in env.team_names:
        actor,critic = make_networks(env,cfg,device)
        optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),
                                     lr=cfg.ppo['learning_rate'])
        learners[team] = dict(actor=actor,critic=critic,optimizer=optimizer)
    return learners


def save_checkpoint(path,cfg,env,learners,device,update,episodes,metrics):
    state = dict(version=VERSION,config=cfg.to_dict(),signature=env.signature,
                 update=update,episodes=episodes,metrics=metrics,
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                 learners={team:{name:obj.state_dict() for name,obj in parts.items()}
                           for team,parts in learners.items()})
    atomic_write(path,lambda tmp:torch.save(state,tmp))


def load_checkpoint(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path/'checkpoints'/'latest.pt'
    state = torch.load(path,map_location='cpu',weights_only=True)
    if state.get('version') != VERSION:
        raise ValueError('양 팀 자기대전용 v2 체크포인트가 필요합니다. 이전 실험은 archive/에 보관됩니다.')
    return path,state


def restore(state,env,learners,device,*,training=False):
    if state['signature'] != env.signature or set(state['learners']) != set(learners):
        raise ValueError('관측·행동 또는 팀 규격이 저장 당시와 다릅니다. 새 실행을 시작하세요.')
    for team,parts in learners.items():
        for name,obj in parts.items():
            if name != 'optimizer' or training:
                obj.load_state_dict(state['learners'][team][name])
    if training:
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'],device)
