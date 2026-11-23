"""실제 환경과 학습 진영의 좌우 방향을 맞추는 관측·행동 변환."""

import numpy as np
from territory_rl import Action


class HorizontalReflection:
    def __init__(self,action_count):
        swapped={Action.LEFT:Action.RIGHT,Action.RIGHT:Action.LEFT,
                 Action.ATTACK_LEFT:Action.ATTACK_RIGHT,Action.ATTACK_RIGHT:Action.ATTACK_LEFT}
        self.permutation=np.array([swapped.get(Action.from_index(i),Action.from_index(i)).index
                                   for i in range(action_count)],dtype=np.int64)
        if not np.array_equal(self.permutation[self.permutation],np.arange(action_count)):
            raise ValueError('행동의 좌우반전 대응이 대칭적이지 않습니다.')

    def action(self,index):
        return int(self.permutation[index])

    def observation(self,obs):
        """입력을 수정하지 않는 복사본을 반환하며 사망 좌표 (-1,-1)을 보존합니다."""
        result={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in obs.items()}
        width=obs['owner'].shape[1]
        result['owner']=obs['owner'][:,::-1].copy()
        positions=result['agent_positions']
        alive=np.asarray(obs['agent_alive'],dtype=bool)
        positions[alive,1]=width-1-positions[alive,1]
        result['action_mask']=obs['action_mask'][self.permutation].copy()
        if 'team_action_features' in obs:
            result['team_action_features']=obs['team_action_features'][:,self.permutation].copy()
        if 'team_action_codes' in obs:
            for slot,code in enumerate(obs['team_action_codes']):
                if code:
                    result['team_action_codes'][slot]=int(Action.from_index(
                        self.action(Action.parse(code).index)))
        return result
