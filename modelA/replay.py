"""원본 환경 재생 계약을 유지하며 학습 보상을 별도 필드에 기록합니다."""

import json

from .storage import atomic_write


class ReplayWriter:
    def __init__(self, path, env, seed, metadata):
        self.path = path
        self.lines = [dict(type='reset', seed=seed, game=env.game_config,
                           experiment=env.experiment_config,
                           learning_reward=env.reward_config, modelA=metadata)]

    def record(self, actions, result, rewards, components):
        self.lines.append(dict(type='step', actions=actions, **result.to_dict(),
                               learning_rewards=rewards, learning_components=components))

    def finish(self):
        def write(path):
            with path.open('w', encoding='utf-8') as stream:
                for row in self.lines:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        atomic_write(self.path, write)
