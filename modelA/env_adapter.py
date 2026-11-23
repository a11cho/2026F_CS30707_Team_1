"""Territory의 관측과 전이 형식을 학습 코드에서 사용할 형태로 변환합니다."""

from dataclasses import asdict

import numpy as np
from gymnasium.spaces import Discrete
from territory_rl import Action, ExperimentConfig, GameConfig
from territory_rl.parallel_env import TerritoryParallelEnv

from .rewards import calculate_learning_rewards
from .symmetry import HorizontalReflection


class ModelAEnv:
    def __init__(self, cfg):
        # 학습 보상은 연결부에서 계산합니다. 환경 보상은 모두 꺼 이중 가산을 방지합니다.
        experiment = ExperimentConfig(decision_strategy="independent", reward_mode="team",
                                      reward_signal_mask=(0, 0, 0))
        self.env = TerritoryParallelEnv(GameConfig(**cfg.game), experiment)
        self.reward_config = dict(cfg.reward)
        self.game_config = asdict(self.env.core.config)
        self.experiment_config = asdict(self.env.core.experiment)
        self.teams = {}
        for agent in self.env.possible_agents:
            team, _ = agent.rsplit("_", 1)
            self.teams.setdefault(team, []).append(agent)
        self.team_names = tuple(self.teams)
        if len(self.team_names) != 2 or len(set(map(len, self.teams.values()))) != 1:
            raise ValueError("두 팀의 인원 수가 같은 환경이 필요합니다.")
        example = self.env.possible_agents[0]
        action_space = self.env.action_space(example)
        if not isinstance(action_space, Discrete) or action_space.start != 0:
            raise ValueError("0부터 시작하는 이산 행동 공간이 필요합니다.")
        if any(self.env.action_space(a) != action_space for a in self.env.possible_agents):
            raise ValueError("모든 에이전트의 행동 공간이 같아야 합니다.")
        self.action_count = int(action_space.n)
        self.reflection = HorizontalReflection(self.action_count)
        self.noop = Action.NOOP.index
        self.previous_actions = cfg.model["previous_actions"]
        space = self.env.observation_space(example)
        fields = ("owner", "agent_positions", "agent_alive", "agent_attack_ready",
                  "agent_respawn_wait", "self_index", "remaining_ticks", "max_ticks", "areas")
        if any(key not in space.spaces for key in (*fields, "action_mask")):
            raise ValueError("Model A 관측에 필요한 필드가 없습니다. env_adapter.py를 확인하세요.")
        self.map_shape = tuple(space["owner"].shape)
        self.team_size = int(space["self_index"].n)
        self.input_size = (
            sum(int(np.prod(space[key].shape)) for key in fields if key not in
                {"self_index", "remaining_ticks", "max_ticks"})
            + self.team_size + 1
            + (self.team_size * (self.action_count + 1) if self.previous_actions else 0)
        )
        self.signature = {
            "rules_version": self.game_config["rules_version"], "encoder_version": 2,
            "map_shape": list(self.map_shape), "input_size": self.input_size,
            "team_names": list(self.team_names), "team_size": self.team_size,
            "fields": {key: [list(space[key].shape), str(space[key].dtype)] for key in fields},
            "actions": [Action.from_index(index).name for index in range(self.action_count)],
            "previous_actions": self.previous_actions,
        }
        self._previous = {}

    def reset(self, seed):
        self._previous.clear()
        observations, _ = self.env.reset(seed=seed)
        self._observations = observations
        return observations

    def encode(self, observations, agents, *, mirror=False):
        """에이전트별 특징, 합법 행동 및 생존 여부를 배치로 반환합니다."""
        features, masks, alive = [], [], []
        for agent in agents:
            obs = observations[agent]
            if mirror:
                obs=self.reflection.observation(obs)
            slot = int(obs["self_index"])
            mask = np.asarray(obs["action_mask"], dtype=bool)
            if mask.shape != (self.action_count,) or not mask.any():
                raise ValueError(f"{agent}: 유효한 행동 마스크가 없습니다.")
            identity = np.eye(self.team_size, dtype=np.float32)[slot]
            scale = np.maximum(1, np.asarray(self.map_shape) - 1)
            positions = np.asarray(obs["agent_positions"], dtype=np.float32)
            positions = np.where(positions >= 0, positions / scale, -1)
            horizon = max(1, int(obs["max_ticks"]))
            chunks = [
                obs["owner"].ravel(), positions.ravel(), obs["agent_alive"],
                obs["agent_attack_ready"], np.asarray(obs["agent_respawn_wait"]) / horizon,
                identity, [obs["remaining_ticks"] / horizon],
                np.asarray(obs["areas"]) / np.prod(self.map_shape),
            ]
            if self.previous_actions:
                team = agent.rsplit("_", 1)[0]
                history = np.zeros((self.team_size, self.action_count + 1), dtype=np.float32)
                for index, ally in enumerate(self.teams[team]):
                    if ally in self._previous:
                        action=self._previous[ally]
                        if mirror:
                            action=self.reflection.action(action)
                        history[index, action] = 1
                        history[index, -1] = 1
                chunks.append(history.ravel())
            features.append(np.concatenate(chunks).astype(np.float32))
            masks.append(mask)
            alive.append(bool(obs["agent_alive"][slot]))
        features = np.stack(features)
        if features.shape[1] != self.input_size or not np.isfinite(features).all():
            raise ValueError("관측 변환 결과가 저장된 입력 규격과 다릅니다.")
        return features, np.stack(masks), np.asarray(alive, dtype=bool)

    def step(self, actions):
        observations, _, terminated, truncated, infos = self.env.step(actions)
        rewards, components = calculate_learning_rewards(
            self._observations, observations, infos, self.teams, self.reward_config)
        self._observations = observations
        result = self.env.last_result
        if self.previous_actions:
            self._previous = {
                agent.agent_id: agent.last_action.index for agent in self.env.core.snapshot().agents
                if agent.action_valid
            }
        flags = {(terminated[a], truncated[a]) for a in actions}
        if len(flags) != 1:
            raise ValueError("팀 전체가 같은 에피소드 종료 신호를 받아야 합니다.")
        terminal, cutoff = flags.pop()
        return observations, rewards, terminal, cutoff, result, components

    def replay_actions(self, actions):
        return {agent: int(Action.from_index(index)) for agent, index in actions.items()}

    def close(self):
        self.env.close()
