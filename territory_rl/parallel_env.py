import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .environment import TerritoryEnvironment
from .models import AGENT_IDS, Action


class TerritoryParallelEnv(ParallelEnv):
    metadata = {"name": "territory_v1", "render_modes": ["human", "rgb_array"], "render_fps": 5,
                "is_parallelizable": True}

    def __init__(self, config=None, experiment=None, render_mode=None):
        """규칙 엔진과 고정 관측/행동 공간을 만들고 렌더링 모드를 저장한다.
        창은 human 렌더링을 실제 호출할 때 생성하며, 경기 시작에는 별도 reset이 필요하다.
        """
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("unsupported render mode")
        self.core = TerritoryEnvironment(config, experiment)
        self.possible_agents = list(AGENT_IDS)
        self.agents = []
        self.render_mode = render_mode
        self._renderer = None
        cfg = self.core.config
        import string
        self._observation_spaces = {aid: spaces.Dict({
            "owner": spaces.Box(-1, 1, (cfg.height, cfg.width), np.int8),
            "agent_positions": spaces.Box(-1, max(cfg.height, cfg.width)-1, (6, 2), np.int32),
            "agent_teams": spaces.MultiBinary(6), "agent_alive": spaces.MultiBinary(6),
            "agent_attack_ready": spaces.MultiBinary(6),
            "agent_respawn_wait": spaces.Box(0, np.iinfo(np.int32).max, (6,), np.int32),
            "self_index": spaces.Discrete(3), "self_team": spaces.Discrete(1, start=1),
            "tick": spaces.Discrete(cfg.max_ticks+1),
            "max_ticks": spaces.Discrete(1, start=cfg.max_ticks),
            "remaining_ticks": spaces.Discrete(cfg.max_ticks+1),
            "areas": spaces.Box(0, cfg.height*cfg.width, (2,), np.int32),
            "decision_strategy": spaces.Text(32, charset=string.ascii_lowercase+"_"),
            "decision_round": spaces.Discrete(3),
            "action_source": spaces.Text(32, charset=string.ascii_lowercase+"_"),
            "team_action_codes": spaces.Box(0, 9, (3,), np.int8),
            "team_actions_known": spaces.MultiBinary(3),
            "team_action_features": spaces.Box(0, 1, (3, 9), np.float32),
            "action_mask": spaces.MultiBinary(9)}) for aid in AGENT_IDS}
        self._action_spaces = {aid: spaces.Discrete(9) for aid in AGENT_IDS}
        self.state_space = spaces.Box(-np.inf, np.inf, (self.core.config.height*self.core.config.width + 6*8 + 1,), np.float32)
        self.last_result = None

    def observation_space(self, agent):
        """agent의 관측 형식·shape·dtype을 정의한 고정 Gymnasium 공간을 반환한다."""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """agent의 고정 행동 공간 Discrete(9)를 반환한다. 틱별 제약은 관측의 action_mask에 둔다."""
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Standard wrappers may supply unrelated reset options; only scenario keys
        # cross into the strictly validated core.
        """코어를 초기화하고 (ID별 관측, ID별 info) 사전을 반환한다.
        여섯 ID를 활성 목록에 복구하고 마지막 결과를 비운다. options의 시나리오 키만
        코어에 전달하며 외부 wrapper가 넘긴 그 밖의 옵션은 무시한다.
        """
        scenario = {k: v for k, v in (options or {}).items() if k in ("owner", "agents", "tick", "decision_strategy")}
        self.core.reset(seed, scenario)
        self.agents = self.possible_agents.copy()
        self.last_result = None
        return {a: self._observe(a) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        """여섯 ID의 행동을 한 틱 처리해 관측·보상·termination·truncation·info를 반환한다.

        입력은 Discrete(9)의 인덱스 0..8이며 실행 코드 1..9로 변환한다. key 누락은 오류다. 사망 대기자는 목록에 남기고
        정지 행동으로 진행하며, 경기 종료의 마지막 관측과 보상도 여섯 명 모두에게 반환한다.
        마지막 반환을 준비한 뒤 agents를 비운다. 자체적인 외부 truncation은 만들지 않는다.
        """
        if not self.agents:
            raise RuntimeError("call reset before stepping a finished/uninitialized episode")
        if set(actions) != set(self.agents):
            raise ValueError("Parallel API requires one action for each of the six agents")
        if self.core.decision_strategy != "independent":
            raise RuntimeError("sequential/feedback requires begin_decision/agent_session/finish_decision_round")
        result = self.core.step({aid: Action.from_index(value) for aid, value in actions.items()})
        return self._result(result)

    def _observe(self, aid):
        """고정 내부 ID를 공개 관측 API의 팀·번호로 변환한다."""
        team, number = aid.rsplit("_", 1)
        return self.core.observe(team, int(number))

    def begin_decision(self):
        """순차/되먹임을 위한 명시적인 수집을 시작한다. 물리 틱은 진행하지 않는다."""
        if not self.agents:
            raise RuntimeError("reset before collecting decisions")
        self.core.begin_decision()

    def agent_session(self, team_id, agent_number):
        """실행 코드 1..9를 받는 세션을 제공한다. 표준 step의 0..8 인덱스와 구별한다."""
        return self.core.agent_session(team_id, agent_number)

    def finish_decision_round(self):
        """회차를 마감한다. 되먹임 1차는 None, 최종 회차는 표준 5개 결과를 반환한다."""
        result = self.core.finish_decision_round()
        return None if result is None else self._result(result)

    def _result(self, result):
        """공통 엔진 결과를 6명에게 반환하고 종료 후 활성 목록을 비운다."""
        self.last_result = result
        obs = {a: self._observe(a) for a in self.agents}
        term = {a: result.terminated for a in self.agents}
        trunc = {a: False for a in self.agents}
        infos = {a: {"tick": result.tick, "capture_credit": result.credits[a], "reward_components": dict(result.reward_components[a]),
                     "termination_reasons": result.termination_reasons, "eliminated_teams": result.eliminated_teams} for a in self.agents}
        if result.terminated:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return obs, dict(result.rewards), term, trunc, infos

    def state(self):
        """전체 소유권, ID순 에이전트 상태, 현재 틱을 하나의 float32 벡터로 반환한다.
        전역 정보이므로 에이전트별 제한 관측과 구별해야 한다. 사망 좌표/없는 부활 시각은 -1이다.
        """
        s = self.core.snapshot()
        features = []
        for a in s.agents:
            features.extend([*(a.position or (-1, -1)), int(a.alive), a.last_attack_tick,
                             -1 if a.respawn_at is None else a.respawn_at, a.team, int(a.last_action), int(a.action_valid)])
        return np.concatenate([s.owner.ravel(), features, [s.tick]]).astype(np.float32)

    def render(self):
        """현재 스냅샷을 표시하거나 RGB 배열로 반환한다.
        rgb_array는 uint8 이미지를 반환하고 human은 Tk 창을 갱신한다. 게임 틱은 진행하지 않는다.
        """
        from .visualization import render_rgb, TkRenderer
        if self.render_mode == "rgb_array":
            return render_rgb(self.core.snapshot())
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = TkRenderer()
            self._renderer.draw(self.core.snapshot())
            self._renderer.root.update()

    def close(self):
        """이 adapter가 생성한 렌더러 창을 닫고 참조를 해제한다. 창이 없으면 아무 일도 하지 않는다."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def parallel_env(**kwargs):
    """전달받은 설정으로 TerritoryParallelEnv를 생성하는 편의 팩토리다."""
    return TerritoryParallelEnv(**kwargs)
