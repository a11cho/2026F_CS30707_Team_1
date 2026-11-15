import numpy as np
from .models import AGENT_IDS, Action


class ActionCollector:
    """환경이 소유하는 틱/회차별 수집기. 물리 상태는 변경하지 않는다."""

    def __init__(self, environment):
        """비활성 수집기와 세션 무효화용 세대 번호를 준비한다."""
        self.environment = environment
        self._generation = 0
        self.clear()

    def clear(self):
        """모든 회차 버퍼와 토큰을 비운다. 이전에 발급한 세션은 무효다."""
        self._generation += 1
        self._token = None
        self.round = 0
        self._actions = {}
        self._requested = {}
        self._proposals = {}
        self._history = []

    @property
    def active(self):
        """현재 틱에서 수집이 진행 중인지 반환한다."""
        return self._token is not None

    @property
    def next_agent(self):
        """고정 ID 순서에서 아직 제출하지 않은 첫 에이전트를 반환한다."""
        return next((a for a in AGENT_IDS if a not in self._actions), None)

    def begin(self):
        """새 틱의 1차 수집을 시작한다. 진행 중 수집을 덮어쓰지 않는다."""
        env = self.environment
        env._require_reset()
        if env.terminated or self.active:
            raise RuntimeError("cannot begin a finished or active decision")
        self.clear()
        self.round = 1
        self._set_token()

    def _set_token(self):
        """현재 에피소드·틱·회차·세대에 묶인 토큰을 갱신한다."""
        self._token = (self.environment.episode_id, self.environment.tick, self.round, self._generation)

    def _check(self, token):
        """이전 틱/회차/reset 세션 사용을 거부한다."""
        if token is None or token != self._token or token[:2] != (self.environment.episode_id, self.environment.tick):
            raise RuntimeError("stale episode/tick/round session")

    def session(self, aid):
        """현재 수집 회차와 에이전트 ID에 묶인 세션을 반환한다."""
        self._check(self._token)
        if aid not in AGENT_IDS:
            raise ValueError("unknown agent ID")
        return AgentSession(self, aid, self._token)

    def submit(self, aid, action, token):
        """한 회차에 한 번 검증·정규화하여 저장한다. 순차 전략은 ID 순서를 강제한다."""
        self._check(token)
        if aid in self._actions:
            raise ValueError("one submission per agent per round")
        if self.environment.decision_strategy == "sequential" and aid != self.next_agent:
            raise ValueError("submit in AGENT_IDS order for sequential decisions")
        requested = Action.parse(action)
        applied = requested if self.environment.action_mask(aid)[requested.index] else Action.NOOP
        self._requested[aid], self._actions[aid] = requested, applied

    def context(self, aid):
        """관측자에게 공개 가능한 같은 틱 행동의 복사본을 반환한다. 적 행동은 제외한다."""
        codes = np.zeros(3, dtype=np.int8)
        if not self.active:
            return codes, 0, "none"
        self._check(self._token)
        strategy = self.environment.decision_strategy
        source = self._proposals if strategy == "feedback" and self.round == 2 else self._actions
        team = aid.split("_")[0]
        for number in range(3):
            key = f"{team}_{number}"
            visible = strategy == "sequential" or (strategy == "feedback" and self.round == 2) or key == aid
            if visible and key in source:
                codes[number] = int(source[key])
        label = "first_pass_proposal" if strategy == "feedback" and self.round == 2 else "current_committed"
        return codes, self.round, label

    def finish_round(self):
        """누락을 정지로 채워 회차를 마감한다. 되먹임 1차면 2차로 전환하고 None을 반환한다.

        최종 회차라면 (원요청 행동, 회차 로그)를 반환한다. 최종 step에서 원요청을 다시
        검증하므로 금지 행동의 rejected 이벤트를 잃지 않는다. 공개 제안은 정규화된 값이다.
        """
        self._check(self._token)
        for aid in AGENT_IDS:
            if aid not in self._actions:
                self.submit(aid, Action.NOOP, self._token)
        self._history.append({"round": self.round,
                              "requested": {a: int(v) for a, v in self._requested.items()},
                              "applied": {a: int(v) for a, v in self._actions.items()}})
        if self.environment.decision_strategy == "feedback" and self.round == 1:
            self._proposals = self._actions.copy()
            self._actions, self._requested = {}, {}
            self.round = 2
            self._set_token()
            return None
        return self._requested.copy(), list(self._history)


class AgentSession:
    def __init__(self, collector, aid, token):
        """접속한 팀·번호와 수집 토큰을 보관한다. 전이/reset 권한은 제공하지 않는다."""
        self._collector, self.agent_id, self._token = collector, aid, token
        self.team_id, number = aid.rsplit("_", 1)
        self.agent_number = int(number)

    def observe(self):
        """저장된 팀·번호로 환경에 관측을 요청한다. 전략/컨텍스트는 환경이 선택한다."""
        self._collector._check(self._token)
        return self._collector.environment.observe(self.team_id, self.agent_number)

    def available_actions(self):
        """마스크를 실행 코드로 변환한다. 사망자는 사용자 선택 가능한 행동이 없다."""
        obs = self.observe()
        if not obs["agent_alive"][obs["self_index"]]:
            return ()
        return tuple(Action.from_index(i) for i in np.flatnonzero(obs["action_mask"]))

    def submit_action(self, action):
        """현재 회차의 행동을 한 번 제출한다. 관측한 1차 제안과 최종 행동은 구분된다."""
        self._collector.submit(self.agent_id, action, self._token)
