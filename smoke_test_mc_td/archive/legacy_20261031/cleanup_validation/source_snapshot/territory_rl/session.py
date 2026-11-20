from .models import AGENT_IDS, Action


class ActionCollector:
    def __init__(self, environment):
        """환경 참조와 비어 있는 행동 버퍼를 준비한다. 수집 전에 begin을 호출해야 한다."""
        self.environment = environment
        self._actions = {}
        self._token = None

    def begin(self):
        """현재 (에피소드 ID, 틱)을 수집 토큰으로 고정하고 이전 행동 버퍼를 비운다.
        같은 틱에 다시 호출해도 이전 제출을 지우므로 매 틱 수집 시작 시 한 번만 호출한다.
        """
        self._token = (self.environment.episode_id, self.environment.tick)
        self._actions.clear()

    def session(self, aid):
        """현재 수집 토큰과 aid에 묶인 AgentSession을 반환한다.
        존재하지 않는 ID는 KeyError, begin 이전 호출은 RuntimeError로 알린다.
        """
        if aid not in AGENT_IDS:
            raise KeyError(aid)
        if self._token is None:
            raise RuntimeError("begin a collection cycle first")
        return AgentSession(self, aid, self._token)

    def _check(self, token):
        """토큰이 현재 수집 주기와 환경의 에피소드·틱에 모두 일치하는지 검사한다.
        전이나 reset 이후 남아 있는 오래된 세션 접근은 RuntimeError로 거부한다.
        """
        if token != self._token or token != (self.environment.episode_id, self.environment.tick):
            raise RuntimeError("stale episode/tick session")

    def actions(self):
        """현재 수집된 ID별 행동을 복사해 반환한다. 누락 행동의 정지 보완은 코어가 담당한다."""
        self._check(self._token)
        return dict(self._actions)


class AgentSession:
    """Per-tick facade; no game mutation or step/reset API is exposed."""
    def __init__(self, collector, aid, token):
        """collector, 에이전트 ID, 발급 시 토큰을 저장한다. 게임 상태는 소유하지 않는다."""
        self._collector, self.agent_id, self._token = collector, aid, token

    def observe(self):
        """세션 유효성을 검사한 뒤 해당 에이전트의 관측 사전을 반환한다. 틱은 진행하지 않는다."""
        self._collector._check(self._token)
        return self._collector.environment.observe(self.agent_id)

    def available_actions(self):
        """현재 mask에서 허용한 Action들을 튜플로 반환한다.
        사망 대기 중에는 빈 튜플을 반환한다. 환경 API의 처리용 NOOP과 사용자 선택 가능 행동을 구별한다.
        """
        obs = self.observe()
        if not obs["self_features"][2]:
            return ()
        return tuple(Action(i) for i, valid in enumerate(obs["action_mask"]) if valid)

    def submit_action(self, action):
        """행동값을 검증해 이번 틱 버퍼에 한 번 제출한다.
        동일 ID의 중복 제출은 ValueError로 거부한다. 실제 상태 전이와 mask 판정은 이후 코어 step이 수행한다.
        """
        self._collector._check(self._token)
        if self.agent_id in self._collector._actions:
            raise ValueError("one submission per agent per tick")
        self._collector._actions[self.agent_id] = Action.parse(action)

