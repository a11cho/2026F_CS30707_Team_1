from .models import AGENT_IDS
from .metrics import EpisodeMetrics


class EpisodeManager:
    def __init__(self, environment, controllers):
        """여섯 ID의 컨트롤러를 검증하고 행동 수집기·지표 집계기를 준비한다.
        환경을 자동 초기화하지 않으므로 실행 전에 환경 또는 manager의 reset이 필요하다.
        """
        if set(controllers) != set(AGENT_IDS):
            raise ValueError("provide a controller for each agent")
        self.environment = environment
        self.controllers = dict(controllers)
        self.metrics = EpisodeMetrics()
        self.last_actions = {}

    def reset(self, seed=None, options=None):
        """환경을 초기화하고 지표·직전 행동 및 clear가 있는 컨트롤러의 예약 입력을 비운다.
        초기 스냅샷을 반환한다. 일반 scripted 컨트롤러의 RNG를 재시드하지는 않는다.
        """
        snapshot = self.environment.reset(seed, options)
        self.metrics = EpisodeMetrics()
        self.last_actions = {}
        for controller in self.controllers.values():
            if hasattr(controller, "clear"):
                controller.clear()
        return snapshot

    def advance(self):
        """수동 조작용 결정 단계를 진행한다. 순차는 한 명, 나머지는 한 회차를 수집한다.

        되먹임 1차/미완료 순차는 None을 반환하여 입력 변경 기회를 준다.
        최종 행동이 모두 모인 때만 물리 전이·지표 집계를 한 번 수행한다.
        """
        env = self.environment
        if env.decision_round == 0:
            env.begin_decision()
        ids = [env.next_decision_agent] if env.decision_strategy == "sequential" else list(AGENT_IDS)
        for aid in ids:
            team, number = aid.rsplit("_", 1)
            session = env.agent_session(team, int(number))
            session.submit_action(self.controllers[aid].act(session.observe()))
        if env.next_decision_agent is not None:
            return None
        result = env.finish_decision_round()
        if result is not None:
            self.last_actions = dict(env.last_decision[-1]["requested"])
            self.metrics.update(result, env.snapshot())
        return result

    def step(self):
        """자동 실행용으로 모든 결정 단계(되먹임 두 회차 포함)를 완료하고 한 틱 진행한다."""
        result = self.advance()
        while result is None:
            result = self.advance()
        return result
