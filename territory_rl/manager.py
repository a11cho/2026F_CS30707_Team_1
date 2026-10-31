from .models import AGENT_IDS
from .session import ActionCollector
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
        self.collector = ActionCollector(environment)
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

    def step(self):
        """같은 완료 상태에서 여섯 컨트롤러의 행동을 모아 환경을 한 틱 진행한다.
        수집 중에는 전이하지 않는다. 결과를 지표에 누적하고 마지막 제출 행동을 저장한 뒤 반환한다.
        """
        self.collector.begin()
        for aid in AGENT_IDS:
            session = self.collector.session(aid)
            session.submit_action(self.controllers[aid].act(session.observe()))
        self.last_actions = self.collector.actions()
        result = self.environment.step(self.last_actions)
        self.metrics.update(result)
        return result

