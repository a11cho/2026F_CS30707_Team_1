from .models import AGENT_IDS
from .session import ActionCollector
from .metrics import EpisodeMetrics


class EpisodeManager:
    def __init__(self, environment, controllers):
        if set(controllers) != set(AGENT_IDS):
            raise ValueError("provide a controller for each agent")
        self.environment = environment
        self.controllers = dict(controllers)
        self.collector = ActionCollector(environment)
        self.metrics = EpisodeMetrics()
        self.last_actions = {}

    def reset(self, seed=None, options=None):
        snapshot = self.environment.reset(seed, options)
        self.metrics = EpisodeMetrics()
        self.last_actions = {}
        for controller in self.controllers.values():
            if hasattr(controller, "clear"):
                controller.clear()
        return snapshot

    def step(self):
        self.collector.begin()
        for aid in AGENT_IDS:
            session = self.collector.session(aid)
            session.submit_action(self.controllers[aid].act(session.observe()))
        self.last_actions = self.collector.actions()
        result = self.environment.step(self.last_actions)
        self.metrics.update(result)
        return result

