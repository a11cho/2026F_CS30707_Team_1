from .models import AGENT_IDS, Action


class ActionCollector:
    def __init__(self, environment):
        self.environment = environment
        self._actions = {}
        self._token = None

    def begin(self):
        self._token = (self.environment.episode_id, self.environment.tick)
        self._actions.clear()

    def session(self, aid):
        if aid not in AGENT_IDS:
            raise KeyError(aid)
        if self._token is None:
            raise RuntimeError("begin a collection cycle first")
        return AgentSession(self, aid, self._token)

    def _check(self, token):
        if token != self._token or token != (self.environment.episode_id, self.environment.tick):
            raise RuntimeError("stale episode/tick session")

    def actions(self):
        self._check(self._token)
        return dict(self._actions)


class AgentSession:
    """Per-tick facade; no game mutation or step/reset API is exposed."""
    def __init__(self, collector, aid, token):
        self._collector, self.agent_id, self._token = collector, aid, token

    def observe(self):
        self._collector._check(self._token)
        return self._collector.environment.observe(self.agent_id)

    def available_actions(self):
        obs = self.observe()
        if not obs["self_features"][2]:
            return ()
        return tuple(Action(i) for i, valid in enumerate(obs["action_mask"]) if valid)

    def submit_action(self, action):
        self._collector._check(self._token)
        if self.agent_id in self._collector._actions:
            raise ValueError("one submission per agent per tick")
        self._collector._actions[self.agent_id] = Action.parse(action)

