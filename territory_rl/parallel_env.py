import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .environment import TerritoryEnvironment
from .models import AGENT_IDS


class TerritoryParallelEnv(ParallelEnv):
    metadata = {"name": "territory_v0", "render_modes": ["human", "rgb_array"], "render_fps": 5,
                "is_parallelizable": True}

    def __init__(self, config=None, experiment=None, render_mode=None):
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("unsupported render mode")
        self.core = TerritoryEnvironment(config, experiment)
        self.possible_agents = list(AGENT_IDS)
        self.agents = []
        self.render_mode = render_mode
        self._renderer = None
        size = 2*self.core.config.observation_radius+1
        self._observation_spaces = {aid: spaces.Dict({
            "local_grid": spaces.Box(0, 1, (6, size, size), np.float32),
            "self_features": spaces.Box(0, 1, (6,), np.float32),
            "teammate_features": spaces.Box(0, 1, (2, 14), np.float32),
            "teammate_feature_mask": spaces.MultiBinary((2, 14)),
            "action_mask": spaces.MultiBinary(9)}) for aid in AGENT_IDS}
        self._action_spaces = {aid: spaces.Discrete(9) for aid in AGENT_IDS}
        self.state_space = spaces.Box(-np.inf, np.inf, (self.core.config.height*self.core.config.width + 6*8 + 1,), np.float32)
        self.last_result = None

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Standard wrappers may supply unrelated reset options; only scenario keys
        # cross into the strictly validated core.
        scenario = {k: v for k, v in (options or {}).items() if k in ("owner", "agents", "tick")}
        self.core.reset(seed, scenario)
        self.agents = self.possible_agents.copy()
        self.last_result = None
        return {a: self.core.observe(a) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("call reset before stepping a finished/uninitialized episode")
        if set(actions) != set(self.agents):
            raise ValueError("Parallel API requires one action for each of the six agents")
        result = self.core.step(actions)
        self.last_result = result
        obs = {a: self.core.observe(a) for a in self.agents}
        term = {a: result.terminated for a in self.agents}
        trunc = {a: False for a in self.agents}
        infos = {a: {"tick": result.tick, "capture_credit": result.credits[a]} for a in self.agents}
        if result.terminated:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return obs, dict(result.rewards), term, trunc, infos

    def state(self):
        s = self.core.snapshot()
        features = []
        for a in s.agents:
            features.extend([*(a.position or (-1, -1)), int(a.alive), a.last_attack_tick,
                             -1 if a.respawn_at is None else a.respawn_at, a.team, int(a.last_action), int(a.action_valid)])
        return np.concatenate([s.owner.ravel(), features, [s.tick]]).astype(np.float32)

    def render(self):
        from .visualization import render_rgb, TkRenderer
        if self.render_mode == "rgb_array":
            return render_rgb(self.core.snapshot())
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = TkRenderer()
            self._renderer.draw(self.core.snapshot())
            self._renderer.root.update()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def parallel_env(**kwargs):
    return TerritoryParallelEnv(**kwargs)
