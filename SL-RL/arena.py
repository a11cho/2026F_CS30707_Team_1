"""원본 3인 환경과 같은 규칙을 쓰는 1인 환경을 정책에 연결합니다."""
import numpy as np
from territory_rl.agent import AgentState
from territory_rl.attack import resolve_attacks
from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.controller import FrontierController, NoopController, RandomController
from territory_rl.environment import TerritoryEnvironment
from territory_rl.map import TileMap
from territory_rl.models import Action, StateSnapshot
from territory_rl.movement import resolve_movements
from territory_rl.respawn import respawn_due_agents

SIDES = ("red", "blue")
BOTS = ("random", "expand", "aggressive", "defend", "noop")


def seed_for(seed, *indices):
    return int(np.random.SeedSequence([seed, *indices]).generate_state(1)[0])


def opening_row(seed, episode, height):
    """높이만큼의 경기마다 모든 행을 정확히 한 번 방문합니다."""
    order = np.random.default_rng(seed_for(seed, episode // height, 31)).permutation(height)
    return int(order[episode % height])


def bot(name, seed):
    if name == "noop":
        return NoopController()
    if name == "random":
        return RandomController(seed)
    return FrontierController(seed, style=name)


class Duel:
    """실제 두 명만 생성하고 원본 공격·동시 이동·부활 함수를 호출합니다."""
    action_mask = TerritoryEnvironment.action_mask
    _require_reset = TerritoryEnvironment._require_reset

    def __init__(self, config, seed, row):
        self.config = config
        self._initialized = True
        self.tick = 0
        self.terminated = False
        self.rng = np.random.default_rng(seed)
        self._map = TileMap(config.height, config.width)
        mid = config.width // 2
        self._agents = {f"{side}_0": AgentState(f"{side}_0", i, (row, mid - 1 + i))
                        for i, side in enumerate(SIDES)}

    def snapshot(self):
        return StateSnapshot(self._map.owner.copy(), tuple(a.snapshot() for a in self._agents.values()),
                             self.tick, self.config.max_ticks, self._map.areas(), self.terminated)

    def step(self, requested):
        if self.terminated:
            raise RuntimeError("종료된 경기입니다.")
        actions = {}
        for aid in self._agents:
            action = Action.parse(requested[aid])
            actions[aid] = action if self.action_mask(aid)[action.index] else Action.NOOP
        self.tick += 1
        owner = self._map.owner
        agents = self._agents
        events = resolve_attacks(owner, agents, actions)
        credits = dict.fromkeys(agents, 0)
        for event in events:
            if event.kind == "capture":
                owner[event.position] = event.new_owner
                for aid in event.contributors:
                    credits[aid] += 1
        for aid, action in actions.items():
            agent = agents[aid]
            agent.last_action, agent.action_valid = action, True
            if action.is_attack:
                agent.last_attack_tick = self.tick

        def kill_exposed():
            for agent in agents.values():
                if agent.alive and owner[agent.position] != agent.team:
                    agent.alive = False
                    agent.respawn_at = self.tick + self.config.respawn_delay

        kill_exposed()
        positions = {aid: a.position for aid, a in agents.items() if a.alive}
        positions, _ = resolve_movements(positions, actions, self._map.contains)
        for aid, pos in positions.items():
            agents[aid].position = pos
        kill_exposed()
        eliminated = any(area == 0 and not any(a.alive and a.team == team for a in agents.values())
                         for team, area in enumerate(self._map.areas()))
        self.terminated = self.tick >= self.config.max_ticks or eliminated
        if not self.terminated:
            respawn_due_agents(agents, owner, self.tick, self.rng)
        return credits


class Arena:
    def __init__(self, game, size, seed, row=0):
        cfg = GameConfig(**game, initial_placement="frontier")
        self.window = 2 * max(cfg.height, cfg.width) - 1
        self.size = size
        self.teams = [[f"{side}_{i}" for i in range(size)] for side in SIDES]
        if size == 3:
            self.env = TerritoryEnvironment(cfg, ExperimentConfig(observation_window_size=self.window))
            self.env.reset(seed)
        elif size == 1:
            self.env = Duel(cfg, seed, row)
        else:
            raise ValueError("팀 인원은 1 또는 3이어야 합니다.")
        self.state = self.env.snapshot()

    def observation(self, side, index):
        if self.size == 3:
            return self.env.observe(SIDES[side], index)
        # 기존 봇이 사용하는 공개 관측 필드를 동일한 타일 코드로 구성합니다.
        from territory_rl.config import TileEncoding
        enc = TileEncoding()
        state = self.state
        me = state.agents[side]
        offset = np.array([self.window // 2] * 2) - me.position
        board = np.full((self.window, self.window), enc.outside, np.int8)
        r, c = offset
        h, w = state.owner.shape
        board[r:r+h, c:c+w] = np.where(state.owner == side, enc.friendly, enc.enemy)
        ordered = [me, state.agents[1-side]]
        for a in ordered:
            if a.alive:
                board[tuple(offset + a.position)] = (enc.friendly_agents if a.team == side else enc.enemy_agents)[0]
        return {"tilemap": board, "auxiliary": {"self_index": 0, "agent_alive": [a.alive for a in ordered],
                "action_mask": self.env.action_mask(me.agent_id)}}

    def step(self, actions, alpha):
        result = self.env.step(actions)
        self.last_result = result
        credits = result.credits if self.size == 3 else result
        self.state = self.env.snapshot()
        return {a.agent_id: self.state.areas[a.team] / self.state.owner.size + alpha * credits[a.agent_id]
                for a in self.state.agents}


def ownership(state, side):
    """양 팀 모두 자기 출발지가 왼쪽인 좌표계로 변환합니다."""
    grid = state.owner == side
    return (grid if side == 0 else np.fliplr(grid)).astype(np.float32).ravel()


def reflect(action, side):
    if side == 0:
        return action
    return {Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT,
            Action.ATTACK_LEFT: Action.ATTACK_RIGHT, Action.ATTACK_RIGHT: Action.ATTACK_LEFT}.get(action, action)


def inputs(arena, side):
    """전체 맵 공개 조건에서 자기 관측으로 복구 가능한 정보만 사용합니다."""
    state = arena.state
    h, w = state.owner.shape
    tick_scale = state.max_ticks
    own, others, masks = [], [], []
    for me in state.agents:
        if me.team != side:
            continue
        row, col = me.position
        def status(a):
            ready = a.alive and state.tick + 1 - a.last_attack_tick >= 2
            wait = max(0, (a.respawn_at or 0) - state.tick)
            return [float(a.alive), float(ready), wait / tick_scale]
        own.append([row / (h-1), (col if not side else w-1-col) / (w-1),
                    1-state.tick/tick_scale, *status(me)])
        entities = []
        for a in state.agents:
            if a.agent_id == me.agent_id:
                continue
            # 사망한 타 개체의 마지막 좌표는 원본 타일 관측에도 나타나지 않습니다.
            dr, dc = (a.position[0]-row, a.position[1]-col) if a.alive else (0, 0)
            entities.append([dr/(h-1), (-dc if side else dc)/(w-1),
                             1 if side == a.team else -1, *status(a)])
        others.append(entities)
        world = arena.env.action_mask(me.agent_id)
        masks.append([world[reflect(a, side).index] for a in Action])
    grid = np.tile(ownership(state, side), (arena.size, 1))
    return grid, np.asarray(own, np.float32), np.asarray(others, np.float32), np.asarray(masks, bool)
