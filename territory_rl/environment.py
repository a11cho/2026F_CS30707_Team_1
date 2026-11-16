from dataclasses import replace
from numbers import Integral
from types import MappingProxyType
import numpy as np
from .agent import AgentState
from .attack import resolve_attacks
from .config import ExperimentConfig, GameConfig
from .events import Event
from .map import TileMap
from .models import AGENT_IDS, Action, StateSnapshot, StepResult
from .movement import resolve_movements
from .respawn import respawn_due_agents
from .reward import calculate_rewards


class TerritoryEnvironment:
    """물리 상태는 reset/step, 결정 버퍼는 명시적 수집 API에서만 갱신한다."""

    def __init__(self, config=None, experiment=None):
        """게임/실험 설정과 에피소드 식별자를 준비한다. 실제 맵·에이전트는 reset에서 생성한다."""
        self.config = config or GameConfig()
        self._experiment = experiment or ExperimentConfig()
        from .session import ActionCollector
        self._collector = ActionCollector(self)
        self.last_decision = []
        self.episode_id = 0
        self._initialized = False

    @property
    def experiment(self):
        """에피소드 중 교체할 수 없는 실험 설정을 반환한다."""
        return self._experiment

    @property
    def decision_strategy(self):
        """환경에 저장된 현재 결정 전략을 반환한다."""
        return self.experiment.decision_strategy

    @property
    def decision_round(self):
        """현재 수집 회차를 반환한다. 수집 전/전이 후에는 0이다."""
        return self._collector.round

    @property
    def next_decision_agent(self):
        """수집 중 다음 제출자를 반환한다. 비활성 상태에서는 None이다."""
        return self._collector.next_agent if self._collector.active else None

    def begin_decision(self):
        """환경 소유 수집기에서 새 틱의 결정을 시작한다."""
        self._collector.begin()

    def agent_session(self, team_id, agent_number):
        """팀·번호를 검증하고 현재 회차 세션을 발급한다."""
        return self._collector.session(self._agent_id(team_id, agent_number))

    def finish_decision_round(self):
        """되먹임 1차는 전이 없이 마감하고, 최종 회차는 한 틱을 실행한다."""
        finished = self._collector.finish_round()
        if finished is None:
            return None
        requested, trace = finished
        self._collector.clear()
        result = self.step(requested)
        self.last_decision = trace
        return result

    @staticmethod
    def _agent_id(team_id, agent_number):
        """공개 API의 red/blue 및 팀 내 정수 번호 0..2를 내부 ID로 변환한다."""
        if team_id not in ("red", "blue") or isinstance(agent_number, bool) or not isinstance(agent_number, Integral) or not 0 <= agent_number < 3:
            raise ValueError("expected team red/blue and agent_number 0..2")
        return f"{team_id}_{int(agent_number)}"

    def reset(self, seed=None, options=None):
        """새 에피소드 상태와 환경 전용 RNG를 만들고 초기 StateSnapshot을 반환한다.

        seed가 같으면 초기 배치와 이후 동일 행동열에 따른 무작위 부활을 재현할 수 있다.
        options에는 owner, agents, tick 및 새 에피소드의 decision_strategy/reward_signal_mask를 지정할 수 있다.
        사용자 상태는 검증·복사한 뒤 반영하고 episode_id를 늘려 기존 세션을 무효화한다.
        """
        options = dict(options or {})
        if set(options) - {"owner", "agents", "tick", "decision_strategy", "reward_signal_mask"}:
            raise ValueError("unsupported scenario option")
        experiment = replace(self.experiment, decision_strategy=options.get("decision_strategy", self.decision_strategy),
                             reward_signal_mask=options.get("reward_signal_mask", self.experiment.reward_signal_mask))
        cfg = self.config
        rng = np.random.default_rng(seed)
        tile_map = TileMap(cfg.height, cfg.width, options.get("owner"))
        tick = options.get("tick", 0)
        if type(tick) is not int or not 0 <= tick < cfg.max_ticks:
            raise ValueError("scenario tick must be an integer in [0,max_ticks)")
        if "agents" in options:
            sequence = list(options["agents"])
            if len(sequence) != 6 or any(not isinstance(a, AgentState) for a in sequence):
                raise ValueError("scenario agents must contain six AgentState objects")
            agents = {a.agent_id: replace(a) for a in sequence}
        else:
            if "owner" in options:
                raise ValueError("custom owner requires explicit scenario agents")
            if cfg.initial_placement == "frontier":
                rows = sorted(int(r) for r in rng.choice(cfg.height, 3, replace=False))
                left = [(r, cfg.width // 2 - 1) for r in rows]
            else:
                cells = [(r, c) for r in range(cfg.height) for c in range(cfg.width // 2)]
                left = [cells[int(i)] for i in rng.choice(len(cells), 3, replace=False)]
            right = [(r, cfg.width - 1 - c) for r, c in left]
            agents = {aid: AgentState(aid, i // 3, (left + right)[i]) for i, aid in enumerate(AGENT_IDS)}
        self._validate_state(tile_map, agents, tick)
        if any(area == 0 and not any(a.alive and a.team == team for a in agents.values())
               for team, area in enumerate(tile_map.areas())):
            raise ValueError("scenario starts with an eliminated team")
        self._experiment = experiment
        self._collector.clear()
        self.last_decision = []
        self._map, self._agents, self._rng = tile_map, agents, rng
        self.tick, self.terminated = tick, False
        self.episode_id += 1
        self._initialized = True
        return self.snapshot()

    @staticmethod
    def _validate_state(tile_map, agents, tick):
        """에이전트 ID·팀·좌표·공격/부활 시점의 상태 불변식을 검사한다.

        생존자는 중복 없는 아군 타일에 있어야 하며, 사망자는 부활 예정 틱이 있어야 한다.
        위반 시 ValueError를 발생시킨다. 검사 과정에서 전달된 상태의 행동과 좌표를
        각각 Action과 정수 튜플로 정규화하므로 완전한 읽기 전용 함수는 아니다.
        """
        if set(agents) != set(AGENT_IDS):
            raise ValueError("expected red_0..2 and blue_0..2")
        seen = set()
        for i, aid in enumerate(AGENT_IDS):
            a = agents[aid]
            if a.team != i // 3:
                raise ValueError("agent team does not match ID")
            if not isinstance(a.last_attack_tick, Integral) or a.last_attack_tick > tick:
                raise ValueError("last attack cannot be in the future")
            a.last_action = Action.parse(a.last_action)
            if a.alive:
                p = a.position
                if len(p) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool) for x in p):
                    raise ValueError("positions must contain two integers")
                p = tuple(map(int, p))
                if not tile_map.contains(p) or tile_map.owner[p] != a.team or p in seen:
                    raise ValueError("living agents require unique friendly positions")
                if a.respawn_at is not None:
                    raise ValueError("living agents cannot have a pending respawn")
                a.position = p
                seen.add(p)
            elif not isinstance(a.respawn_at, Integral) or a.respawn_at < 0:
                raise ValueError("dead agents require a nonnegative respawn_at")

    def _require_reset(self):
        """reset이 한 번도 실행되지 않았다면 RuntimeError를 발생시킨다."""
        if not self._initialized:
            raise RuntimeError("call reset before accessing the environment")

    def snapshot(self):
        """화면·기록용 전역 상태를 내부 저장소와 분리된 스냅샷으로 반환한다.
        소유권 배열은 복사한 뒤 쓰기를 막고, 에이전트는 변경 불가능한 값 객체로 변환한다.
        틱과 RNG는 진행시키지 않는다.
        """
        self._require_reset()
        owner = self._map.owner.copy()
        owner.flags.writeable = False
        return StateSnapshot(owner, tuple(self._agents[a].snapshot() for a in AGENT_IDS),
                             self.tick, self.config.max_ticks, self._map.areas(), self.terminated)

    def action_mask(self, aid):
        """aid가 다음 틱에 제출할 수 있는 행동을 길이 9의 int8 배열로 반환한다.

        정지 코드 1(마스크 인덱스 0)은 항상 허용하며 사망 또는 경기 종료 상태에서는 정지만 남긴다.
        경계와 본인 공격 cooldown만 검사한다. 점유 칸·적 타일도 같은 틱에
        비워지거나 아군이 점령할 수 있으므로 그 이유만으로 이동을 차단하지 않는다.
        """
        self._require_reset()
        a = self._agents[aid]
        mask = np.zeros(9, dtype=np.int8)
        mask[0] = 1
        if a.alive and not self.terminated:
            for action in list(Action)[1:]:
                if self._map.contains(action.target(a.position)) and (not action.is_attack or a.can_attack(self.tick + 1)):
                    mask[action.index] = 1
        return mask

    def observe(self, team_id, agent_number):
        """팀·번호에 맞춘 전체 맵과 전략별 현재 행동을 독립적인 배열로 반환한다.
        아군 영역은 1, 적군 영역은 -1이다. 조회는 물리 상태·회차·난수를 진행하지 않는다.
        """
        from .observation import build_observation
        aid = self._agent_id(team_id, agent_number)
        return build_observation(self.snapshot(), aid, self.config, self.decision_strategy,
                                 self.action_mask(aid), self._collector.context(aid))

    def step(self, actions):
        """ID별 행동 사전을 받아 정확히 한 틱을 진행하고 StepResult를 반환한다.

        미제출은 정지로 채운다. 잘못된 ID·행동값은 상태 변경 전에 오류로 알리고,
        범위 안이지만 mask로 금지된 행동은 정지로 바꿔 거부 이벤트를 남긴다.
        공격 집계/반영 → 공격 후 사망 → 동시 이동 → 이동 후 사망 → 부활 → 보상 순서다.
        종료 틱에는 부활을 생략한다. 종료 이후 호출하려면 먼저 reset해야 한다.
        """
        self._require_reset()
        if self.terminated:
            raise RuntimeError("episode is over; call reset")
        if self._collector.active:
            raise RuntimeError("finish decision rounds before stepping")
        if set(actions) - set(AGENT_IDS):
            raise ValueError("unknown agent ID")
        # Parse ALL actions before mutating anything, including RNG.
        requested = {aid: Action.parse(actions.get(aid, Action.NOOP)) for aid in AGENT_IDS}
        applied = {}
        events = []
        for aid in AGENT_IDS:
            action = requested[aid]
            applied[aid] = action if self.action_mask(aid)[action.index] else Action.NOOP
            if applied[aid] != action:
                events.append(Event("rejected", aid, reason="action_mask", requested=int(action), applied=int(Action.NOOP)))
        tick = self.tick + 1
        before = self._map.areas()
        attack_events = resolve_attacks(self._map.owner, self._agents, applied)
        events.extend(attack_events)
        for aid, action in applied.items():
            a = self._agents[aid]
            a.last_action, a.action_valid = action, True
            if action.is_attack:
                a.last_attack_tick = tick
        for event in attack_events:
            if event.kind == "capture":
                self._map.owner[event.position] = event.new_owner

        def kill_exposed(phase):
            """현재 타일이 적 소유인 생존자를 사망 처리하고 부활 예정 틱을 기록한다.
            phase는 attack 또는 movement로, 사건의 사망 원인 단계를 구별한다.
            공격 직후 사망이면 예약된 이동도 취소하며 위치를 None으로 바꿔 즉시 점유를 해제한다.
            """
            for aid in AGENT_IDS:
                a = self._agents[aid]
                if a.alive and self._map.owner[a.position] != a.team:
                    events.append(Event("death", aid, a.position, reason=phase,
                                        due_tick=tick + self.config.respawn_delay))
                    if phase == "attack" and applied[aid].is_move:
                        events.append(Event("move", aid, applied[aid].target(a.position), a.position, "killed_before_move"))
                    a.position = None
                    a.respawn_at = tick + self.config.respawn_delay

        kill_exposed("attack")
        positions = {aid: a.position for aid, a in self._agents.items() if a.alive}
        positions, moves = resolve_movements(positions, applied, self._map.contains)
        events.extend(moves)
        for aid, position in positions.items():
            self._agents[aid].position = position
        kill_exposed("movement")
        after = self._map.areas()
        eliminated = tuple(("red", "blue")[team] for team, area in enumerate(after)
                           if area == 0 and not any(a.alive and a.team == team for a in self._agents.values()))
        reasons = tuple(reason for reason, condition in (("time_limit", tick >= self.config.max_ticks),
                                                        ("team_eliminated", bool(eliminated))) if condition)
        terminated = bool(reasons)
        if not terminated:
            events.extend(respawn_due_agents(self._agents, self._map.owner, tick, self._rng))
        rewards, credits, components = calculate_rewards(self.experiment, before, after, events, terminated)
        self.tick, self.terminated = tick, terminated
        self._collector.clear()
        self.last_decision = []
        self._validate_state(self._map, self._agents, tick)
        return StepResult(tick, tuple(events), before, after, MappingProxyType(rewards),
                          MappingProxyType(credits), MappingProxyType({a: MappingProxyType(v) for a, v in components.items()}),
                          terminated, termination_reasons=reasons, eliminated_teams=eliminated)

