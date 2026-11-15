from typing import Protocol
from collections import deque
import numpy as np
from .models import Action


class AgentController(Protocol):
    def act(self, observation: dict) -> Action:
        """관측 사전에서 한 개의 행동을 선택하는 컨트롤러 계약이다. 환경의 틱을 직접 진행하지 않는다."""
        ...


class NoopController:
    def act(self, observation):
        """관측 내용에 관계없이 정지 행동을 반환한다. 다른 규칙을 고립시켜 테스트할 때 사용한다."""
        return Action.NOOP


class RandomController:
    def __init__(self, seed=None):
        """정책 선택 전용 RNG를 seed로 초기화해 환경의 부활 난수와 분리한다."""
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        """관측의 action_mask가 허용한 행동 중 하나를 균등하게 뽑아 반환한다. 정책 RNG만 소비한다."""
        return Action.from_index(int(self.rng.choice(np.flatnonzero(observation["action_mask"]))))


class FrontierController(RandomController):
    """요청자 기준 전체 맵만 사용하는 점령/경계 복귀용 scripted 컨트롤러다."""

    def __init__(self, seed=None, style="expand"):
        """전용 RNG와 확장·공격·방어 중 행동 스타일을 준비한다."""
        super().__init__(seed)
        if style not in ("expand", "aggressive", "defend"):
            raise ValueError("unknown scripted style")
        self.style = style

    def act(self, observation):
        """인접 적 땅을 공격하고, 공격할 땅이 없으면 아군 땅을 따라 경계까지 BFS로 이동한다.

        사망 중에는 정지한다. 모든 판단은 상대적 관측만 사용하며 내부 환경에 접근하지 않는다.
        """
        me = observation["self_index"]
        if not observation["agent_alive"][me]:
            return Action.NOOP
        owner, mask = observation["owner"], observation["action_mask"]
        start = tuple(observation["agent_positions"][me])
        allies = {tuple(p) for i, p in enumerate(observation["agent_positions"][:3])
                  if i != me and observation["agent_alive"][i]}
        enemies = {tuple(p) for i, p in enumerate(observation["agent_positions"][3:], 3)
                   if observation["agent_alive"][i]}

        def contains(p):
            """관측 격자의 경계 안에 있는 좌표인지 확인한다."""
            return 0 <= p[0] < owner.shape[0] and 0 <= p[1] < owner.shape[1]

        def frontier(p):
            """주어진 칸이 적 땅과 상하좌우로 맞닿는지 확인한다."""
            return any(contains(q) and owner[q] == -1 for q in (a.target(p) for a in Action if a.is_move))

        attacks = []
        for action in Action:
            if not action.is_attack or not mask[action.index]:
                continue
            p = action.target(start)
            if owner[p] == -1:
                attacks.append((1 + (3 if self.style == "aggressive" and p in enemies else 0), action))
            elif self.style == "defend" and p in allies:
                attacks.append((5, action))
        if attacks:
            best = max(score for score, _ in attacks)
            return Action(int(self.rng.choice([a for score, a in attacks if score == best])))
        if frontier(start):
            return Action.NOOP
        queue, visited = deque([(start, Action.NOOP)]), {start}
        directions = [Action(int(i)) for i in self.rng.permutation([int(a) for a in Action if a.is_move])]
        while queue:
            cell, first = queue.popleft()
            if cell != start and frontier(cell):
                return first
            for action in directions:
                q = action.target(cell)
                if q in visited or not contains(q) or owner[q] != 1 or q in allies or q in enemies:
                    continue
                visited.add(q)
                queue.append((q, action if cell == start else first))
        return Action.NOOP
