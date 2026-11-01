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
        return Action(int(self.rng.choice(np.flatnonzero(observation["action_mask"]))))


class FrontierController(RandomController):
    """Local-only baseline: capture adjacent cells, otherwise path to visible frontier."""
    def __init__(self, seed=None, style="expand"):
        """전용 RNG와 확장·공격·방어 중 scripted 행동 스타일을 준비한다. 미지원 스타일은 오류다."""
        super().__init__(seed)
        if style not in ("expand", "aggressive", "defend"):
            raise ValueError("unknown scripted style")
        self.style = style

    def act(self, observation):
        """지역 관측만 사용해 공격 또는 전선 접근 행동을 선택한다.

        인접 적 타일을 공격하되 aggressive는 적 에이전트가 있는 칸을 더 우선한다.
        defend는 인접 아군 위치의 방어 공격을 우선한다. 공격할 수 없으면 BFS로
        보이는 아군 땅 안의 전선 경로를 찾고, 적 타일 앞의 cooldown 중에는 대기한다.
        환경 상태를 직접 수정하지 않으며 동률 선택 등에 정책 전용 RNG를 사용한다.
        """
        if not observation["self_features"][2]:
            return Action.NOOP
        grid, mask = observation["local_grid"], observation["action_mask"]
        radius = grid.shape[1] // 2
        center = (radius, radius)
        attacks = []
        for action in list(Action)[5:]:
            p = action.target(center)
            if mask[action] and grid[1][p]:
                score = 1 + (3*grid[5][p] if self.style == "aggressive" else 0)
                attacks.append((score, action))
            if self.style == "defend" and mask[action] and grid[4][p]:
                attacks.append((5, action))
        if attacks:
            best = max(score for score, _ in attacks)
            return Action(int(self.rng.choice([a for score, a in attacks if score == best])))
        # BFS through friendly, visibly unoccupied cells to a cell next to enemy land.
        queue = deque([(center, Action.NOOP)])
        visited = {center}
        directions = list(self.rng.permutation([1, 2, 3, 4]))
        while queue:
            cell, first = queue.popleft()
            if cell != center and any(0 <= q[0] < grid.shape[1] and 0 <= q[1] < grid.shape[2] and grid[1][q]
                                      for q in (Action(i).target(cell) for i in range(1, 5))):
                return first
            for i in directions:
                action = Action(int(i))
                q = action.target(cell)
                if q in visited or not (0 <= q[0] < grid.shape[1] and 0 <= q[1] < grid.shape[2]):
                    continue
                if not grid[0][q] or grid[4][q] or grid[5][q]:
                    continue
                visited.add(q)
                queue.append((q, action if cell == center else first))
        # Attack cooldown often makes waiting at the frontier preferable to suicide.
        if any(grid[1][Action(i).target(center)] for i in range(1, 5)):
            return Action.NOOP
        moves = [Action(i) for i in range(1, 5) if mask[i] and grid[0][Action(i).target(center)]
                 and not grid[4][Action(i).target(center)]]
        return Action(int(self.rng.choice(moves))) if moves else Action.NOOP

