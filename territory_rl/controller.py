from typing import Protocol
from collections import deque
import numpy as np
from .models import Action


class AgentController(Protocol):
    def act(self, observation: dict) -> Action: ...


class NoopController:
    def act(self, observation):
        return Action.NOOP


class RandomController:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return Action(int(self.rng.choice(np.flatnonzero(observation["action_mask"]))))


class FrontierController(RandomController):
    """Local-only baseline: capture adjacent cells, otherwise path to visible frontier."""
    def __init__(self, seed=None, style="expand"):
        super().__init__(seed)
        if style not in ("expand", "aggressive", "defend"):
            raise ValueError("unknown scripted style")
        self.style = style

    def act(self, observation):
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

