"""Observable cooperation in the seven scenes defined by the project team."""

from collections import Counter, defaultdict, deque
from functools import cached_property, lru_cache
from itertools import combinations

from territory_rl.attack import resolve_attacks
from territory_rl.models import Action
from territory_rl.movement import resolve_movements

from .cooperation import Opportunity


SCENES = {
    "breakthrough": "집중 공격으로 방어 돌파",
    "cover": "위험한 아군 엄호",
    "capture_entry": "점령과 진입 연결",
    "yield_move": "길 양보와 연쇄 이동",
    "split_targets": "공격 목표 분담",
    "cooldown_rotation": "쿨다운 동안 교대",
    "role_switch": "역할 분담과 전환",
}
ROLE_WINDOW = 4


@lru_cache(maxsize=8)
def neighbors(height, width):
    return {
        (row, col): tuple(
            (r, c) for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if 0 <= r < height and 0 <= c < width
        )
        for row in range(height) for col in range(width)
    }


class SceneTick:
    def __init__(self, before, result, team, learners, proposals, applied):
        self.owner = before.owner
        self.adjacent = neighbors(*self.owner.shape)
        self.team = team
        self.learners = learners
        self.proposals = proposals
        self.applied = applied
        self.tick = result.tick
        self.agents = {agent.agent_id: agent for agent in before.agents if agent.alive}
        self.allies = {aid: self.agents[aid].position for aid in learners if aid in self.agents}
        self.occupant = {agent.position: aid for aid, agent in self.agents.items()}
        self.ready = {aid for aid in self.allies if self.tick - self.agents[aid].last_attack_tick >= 2}
        self.enemy_hits = Counter()
        self.ally_attacks = {}
        self.captures = {}
        self.moves = {}
        self.deaths = set()
        for event in result.events:
            if event.kind == "attack":
                if event.agent_id in self.allies:
                    self.ally_attacks[event.agent_id] = event.position
                else:
                    self.enemy_hits[event.position] += 1
            elif event.kind == "capture":
                self.captures[event.position] = event
            elif event.kind == "move" and event.agent_id in self.allies and event.reason == "success":
                self.moves[event.agent_id] = event.position
            elif event.kind == "death":
                self.deaths.add(event.agent_id)
        # A move event can succeed mechanically and still kill the mover.
        self.moves = {aid: pos for aid, pos in self.moves.items() if aid not in self.deaths}

    @cached_property
    def attackers(self):
        by_cell = defaultdict(set)
        for aid, position in self.allies.items():
            if aid in self.ready:
                for cell in self.adjacent[position]:
                    by_cell[cell].add(aid)
        return by_cell

    @cached_property
    def weak_targets(self):
        return {
            aid: {cell for cell in self.adjacent[self.allies[aid]]
                  if self.owner[cell] != self.team and not self.enemy_hits[cell]}
            for aid in self.ready
        }

    def owner_after(self, cell):
        event = self.captures.get(cell)
        return self.owner[cell] if event is None else event.new_owner

    def replay_moves(self, overrides):
        """Recalculate the attack and move phases, before any random respawn."""
        actions = {aid: Action(self.applied[aid]) for aid in self.agents}
        actions.update(overrides)
        board = self.owner.copy()
        for event in resolve_attacks(board, self.agents, actions):
            if event.kind == "capture":
                board[event.position] = event.new_owner
        surviving = {aid: agent.position for aid, agent in self.agents.items()
                     if board[agent.position] == agent.team}
        _, events = resolve_movements(surviving, actions, self.adjacent.__contains__)
        return {event.agent_id: event.position for event in events
                if event.reason == "success" and board[event.position] == self.agents[event.agent_id].team}

    @cached_property
    def cover_targets(self):
        return {cell: self.occupant[cell] for cell, hits in self.enemy_hits.items()
                if self.occupant.get(cell) in self.allies and len(self.attackers[cell]) >= hits}

    @cached_property
    def successful_covers(self):
        return {cell: victim for cell, victim in self.cover_targets.items()
                if victim not in self.deaths and self.owner_after(cell) == self.team}

    def allied_capture(self, cell):
        event = self.captures.get(cell)
        return event if event is not None and event.new_owner == self.team else None


def breakthrough(tick):
    opportunities = [
        cell for cell, defense in tick.enemy_hits.items()
        if tick.owner[cell] != tick.team and len(tick.attackers[cell]) > defense
    ]
    # With one fewer allied attacker, the existing enemy owner wins the tie.
    success = sum(len(capture.contributors) == tick.enemy_hits[cell] + 1
                  for cell in opportunities if (capture := tick.allied_capture(cell)))
    return Opportunity(success, len(opportunities))


def capture_entry(tick):
    possible = defaultdict(set)
    for mover, position in tick.allies.items():
        for cell in tick.adjacent[position]:
            if tick.owner[cell] != tick.team and len(tick.attackers[cell] - {mover}) > tick.enemy_hits[cell]:
                possible[cell].add(mover)
    successes = set()
    for mover, cell in tick.moves.items():
        capture = tick.allied_capture(cell)
        if capture and mover in possible.get(cell, ()) and len(capture.contributors) == tick.enemy_hits[cell] + 1:
            reverted = tick.replay_moves({capture.contributors[0]: Action.NOOP})
            if reverted.get(mover) != cell:
                successes.add(cell)
    return Opportunity(len(successes), len(possible))


def can_vacate(tick, cell, visited):
    for goal in tick.adjacent[cell]:
        if goal in visited or tick.owner[goal] != tick.team:
            continue
        occupant = tick.occupant.get(goal)
        if occupant is None or (occupant in tick.allies and can_vacate(tick, goal, visited | {goal})):
            return True
    return False


def distinct_routes(options, reserved):
    if not options:
        return True
    return any(distinct_routes(options[1:], reserved | {cell}) for cell in options[0] - reserved)


def yield_moves(tick):
    chains = 0
    yielded = 0
    for follower, start in tick.allies.items():
        for target in tick.adjacent[start]:
            leader = tick.occupant.get(target)
            if leader not in tick.allies or not any(tick.owner[cell] != tick.team for cell in tick.adjacent[target]):
                continue
            if can_vacate(tick, target, {start, target}):
                chains += 1
                if leader in tick.moves and tick.moves.get(follower) == target:
                    yielded += tick.replay_moves({leader: Action.NOOP}).get(follower) != target

    proposals = defaultdict(dict)
    for aid, index in zip(tick.learners, tick.proposals, strict=True):
        action = Action.from_index(index)
        if aid in tick.allies and action.is_move:
            target = action.target(tick.allies[aid])
            if target not in tick.occupant and tick.owner[target] == tick.team:
                proposals[target][aid] = action
    conflicts = 0
    resolved = 0
    for target, agents in proposals.items():
        if len(agents) < 2:
            continue
        options = {
            aid: {cell for cell in tick.adjacent[tick.allies[aid]]
                  if cell not in tick.occupant and tick.owner[cell] == tick.team}
            for aid in agents
        }
        if not any(distinct_routes([cells for other, cells in options.items() if other != aid], {target})
                   for aid in agents):
            continue
        conflicts += 1
        if all(aid in tick.moves for aid in agents) and any(tick.moves[aid] == target for aid in agents):
            replay = tick.replay_moves(agents)
            resolved += not all(aid in replay for aid in agents)
    return Opportunity(yielded, chains), Opportunity(resolved, conflicts)


def split_targets(tick):
    opportunities = 0
    successes = 0
    for one, two in combinations(tick.weak_targets, 2):
        first, second = tick.weak_targets[one], tick.weak_targets[two]
        common = first & second
        if not common or len(first | second) < 2:
            continue
        opportunities += 1
        a, b = tick.ally_attacks.get(one), tick.ally_attacks.get(two)
        if a == b or a not in first or b not in second:
            continue
        # Redirect the sole attacker on one tile to the other's shared target.
        captures_a, captures_b = tick.allied_capture(a), tick.allied_capture(b)
        if captures_a and captures_b and (
            (a in common and len(captures_b.contributors) == 1)
            or (b in common and len(captures_a.contributors) == 1)
        ):
            successes += 1
    return Opportunity(successes, opportunities)


def near(cell, anchor):
    return abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]) <= 1


def rotation(tick, previous_attacks):
    opportunities = 0
    successes = 0
    for aid, anchor in previous_attacks.items():
        if aid not in tick.allies or aid in tick.ready:
            continue
        # Attack an enemy tile on the same front, or cover a threatened ally there.
        front = {cell for cell in (anchor, *tick.adjacent[anchor])
                 if tick.owner[cell] != tick.team or cell in tick.cover_targets}
        successors = {cell for cell in front if tick.attackers[cell] - {aid}}
        options = {cell for cell in tick.adjacent[tick.allies[aid]]
                   if tick.owner[cell] == tick.team and cell not in tick.occupant
                   and any(enemy in front and tick.owner[enemy] != tick.team for enemy in tick.adjacent[cell])}
        if successors and options:
            opportunities += 1
            moved = tick.moves.get(aid)
            if moved in options and any(
                target in successors and (tick.allied_capture(target) or target in tick.successful_covers)
                for other, target in tick.ally_attacks.items() if other != aid
            ) and any(tick.owner_after(cell) != tick.team and near(cell, anchor)
                      for cell in tick.adjacent[moved]):
                successes += 1
    return Opportunity(successes, opportunities)


def role_assignments(tick):
    possible, realized = set(), set()
    for target, victim in tick.cover_targets.items():
        if tick.enemy_hits[target] != 1:
            continue
        for defender in tick.attackers[target]:
            expanders = [aid for aid in tick.learners if aid != defender]
            first, second = [tick.weak_targets.get(aid, set()) for aid in expanders]
            if not first or not second or len(first | second) < 2:
                continue
            assignment = (defender, target, victim)
            possible.add(assignment)
            a, b = (tick.ally_attacks.get(aid) for aid in expanders)
            if (target in tick.successful_covers and tick.ally_attacks.get(defender) == target
                    and a in first and b in second and a != b
                    and tick.allied_capture(a) and tick.allied_capture(b)):
                realized.add(assignment)
    return possible, realized


def switched(before, after):
    return any(a[0] != b[0] and a[1:] != b[1:] for a in before for b in after)


class SceneTracker:
    def __init__(self):
        self.counts = Counter()
        self.previous_attacks = {}
        self.role_history = deque(maxlen=ROLE_WINDOW)

    def record(self, name, outcome):
        self.counts[(name, "count")] += outcome.completed
        self.counts[(name, "opportunities")] += outcome.total

    def update(self, tick):
        if not tick.allies:
            self.previous_attacks = {}
            self.role_history.append((set(), set()))
            return
        self.record("breakthrough", breakthrough(tick))
        self.record("cover", Opportunity(len(tick.successful_covers), len(tick.cover_targets)))
        self.record("capture_entry", capture_entry(tick))
        chain, conflict = yield_moves(tick)
        self.record("yield_move", Opportunity(chain.completed + conflict.completed, chain.total + conflict.total))
        self.record("chain_move", chain)
        self.record("conflicting_proposals", conflict)
        self.record("split_targets", split_targets(tick))
        self.record("cooldown_rotation", rotation(tick, self.previous_attacks))
        self.previous_attacks = {
            aid: target for aid, target in tick.ally_attacks.items()
            if aid not in tick.deaths and (tick.owner[target] != tick.team or tick.enemy_hits[target])
        }
        possible, realized = role_assignments(tick)
        can_switch = any(switched(old_possible, possible) for old_possible, _ in self.role_history)
        did_switch = any(switched(old_realized, realized) for _, old_realized in self.role_history)
        self.record("role_switch", Opportunity(int(did_switch), int(can_switch)))
        self.record("role_split", Opportunity(int(bool(realized)), int(bool(possible))))
        self.role_history.append((possible, realized))


def summarize_scenes(counts):
    return {
        name: Opportunity(counts[(name, "count")], counts[(name, "opportunities")]).as_metric()
        for name in (*SCENES, "chain_move", "conflicting_proposals", "role_split")
    }
