from collections import Counter, defaultdict
from typing import NamedTuple

from territory_rl.models import Action


class Opportunity(NamedTuple):
    completed: int
    total: int

    def as_metric(self):
        return {
            "count": self.completed,
            "opportunities": self.total,
            "rate": self.completed / self.total if self.total else None,
        }


def cooperation_counts(before, events, learners, proposals):
    """Count same-tick opportunities and outcomes from observations and rule events."""
    counts = Counter()
    positions = [tuple(position) for position in before["agent_positions"]]
    occupied = {position for position, alive in zip(positions, before["agent_alive"], strict=True) if alive}
    available = {
        position
        for position, ready in zip(positions[:3], before["agent_attack_ready"][:3], strict=True)
        if ready
    }
    ally_index = int(learners[0].startswith("blue"))
    ally_moves = [event for event in events if event.agent_id in learners and event.kind == "move"
                 and event.reason != "killed_before_move"]
    counts["movement_attempts"] = len(ally_moves)
    counts["successful_moves"] = sum(event.reason == "success" for event in ally_moves)
    contenders = Counter(event.position for event in ally_moves if event.reason == "contested")
    counts["allied_collision_moves"] = sum(size for size in contenders.values() if size >= 2)

    intended = defaultdict(set)
    for agent, position, index in zip(learners, positions[:3], proposals, strict=True):
        action = Action.from_index(index)
        if action.is_move:
            target = action.target(position)
            if target not in occupied and before["owner"][target] == 1:
                intended[target].add(agent)
    resolved_moves = {event.position: event.agent_id for event in ally_moves if event.reason == "success"}
    for target, agents in intended.items():
        if len(agents) >= 2:
            counts["proposal_conflicts"] += 1
            counts["resolved_proposal_conflicts"] += resolved_moves.get(target) in agents

    threatened = {pos for pos, alive in zip(positions[:3], before["agent_alive"][:3], strict=True) if alive}
    attacks = {event.position: event.counts for event in events if event.kind == "attack"}
    for target, hits in attacks.items():
        enemy_hits = hits[1 - ally_index]
        if not enemy_hits or target not in threatened:
            continue
        row, col = target
        adjacent = {(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)}
        if len(available & adjacent) >= enemy_hits:
            counts["cover_opportunities"] += 1
            # Removing all allied counterattacks would allow the enemy to take the occupied tile.
            counts["protected_tiles"] += hits[ally_index] >= enemy_hits

    for event in events:
        if event.kind == "capture" and event.new_owner == ally_index:
            counts["team_captures"] += 1
            ally_hits = event.counts[ally_index]
            # An added attacker makes the difference if removing one hit creates a tie.
            counts["necessary_joint_captures"] += ally_hits >= 2 and ally_hits == event.counts[1 - ally_index] + 1
    return counts


def summarize_cooperation(counts):
    return {
        "movement_success": Opportunity(counts["successful_moves"], counts["movement_attempts"]).as_metric(),
        "allied_collision": Opportunity(counts["allied_collision_moves"], counts["movement_attempts"]).as_metric(),
        "proposal_conflict_resolution": Opportunity(
            counts["resolved_proposal_conflicts"], counts["proposal_conflicts"]
        ).as_metric(),
        "cover_protection": Opportunity(counts["protected_tiles"], counts["cover_opportunities"]).as_metric(),
        "necessary_joint_captures": counts["necessary_joint_captures"],
        "team_captures": counts["team_captures"],
    }
