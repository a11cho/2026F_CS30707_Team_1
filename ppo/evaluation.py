from collections import Counter
from dataclasses import dataclass, field

from .cooperation import cooperation_counts, summarize_cooperation
from .cooperation_scenes import SceneTick, SceneTracker, summarize_scenes
from .rollout import FeedbackMatch, collect_tick


@dataclass
class MatchMetrics:
    area_gap: float = 0.0
    ticks: int = 0
    deaths: int = 0
    contested_moves: int = 0
    proposal_changes: int = 0
    listening_sum: float = 0.0
    survivors: int = 3
    respawns: int = 0
    cooperation: Counter = field(default_factory=Counter)
    correct_predictions: int = 0
    predictable_actions: int = 0

    def update(self, match, proposal_changes, listening):
        result = match.env.last_result
        own, other = result.area_after if match.team == "red" else result.area_after[::-1]
        self.area_gap = (own - other) / (own + other)
        self.ticks += 1
        self.proposal_changes += proposal_changes
        self.listening_sum += listening
        for event in result.events:
            if event.agent_id in match.learners:
                self.deaths += event.kind == "death"
                self.contested_moves += event.kind == "move" and event.reason == "contested"
                self.respawns += event.kind == "respawn"
        self.survivors = int(match.observations[match.learners[0]]["agent_alive"][:3].sum())


def summarize(games, measure_cooperation=False, measure_prediction=False, scene_trackers=None):
    decisions = 3 * sum(game.ticks for game in games)
    wins = sum(game.area_gap > 0.0 for game in games)
    draws = sum(game.area_gap == 0.0 for game in games)
    summary = {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "game_score": (wins + 0.5 * draws) / len(games),
        "mean_area_gap": sum(game.area_gap for game in games) / len(games),
        "deaths": sum(game.deaths for game in games),
        "contested_moves": sum(game.contested_moves for game in games),
        "proposal_change_rate": sum(game.proposal_changes for game in games) / decisions,
        "mean_listening_tv": sum(game.listening_sum for game in games) / decisions,
    }
    if measure_cooperation:
        counts = Counter()
        for game in games:
            counts.update(game.cooperation)
        summary.update(
            mean_survivors=sum(game.survivors for game in games) / len(games),
            zero_survivor_games=sum(game.survivors == 0 for game in games),
            respawns=sum(game.respawns for game in games),
            cooperation=summarize_cooperation(counts),
        )
    if measure_prediction:
        correct = sum(game.correct_predictions for game in games)
        choices = sum(game.predictable_actions for game in games)
        summary["sender_prediction"] = {
            "correct": correct, "decisions": choices, "accuracy": correct / choices if choices else None,
        }
    if scene_trackers is not None:
        scenes = Counter()
        for tracker in scene_trackers:
            scenes.update(tracker.counts)
        summary["cooperation_scenes"] = summarize_scenes(scenes)
    return summary


def evaluate(policy, game, experiment, opponents, seed, pairs, hide_teammates=False, *,
             measure_cooperation=False, sender_predictor=None, measure_cooperation_scenes=False):
    by_opponent = {}
    scenes_by_opponent = {}
    for opponent in opponents:
        metrics = {}
        for offset in range(pairs):
            for team in ("red", "blue"):
                match = FeedbackMatch(game, experiment)
                match.reset(seed + offset, team, opponent)
                metrics[match] = MatchMetrics()
        active = list(metrics)
        trackers = {match: SceneTracker() for match in active} if measure_cooperation_scenes else None
        while active:
            if measure_cooperation:
                before = [match.observations[match.learners[0]] for match in active]
            if trackers is not None:
                scene_states = [match.env.core.snapshot() for match in active]
            batch = collect_tick(policy, active, hide_teammates=hide_teammates, greedy=True,
                                 measure_listening=not hide_teammates)
            changes = (batch.proposal != batch.action).reshape(-1, 3).sum(dim=-1)
            listening = batch.listening.reshape(-1, 3).sum(dim=-1) if batch.listening is not None else None
            for index, match in enumerate(active):
                metrics[match].update(match, int(changes[index]), 0.0 if listening is None else float(listening[index]))
            if measure_cooperation or trackers is not None:
                proposals = batch.proposal.reshape(-1, 3).tolist()
            if measure_cooperation:
                for match, obs, moves in zip(active, before, proposals, strict=True):
                    metrics[match].cooperation.update(
                        cooperation_counts(obs, match.env.last_result.events, match.learners, moves)
                    )
            if trackers is not None:
                for match, state, actions in zip(active, scene_states, proposals, strict=True):
                    trackers[match].update(SceneTick(
                        state, match.env.last_result, int(match.team == "blue"), match.learners, actions,
                        match.env.core.last_decision[-1]["applied"],
                    ))
            if sender_predictor is not None:
                correct, eligible = sender_predictor.correct(batch.proposal, batch.action, batch.action_mask)
                correct = correct.reshape(-1, 3).sum(dim=1).tolist()
                eligible = eligible.reshape(-1, 3).sum(dim=1).tolist()
                for match, matches_action, decisions in zip(active, correct, eligible, strict=True):
                    metrics[match].correct_predictions += matches_action
                    metrics[match].predictable_actions += decisions
            active = [match for match, done in zip(active, batch.terminated.tolist(), strict=True) if not done]
        by_opponent[opponent] = list(metrics.values())
        scenes_by_opponent[opponent] = tuple(trackers.values()) if trackers is not None else None
    all_games = [game for games in by_opponent.values() for game in games]
    all_trackers = ([tracker for group in scenes_by_opponent.values() for tracker in group]
                    if measure_cooperation_scenes else None)
    return {
        "overall": summarize(all_games, measure_cooperation, sender_predictor is not None, all_trackers),
        "by_opponent": {
            opponent: summarize(games, measure_cooperation, sender_predictor is not None, scenes_by_opponent[opponent])
            for opponent, games in by_opponent.items()
        },
    }
