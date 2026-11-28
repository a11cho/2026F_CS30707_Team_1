"""Paired evaluation of search teachers against a fixed opponent pool."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from itertools import product
from time import perf_counter

import numpy as np

from territory_rl.config import GameConfig

from .match import TEAMS, TeamMatch
from .search import SEARCH_BUDGETS, SearchBudget, choose_search_action
from .tactics import tactical_counts
from .teacher import cooperative_joint


BENCHMARK_OPPONENTS = ("search_tactical_0", "search_strong", "cooperative_holdout")
BENCHMARK_SEARCH_BUDGETS = {
    "search_tactical_0": SearchBudget(12, 3, 0.0, 0.2),
    "search_strong": SearchBudget(20, 5, 0.0, 0.4),
}
BENCHMARK_GAME = GameConfig(height=8, width=8, max_ticks=96, respawn_delay=0, initial_placement="frontier")
BENCHMARK_SEEDS = {"calibration": range(661000, 661006), "test": range(660000, 660016)}


def _play(fixture):
    policy, opponent, seed, team = fixture
    reference = (partial(choose_search_action, budget=BENCHMARK_SEARCH_BUDGETS[opponent])
                 if opponent in BENCHMARK_SEARCH_BUDGETS else partial(cooperative_joint, defensive=True))
    match = TeamMatch(BENCHMARK_GAME, seed, team, opponent, force_entry=True, frozen_opponent=reference)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,)))
    seconds = 0.0
    deaths = losses = threats = 0
    while not match.done:
        state = match.state
        start = perf_counter()
        action = choose_search_action(state, rng, SEARCH_BUDGETS[policy], force_entry=True)
        seconds += perf_counter() - start
        match.step(action)
        counts = tactical_counts(state, team, match.env.last_result.events, force_entry=True)
        deaths += counts["enemy_entry_deaths"]
        losses += counts["defensible_losses"]
        threats += counts["defensible_attacks"]
    return {"policy": policy, "opponent": opponent, "seed": seed, "team": team, "ticks": match.ticks,
            "area": match.state.areas[0], "gap": match.gap, "search_seconds": seconds,
            "entry_deaths": deaths, "defensible_losses": losses, "defensible_attacks": threats}


def _summarize(matches):
    games = len(matches)
    wins = sum(row["gap"] > 0 for row in matches)
    draws = sum(row["gap"] == 0 for row in matches)
    ticks = sum(row["ticks"] for row in matches)
    return {"games": games, "wins": wins, "draws": draws, "losses": games - wins - draws,
            "score": (wins + draws / 2) / games, "mean_area": sum(row["area"] for row in matches) / games,
            "mean_search_ms": 1000 * sum(row["search_seconds"] for row in matches) / ticks,
            **{key: sum(row[key] for row in matches) for key in
               ("entry_deaths", "defensible_losses", "defensible_attacks")}}


def benchmark(policies, split="test", workers=1):
    policies = tuple(policies)
    if not policies or len(policies) != len(set(policies)) or any(p not in SEARCH_BUDGETS for p in policies):
        raise ValueError("choose distinct search policies")
    if workers < 1:
        raise ValueError("workers must be positive")
    seeds = BENCHMARK_SEEDS[split]
    fixtures = product(policies, BENCHMARK_OPPONENTS, seeds, TEAMS)
    if workers == 1:
        matches = list(map(_play, fixtures))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            matches = list(pool.map(_play, fixtures))
    by_policy = {}
    for policy in policies:
        rows = [r for r in matches if r["policy"] == policy]
        by_policy[policy] = {"overall": _summarize(rows),
                             "by_opponent": {o: _summarize([r for r in rows if r["opponent"] == o])
                                             for o in BENCHMARK_OPPONENTS}}
    return {"split": split, "seeds": list(seeds), "teams": TEAMS, "game": asdict(BENCHMARK_GAME),
            "force_entry": True, "opponents": BENCHMARK_OPPONENTS,
            "opponent_search_budgets": {profile: asdict(budget) for profile, budget in BENCHMARK_SEARCH_BUDGETS.items()},
            "policies": {policy: asdict(SEARCH_BUDGETS[policy]) for policy in policies},
            "workers": workers, "results": by_policy, "matches": matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", choices=tuple(SEARCH_BUDGETS), default=["search_strong"])
    parser.add_argument("--split", choices=tuple(BENCHMARK_SEEDS), default="test")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or len(args.policies) != len(set(args.policies)):
        parser.error("use distinct policies and at least one worker")
    print(json.dumps(benchmark(args.policies, args.split, args.workers), allow_nan=False))


if __name__ == "__main__":
    main()
