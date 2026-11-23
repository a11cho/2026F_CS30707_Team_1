"""Two-tick search with simulated respawns and an exhaustive last-tick payoff."""
import numpy as np

from territory_rl.agent import AgentState
from territory_rl.config import ExperimentConfig, GameConfig
from territory_rl.environment import TerritoryEnvironment
from territory_rl.models import AGENT_IDS

from .search import SearchBudget, commands, opponent_view, public_snapshot, reply_scores, search_matrix
from .state import JOINT_ACTIONS, JOINT_ATTACKS, REFLECT_ACTION, TeamState


def _attack_patterns(state, mask):
    """Compress legal joint actions with identical tile-wise attack counts."""
    actions = np.flatnonzero(mask)
    attacks = JOINT_ATTACKS[actions]
    totals = np.zeros((len(actions), state.board.size), dtype=np.uint8)
    rows, cols = state.targets[actions][attacks].T
    np.add.at(totals, (np.nonzero(attacks)[0], rows * state.board.shape[1] + cols), 1)
    _, unique = np.unique(totals, axis=0, return_index=True)
    indices = np.sort(unique)
    return actions[indices], totals[indices]


def terminal_payoffs(state, force_entry):
    """Return all distinct legal last-tick attack payoffs in win/draw/loss units."""
    if state.tick + 1 != state.max_ticks:
        raise ValueError("terminal payoffs require the final tick")
    own_actions, own = _attack_patterns(state, state.choices.mask(force_entry))
    opponent = opponent_view(state)
    _, enemy = _attack_patterns(opponent, opponent.choices.legal)
    enemy = enemy.reshape(-1, *state.board.shape)[:, :, ::-1].reshape(-1, state.board.size)
    contested = (own != 0).any(axis=0) | (enemy != 0).any(axis=0)
    owner = state.board.ravel()
    ours = (owner == 0) | ((owner >= 2) & (owner <= 4))
    mine = own[:, None, contested]
    theirs = enemy[None, :, contested]
    captured = (mine > theirs) | ((mine == theirs) & ours[contested])
    areas = int((ours & ~contested).sum()) + captured.sum(axis=2)
    gap = 2 * areas - state.board.size
    return own_actions, np.sign(gap).astype(np.float64), gap


def _strategy(regret):
    positive = np.maximum(regret, 0)
    total = positive.sum()
    return positive / total if total > 0 else np.full(len(regret), 1.0 / len(regret))


def _solve_matrix(values, gaps):
    """Use exact pure saddles when available and bound mixed play by best responses."""
    guarantees = values.min(axis=1)
    lower = guarantees.max()
    upper = values.max(axis=0).min()
    if lower == upper:
        best = guarantees == lower
        margins = np.where(best, gaps.mean(axis=1), -np.inf)
        tied = margins == margins.max()
        return tied / tied.sum(), float(lower), float(upper)

    own_regret = np.zeros(values.shape[0])
    enemy_regret = np.zeros(values.shape[1])
    own_total = np.zeros_like(own_regret)
    enemy_total = np.zeros_like(enemy_regret)
    iterations = 512
    for _ in range(iterations):
        own, enemy = _strategy(own_regret), _strategy(enemy_regret)
        own_values = values @ enemy
        enemy_values = own @ values
        expectation = own @ own_values
        own_regret += own_values - expectation
        enemy_regret += expectation - enemy_values
        own_total += own
        enemy_total += enemy
    policy = own_total / own_total.sum()
    other = enemy_total / enemy_total.sum()
    return policy, float((policy @ values).min()), float((values @ other).max())


def terminal_plan(state, force_entry):
    actions, values, gaps = terminal_payoffs(state, force_entry)
    probabilities, lower, upper = _solve_matrix(values, gaps)
    return actions, probabilities, lower, upper


def _model_match(state, respawn_delay):
    height, width = state.board.shape
    game = GameConfig(height=height, width=width, max_ticks=state.max_ticks, respawn_delay=respawn_delay)
    env = TerritoryEnvironment(game, ExperimentConfig(observation_window_size=2 * max(height, width) - 1))
    snapshot = public_snapshot(state)
    agents = [
        AgentState(a.agent_id, a.team, a.position if a.alive else (0, 0),
                   last_attack_tick=a.last_attack_tick, alive=a.alive,
                   respawn_at=None if a.alive else state.tick + int(wait))
        for a, wait in zip(snapshot.agents, state.waits, strict=True)
    ]
    return env, {"owner": snapshot.owner, "agents": agents, "tick": state.tick}


def _continuation(env, options, seed, actions, rng, budget, force_entry):
    env.reset(seed=seed, options=options)
    result = env.step(actions)
    if result.terminated:
        return float(np.sign(result.area_after[0] - result.area_after[1]))
    observations = [env.observe("red", i) for i in range(3)]
    state = TeamState.from_observations(observations, env.experiment.tile_encoding, "red")
    if state.tick + 1 == state.max_ticks:
        return terminal_plan(state, force_entry)[2]
    _, _, values = search_matrix(state, rng, budget, state.choices.mask(force_entry))
    return float(reply_scores(values, budget).max())


def choose_lookahead_action(state, rng, budget, force_entry, respawn_delay):
    if state.tick + 1 == state.max_ticks:
        actions, probabilities, _, _ = terminal_plan(state, force_entry)
        return int(rng.choice(actions, p=probabilities))

    mine, replies, values = search_matrix(state, rng, budget, state.choices.mask(force_entry))
    selected = np.argsort(-reply_scores(values, budget), kind="stable")[:budget.lookahead_candidates]
    futures = values[selected].copy()
    env, options = _model_match(state, respawn_delay)
    # Use the same modeled respawn and child search streams for every candidate pair.
    respawn_seed, search_seed = map(int, rng.integers(2**31, size=2))
    child_budget = SearchBudget(4, 2, 0.0, budget.worst_reply_weight)
    enemies = [commands(REFLECT_ACTION[JOINT_ACTIONS[index]], AGENT_IDS[3:]) for index in replies]
    for row, candidate in enumerate(mine[selected]):
        allies = commands(JOINT_ACTIONS[candidate], AGENT_IDS[:3])
        for column, enemy in enumerate(enemies):
            futures[row, column] += 0.8 * _continuation(
                env, options, respawn_seed, allies | enemy, np.random.default_rng(search_seed), child_budget, force_entry
            )
    scores = reply_scores(futures, budget)
    return int(rng.choice(mine[selected][np.isclose(scores, scores.max(), rtol=0, atol=1e-9)]))
