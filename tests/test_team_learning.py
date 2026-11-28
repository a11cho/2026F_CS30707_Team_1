from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from team_policy.curriculum import Curriculum, LESSONS, LESSONS_WITH_INTERMEDIATE_SEARCH
from team_policy.distillation import jensen_shannon, tactical_feedback
from team_policy.experiment import MatchSource, collect, restore_optimizer
from team_policy.fixed_scenes import SPLITS, fixed_cases
from team_policy.group_learning import collect_groups, relative_advantages
from team_policy.match import TeamMatch, entry_scenario
from team_policy.optimization import generalized_advantage
from team_policy.policy import TeamPolicy
from team_policy.search import TACTICAL_EXPLORATION_BUDGETS, TACTICAL_LEAF_BUDGETS, SearchBudget, choose_search_action, opponent_view
from team_policy.state import RULES_AND_COVER, RULES_AND_COVER_V2, RULES_PROMPTS, TeamState, joint_index
from team_policy.tactics import make_exercise, tactical_counts
from team_policy.teacher import TacticalScorer, world_actions
from territory_rl.config import GameConfig
from territory_rl.models import Action


def test_rules_prompt_preserves_board_history_and_action_cue():
    match = TeamMatch(GameConfig(max_ticks=96), 640002, "red", "search_tactical")
    match.step(int(np.flatnonzero(match.view()[2])[0]))
    state, history, _ = match.view()
    plain = state.text_input(history)
    prompted = state.text_input(history, rules_prompt=True)

    assert prompted.replace(RULES_AND_COVER, "", 1) == plain
    assert "A tie keeps its current owner." in prompted
    assert "one enemy attack" in prompted and "threatened ally's starting tile" in prompted
    assert "A tie keeps its current owner." not in plain
    assert prompted.endswith("\nTeam action:")


def test_second_rules_prompt_preserves_existing_state_encoding():
    match = TeamMatch(GameConfig(max_ticks=96), 640006, "blue", "search_tactical")
    state, history, _ = match.view()
    original = state.text_input(history)
    with_rules = state.text_input(history, rules_prompt=True, rules_prompt_version=2)

    assert state.text_input(history, rules_prompt_version=2) == original
    assert with_rules.replace(RULES_AND_COVER_V2, "", 1) == original
    assert "areas draw" in with_rules and "attack the ally's current tile" in with_rules
    assert "two allies can take an enemy tile" in with_rules
    assert state.text_input(history, rules_prompt=True, rules_prompt_version=1) == state.text_input(history, rules_prompt=True)


@pytest.mark.parametrize(("team", "height", "counts", "shares"), [
    ("red", 8, "33/31", "Area shares 51.56% ours, 48.44% enemy. Lead +3.12 percentage points. "),
    ("blue", 6, "25/23", "Area shares 52.08% ours, 47.92% enemy. Lead +4.17 percentage points. "),
])
@pytest.mark.parametrize("version", [3, 4])
def test_rules_prompt_reaches_qwen_tokenizer_with_live_area_shares(monkeypatch, team, height, counts, shares, version):
    game = GameConfig(height=height, max_ticks=96)
    match = TeamMatch(game, 640000, team, "noop", scenario=entry_scenario(game, team))
    match.step(joint_index((Action.RIGHT.index, Action.ATTACK_DOWN.index, Action.ATTACK_RIGHT.index)))
    view = match.view()
    state, history, _ = view
    captured = []

    def tokenize(texts, *, add_special_tokens):
        assert not add_special_tokens
        captured.extend(texts)
        return {"input_ids": [[11, 13]]}

    encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
    monkeypatch.setattr("team_policy.policy.load_qwen", lambda *_: (tokenize, encoder))
    policy = TeamPolicy("bf16", "cpu", rules_prompt=True, rules_prompt_version=version)
    sample, = policy.prepare([view])
    prompt, = captured

    assert sample.tokens == [11, 13]
    assert "Capturing an occupied" in prompt
    assert "respawn" in prompt and "random empty friendly tiles" in prompt
    assert f"Area {counts}. {shares}" in prompt
    assert prompt.replace(RULES_PROMPTS[version], "", 1).replace(shares, "", 1) == state.text_input(history)


def test_compact_rules_preserve_cover_and_concentrated_attack_guidance():
    prompt = RULES_PROMPTS[4]
    assert len(prompt) < len(RULES_PROMPTS[3])
    assert "Cover a threatened ally's starting tile" in prompt
    assert "Match attacks to hold and outnumber them to capture." in prompt
    assert "Pair entry with same-tick capture." in prompt


@pytest.mark.parametrize("team", ["red", "blue"])
def test_explicit_map_prompt_changes_only_current_map_text(monkeypatch, team):
    game = GameConfig(height=6, max_ticks=96)
    match = TeamMatch(game, 640000, team, "noop", scenario=entry_scenario(game, team))
    match.step(joint_index((Action.RIGHT.index, Action.ATTACK_DOWN.index, Action.ATTACK_RIGHT.index)))
    view = match.view()
    state, _, _ = view
    captured = []

    def tokenize(texts, *, add_special_tokens):
        captured.extend(texts)
        return {"input_ids": [[11, 13]]}

    encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
    monkeypatch.setattr("team_policy.policy.load_qwen", lambda *_: (tokenize, encoder))
    policy = TeamPolicy("bf16", "cpu", rules_prompt=True, rules_prompt_version=4)
    plain, = policy.prepare([view])
    policy.explicit_map_prompt = True
    explicit, = policy.prepare([view])

    baseline, coordinate_prompt = captured
    rules, _, tail = baseline.partition(RULES_PROMPTS[4])
    grid, _, state_tail = tail.partition("\nTick ")
    coordinate_map = coordinate_prompt.removeprefix(rules + RULES_PROMPTS[4]).removesuffix("\nTick " + state_tail)
    heading, columns, *rows, positions = coordinate_map.splitlines()
    assert heading == "Current map in team-relative zero-based coordinates"
    assert columns == "Columns 0 1 2 3 4 5 6 7"
    assert len(rows) == 6
    for row, (annotated, original) in enumerate(zip(rows, grid.splitlines(), strict=True)):
        assert annotated.split() == [f"r{row}", *original]
    assert positions.startswith("Agent positions (row,column) ")
    for slot, (row, col) in zip("abcxyz", state.positions, strict=True):
        assert (f"{slot}=({row},{col})" if row >= 0 else f"{slot}=absent") in positions
    assert coordinate_prompt.replace(coordinate_map, grid, 1) == baseline
    np.testing.assert_array_equal(explicit.board, plain.board)
    np.testing.assert_array_equal(explicit.status, plain.status)
    np.testing.assert_array_equal(explicit.mask, plain.mask)


def test_gae_distinguishes_game_termination_from_a_rollout_cut():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.zeros_like(rewards)
    terminal = torch.tensor([[True, False], [False, False]])
    bootstrap = torch.tensor([6.0, 6.0])
    advantage = generalized_advantage(rewards, values, terminal, bootstrap, gamma=0.5, gae_lambda=1.0)
    assert torch.equal(advantage, torch.tensor([[1.0, 3.5], [5.0, 5.0]]))


def test_group_advantages_are_centered_and_constant_groups_have_no_signal():
    values, std = relative_advantages([1.0, 1.0, 1.0, 1.0])
    assert std == 0
    np.testing.assert_array_equal(values, np.zeros(4))
    values, std = relative_advantages([-0.5, 0.5, 0.5, -0.5])
    assert std == pytest.approx(0.5)
    np.testing.assert_array_equal(values, [-1, 1, 1, -1])


@pytest.mark.parametrize("algorithm,kept", [("grpo", 1), ("dapo", 0)])
def test_group_replicas_share_environment_and_opponent_seed(algorithm, kept):
    class HoldPolicy:
        device = torch.device("cpu")

        def prepare(self, views):
            return [SimpleNamespace(mask=view[2]) for view in views]

        def actor(self, samples):
            logits = torch.full((len(samples), 729), -torch.inf)
            logits[:, 0] = 0
            return torch.distributions.Categorical(logits=logits)

    game = GameConfig(max_ticks=3)
    source = MatchSource(game, 7, ("random",), 0, False)
    log = SimpleNamespace(write=lambda *args, **kwargs: None)
    args = SimpleNamespace(group_size=4, envs=4, rollout_steps=3, inference_batch=2, seed=7, algorithm=algorithm)
    rollout, stats = collect_groups(HoldPolicy(), source, args, 13, log, 0)
    assert stats["ticks"] == 12
    assert stats["groups_kept"] == kept
    group, = stats["groups"]
    assert len(set(group["returns"])) == len(set(group["gaps"])) == 1
    assert len(rollout.inputs) == 12 * kept
    assert not rollout.advantages.any()


@pytest.mark.parametrize("team", ["red", "blue"])
def test_tactical_feedback_recommends_only_replayed_improvements(team):
    game = GameConfig()
    match = TeamMatch(game, 7, team, "aggressive", force_entry=False, scenario=entry_scenario(game, team))
    state, _, mask = match.view()
    before = match.env.core.snapshot()
    match.step(0)
    feedback = tactical_feedback(match, before, state, mask, 0, np.random.default_rng(8))
    assert feedback is not None and feedback.preferred
    assert "ally losses" in feedback.text
    scorer = TacticalScorer(before, match.team_number, match.allies)
    enemy = {aid: Action.from_index(match.last_actions[aid]) for aid in match.enemies}
    for candidate in feedback.preferred:
        assert mask[candidate]
        assert scorer(world_actions(match, candidate) | enemy) - scorer(world_actions(match, 0) | enemy) >= 0.01


def test_distillation_backpropagates_through_masked_joint_distribution():
    logits = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    invalid = torch.tensor([[False, True, False]])
    distribution = torch.distributions.Categorical(logits=logits.masked_fill(invalid, -torch.inf))
    divergence = jensen_shannon(distribution.logits, torch.tensor([[0.75, 0.0, 0.25]]))
    divergence.backward()
    assert 0 < divergence.item() < 1
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0


def test_fixed_suite_has_verified_pairs_and_disjoint_initial_states():
    seen = set()
    for split in SPLITS:
        cases = fixed_cases(GameConfig(), split, 3, 610000)
        signatures = {case.initial_hash() for case in cases}
        assert len(signatures) == len(cases) == 7 * 3 * 2
        assert seen.isdisjoint(signatures)
        seen.update(signatures)


def test_terminal_margin_preserves_intermediate_rewards_and_adds_final_gap():
    game = GameConfig(max_ticks=8)
    original = TeamMatch(game, 7, "red", "expand", force_entry=False)
    modified = TeamMatch(game, 7, "red", "expand", force_entry=False, terminal_margin_weight=0.25)
    while not original.done:
        old_reward, old_done = original.step(0)
        reward, done = modified.step(0)
        assert done == old_done
        assert original.gap == modified.gap
        assert reward - old_reward == pytest.approx(0.25 * modified.gap if done else 0.0)


def test_search_can_reconstruct_public_opponent_masks_after_cooldown_and_death():
    game = GameConfig(respawn_delay=3)
    match = TeamMatch(game, 7, "red", "noop", force_entry=False)
    match.step(joint_index((Action.ATTACK_RIGHT.index, 0, 0)))
    reconstructed = opponent_view(match.state)
    engine_view = TeamState.from_observations([match.observations[aid] for aid in match.enemies],
                                              match.env.core.experiment.tile_encoding, "blue")
    for field in ("board", "positions", "ready", "waits", "action_masks"):
        np.testing.assert_array_equal(getattr(reconstructed, field), getattr(engine_view, field))
    assert reconstructed.areas == engine_view.areas
    assert reconstructed.waits[0] > 0 and not reconstructed.ready[3]


@pytest.mark.parametrize("team", ["red", "blue"])
def test_search_joint_decision_respects_certified_entry(team):
    game = GameConfig()
    match = TeamMatch(game, 7, team, "aggressive", scenario=entry_scenario(game, team))
    action = choose_search_action(match.state, np.random.default_rng(26), SearchBudget(12, 3, 0.0, 0.3))
    assert match.view()[2][action]
    match.step(action)
    assert match.entry_successes == 1


def test_curriculum_checks_fixed_intervals_and_advances_one_step():
    course = Curriculum(0, interval=20, target=0.625)
    assert course.opponents() == (("noop",), (1.0,))
    assert not course.check_due()
    course.stage_ticks = 20
    assert course.check_due()
    assert not course.check(0.5)
    assert course.level == 0 and course.next_check == 40
    course.stage_ticks = 40
    assert course.check(0.625)
    assert course.level == 1 and course.stage_ticks == 0 and course.next_check == 20
    assert course.opponents() == (("random", "noop"), (0.8, 0.2))
    assert course.lesson.game(GameConfig()).max_ticks == 48
    final = Curriculum(len(LESSONS) - 1, interval=20, target=0.625)
    assert not final.check(1.0)


def test_fine_search_changes_training_opponents_while_exam_stays_fixed():
    course = Curriculum(4, interval=20, target=0.625, lessons=LESSONS_WITH_INTERMEDIATE_SEARCH, fine_search=True)
    for opponent in ("search_explore_67", "search_explore_64", "search_explore_61", "search_explore_58",
                     "search_intermediate", "search_intermediate"):
        assert course.opponents() == ((opponent, "search_intro"), (0.8, 0.2))
        assert (course.lesson.opponent, course.lesson.max_ticks) == ("search_intermediate", 80)
        course.stage_ticks += 20
        assert course.check_due()
        assert not course.check(0.60)
    assert course.check(0.625)
    assert course.opponents() == (("search_tactical", "search_intermediate"), (0.8, 0.2))


def test_fine_tactical_ramps_training_budget_without_changing_stage_five_exam():
    course = Curriculum(5, interval=20, target=0.625, lessons=LESSONS_WITH_INTERMEDIATE_SEARCH, fine_tactical=True)
    for opponent in ("search_tactical_50", "search_tactical_45", "search_tactical_40", "search_tactical",
                     "search_tactical"):
        assert course.opponents() == ((opponent, "search_intermediate"), (0.8, 0.2))
        assert (course.lesson.opponent, course.lesson.max_ticks) == ("search_tactical", 96)
        course.stage_ticks += 20
        assert course.check_due()
        assert not course.check(0.60)
    assert course.check(0.625)
    assert course.opponents() == (("search_strong", "search_tactical"), (0.8, 0.2))


def test_leaf_curriculum_changes_only_candidate_limit_and_advances_by_score():
    names = tuple(TACTICAL_LEAF_BUDGETS)
    assert [b.candidates * b.replies for b in TACTICAL_LEAF_BUDGETS.values()] == [8, 12, 16, 20, 24]
    assert {(b.replies, b.exploration, b.worst_reply_weight) for b in TACTICAL_LEAF_BUDGETS.values()} == {
        (2, 0.50, 0.10)
    }
    course = Curriculum(5, interval=20, target=0.625, lessons=LESSONS_WITH_INTERMEDIATE_SEARCH, leaf_budget=True)
    course.select_tactical_level(dict.fromkeys(names, 0.0))
    assert course.opponents() == ((names[0],), (1.0,))
    course.select_tactical_level(dict(zip(names, [0.75, 0.50, 0.50, 0.25, 0.125], strict=True)))
    assert course.tactical_opponent == names[1]
    course.adjust_tactical_level(0.625)
    assert course.tactical_opponent == names[2]
    course.adjust_tactical_level(0.25)
    assert course.tactical_opponent == names[2]
    course.adjust_tactical_level(0.1875)
    assert course.tactical_opponent == names[1]
    course.stage_ticks = 20
    assert not course.check(0.625)
    assert (course.lesson.opponent, course.lesson.max_ticks, course.next_check) == ("search_tactical", 96, 40)


def test_tactical_exploration_curriculum_changes_only_random_action_probability():
    budgets = TACTICAL_EXPLORATION_BUDGETS
    assert {(b.candidates, b.replies, b.worst_reply_weight) for b in budgets.values()} == {(4, 2, 0.10)}
    assert [b.exploration for b in budgets.values()] == [n / 100 for n in range(95, 50, -5)]
    course = Curriculum(5, interval=20, target=0.625, lessons=LESSONS_WITH_INTERMEDIATE_SEARCH,
                        tactical_exploration=True)
    assert [budgets[name].exploration for name in course.calibration_opponents] == [.95, .85, .75, .65, .55]
    course.select_tactical_level(dict(zip(course.calibration_opponents, [.875, .75, .50, .25, .125], strict=True)))
    assert course.opponents() == (("search_tactical_random_75",), (1.0,))
    course.adjust_tactical_level(0.625)
    assert course.opponents() == (("search_tactical_random_70",), (1.0,))
    course.adjust_tactical_level(0.0)
    assert course.opponents() == (("search_tactical_random_75",), (1.0,))
    assert course.lesson.opponent == "search_tactical"


def test_tactical_schedule_replays_opponents_independently_of_probe_score():
    course = Curriculum(5, 4096, 0.625, LESSONS_WITH_INTERMEDIATE_SEARCH, tactical_exploration=True,
                        exploration_schedule=(65, 70, 70, 70))
    assert course.opponents() == (("search_tactical_random_65",), (1.0,))
    course.stage_ticks = 4096
    assert course.tactical_opponent == "search_tactical_random_65"
    course.adjust_tactical_level(0.9)
    assert course.opponents() == (("search_tactical_random_70",), (1.0,))
    course.stage_ticks = 8192
    course.adjust_tactical_level(0.0)
    assert course.tactical_opponent == "search_tactical_random_70"
    assert not course.check(1.0)


def test_resume_head_only_optimizer_with_lora_adds_empty_adapter_state():
    head = torch.nn.Parameter(torch.ones(2))
    previous = torch.optim.AdamW([head], lr=0.001)
    head.sum().backward()
    previous.step()
    head_new = torch.nn.Parameter(head.detach().clone())
    adapter = torch.nn.Parameter(torch.ones(3))
    current = torch.optim.AdamW([{"params": [head_new], "lr": 0.003}, {"params": [adapter], "lr": 0.002}])
    payload = {"optimizer": previous.state_dict(), "run": {"freeze_text_backbone": True}}

    assert restore_optimizer(current, payload)
    assert current.state[head_new]["step"].item() == 1
    assert torch.equal(current.state[head_new]["exp_avg"], previous.state[head]["exp_avg"])
    assert adapter not in current.state
    assert [group["lr"] for group in current.param_groups] == [0.003, 0.002]
    (head_new.sum() + adapter.sum()).backward()
    current.step()
    assert [current.state[param]["step"].item() for param in (head_new, adapter)] == [2, 1]


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("kind, error", [("enemy_entry", "enemy_entry_deaths"), ("tile_defense", "defensible_losses")])
def test_tactical_exercises_have_a_legal_action_that_avoids_the_intended_error(team, kind, error):
    game = GameConfig(max_ticks=16)
    exercise = make_exercise(game, kind, 29, team)
    target = exercise.target if team == "red" else (exercise.target[0], game.width - 1 - exercise.target[1])

    def run(attack):
        match = TeamMatch(game, 29, team, "search_intermediate", exercise=exercise)
        state = match.state
        commands = [Action.NOOP.index] * 3
        if attack or kind == "enemy_entry":
            slot = next(i for i, pos in enumerate(state.positions[:3]) if np.abs(pos - target).sum() == 1)
            start = state.positions[slot]
            commands[slot] = next(a.index for a in Action if a != Action.NOOP and a.is_attack == attack
                                  and a.target(start) == target)
        match.step(joint_index(commands))
        return tactical_counts(state, team, match.env.last_result.events, force_entry=True, target=exercise.target)

    assert run(False)[error] == 1
    assert run(True)[error] == 0


def test_search_imitation_labels_preserve_on_policy_actions():
    class SamplePolicy:
        tokenizer = None

        def prepare(self, views):
            return [SimpleNamespace(mask=view[2]) for view in views]

        def __call__(self, inputs):
            masks = torch.from_numpy(np.stack([sample.mask for sample in inputs]))
            return torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -torch.inf), torch.zeros(len(inputs))

        def distribution(self, inputs, *, cached_text=None):
            logits, values = self(inputs)
            return torch.distributions.Categorical(logits=logits), values

    def rollout(search_rng):
        game = GameConfig(max_ticks=16)
        source = MatchSource(game, 7, ("random",), 0, True)
        matches = [source.next() for _ in range(4)]
        args = SimpleNamespace(inference_batch=4, freeze_text_backbone=False)
        log = SimpleNamespace(write=lambda *args, **kwargs: None)
        torch.manual_seed(35)
        return collect(SamplePolicy(), matches, source, 16, args, log, 0, search_rng=search_rng)[0]

    baseline = rollout(None)
    guided = rollout(np.random.default_rng(19))
    torch.testing.assert_close(baseline.actions, guided.actions)
    labels = guided.search_targets.tolist()
    assert any(label >= 0 for label in labels)
    assert all(label < 0 or sample.mask[label] for sample, label in zip(guided.inputs, labels, strict=True))
