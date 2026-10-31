"""여섯 행동을 모두 모은 뒤 한 틱을 진행합니다. GUI나 실제 시간 대기는 없습니다.

기존 PettingZoo Parallel 어댑터를 사용하며 정책에 디버그 이벤트를 주지 않습니다.
"""
from dataclasses import asdict
import numpy as np
from territory_rl import GameConfig
from territory_rl.parallel_env import TerritoryParallelEnv
from territory_rl.controller import FrontierController, NoopController, RandomController
from territory_rl.models import AGENT_IDS
from .encoding import encode
from .learning import TabularControl


def derived_seed(master, domain, index=0):
    """기준 seed에서 용도·인덱스별 재현 가능한 난수 seed를 파생합니다."""
    return int(np.random.SeedSequence([master, domain, index]).generate_state(1)[0])


def make_environment(config, experiment):
    game = GameConfig(config.height, config.width, config.ticks, config.respawn_delay, config.radius)
    return TerritoryParallelEnv(game, experiment)


def controller(style, seed):
    if style == "noop":
        return NoopController()
    if style == "random":
        return RandomController(seed)
    return FrontierController(seed, style)


def episode(env, models, team, seed, policy_seed, epsilon, opponent="random", training=False):
    """완결된 한 경기를 실행하며 training일 때만 Q를 갱신합니다."""
    observations, _ = env.reset(seed=seed)
    learner_ids = AGENT_IDS[team*3:team*3+len(models)]
    assigned = dict(zip(learner_ids, models))
    rngs = {aid: np.random.default_rng(derived_seed(policy_seed, 1, i))
            for i, aid in enumerate(learner_ids)}
    bots = {aid: controller(opponent if i//3 != team else "expand", derived_seed(policy_seed, 2, i))
            for i, aid in enumerate(AGENT_IDS) if aid not in assigned}
    trajectories = {aid: [] for aid in assigned}
    counters = dict(rejected=0, learner_rejected=0, deaths=0, captures=0, moves_failed=0,
                    dead_samples=0, learner_samples=0, known_samples=0)
    returns = {aid: 0.0 for aid in assigned}

    def choose(observations):
        keys = {aid: encode(observations[aid]) for aid in assigned}
        actions = {aid: model.act(keys[aid], observations[aid]["action_mask"], epsilon, rngs[aid], training)
                   for aid, model in assigned.items()}
        actions.update({aid: bot.act(observations[aid]) for aid, bot in bots.items()})
        assert set(actions) == set(AGENT_IDS)
        assert all(observations[a]["action_mask"][actions[a]] for a in AGENT_IDS)
        return keys, actions

    keys, actions = choose(observations)
    while env.agents:
        for aid in assigned:
            counters["learner_samples"] += 1
            counters["dead_samples"] += int(not observations[aid]["self_features"][2])
            counters["known_samples"] += int(keys[aid] in assigned[aid].q)
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        result = env.last_result
        if any(truncs.values()):
            raise RuntimeError("완결된 MC 에피소드가 필요하지만 외부 truncation이 발생했습니다.")
        assert all(terms.values()) == result.terminated
        # SARSA에서 선택한 다음 행동은 실제 다음 전이에도 그대로 사용합니다.
        # 독립 테이블의 값을 갱신하기 전에 모든 다음 행동을 선택합니다.
        next_keys, next_actions = ({}, {}) if result.terminated else choose(next_obs)
        for aid, model in assigned.items():
            reward = result.rewards[aid]
            returns[aid] += reward
            if training:
                if model.algorithm == "mc":
                    trajectories[aid].append((keys[aid], actions[aid], reward))
                else:
                    model.sarsa_step(keys[aid], actions[aid], reward, next_keys.get(aid),
                                     next_actions.get(aid), result.terminated)
        for event in result.events:
            if event.kind == "rejected":
                counters["rejected"] += 1
                counters["learner_rejected"] += int(event.agent_id in assigned)
            if event.kind == "death" and event.agent_id in assigned:
                counters["deaths"] += 1
            if event.kind == "capture" and event.new_owner == team:
                counters["captures"] += 1
            if event.kind == "move" and event.agent_id in assigned and event.reason != "success":
                counters["moves_failed"] += 1
        observations, keys, actions = next_obs, next_keys, next_actions
    if training:
        for aid, model in assigned.items():
            if model.algorithm == "mc":
                model.finish_episode(trajectories[aid])
    snapshot = env.core.snapshot()
    area = snapshot.areas[team]
    other_area = snapshot.areas[1-team]
    return dict(team=team, ticks=snapshot.tick, score=1.0 if area > other_area else 0.0 if area < other_area else 0.5,
                win=int(area > other_area), draw=int(area == other_area),
                area_fraction=area/sum(snapshot.areas), mean_return=float(np.mean(list(returns.values()))),
                **counters)


def train(config, experiment, algorithm, seed):
    """독립 Q-table을 만들고 진영을 교대하며 지정한 에피소드 수만큼 학습합니다."""
    models = [TabularControl(algorithm, config.alpha, config.gamma) for _ in range(config.learner_count)]
    env = make_environment(config, experiment)
    rows = []
    for i in range(config.episodes):
        row = episode(env, models, i % 2, derived_seed(seed, 10, i), derived_seed(seed, 11, i),
                      config.epsilon(i), config.train_opponent, True)
        row.update(episode=i+1, epsilon=config.epsilon(i),
                   q_states=sum(len(m.q) for m in models), updates=sum(m.updates for m in models))
        rows.append(row)
    return models, rows


def evaluate(config, experiment, models, split, opponents=None):
    """고정된 validation/test 경기에서 양쪽 진영을 평가하며 학습하지 않습니다."""
    if split not in ("validation", "test"):
        raise ValueError(split)
    count = config.validation_pairs if split == "validation" else config.test_pairs
    # 모든 알고리즘·조건·학습 seed에서 동일한 별도 평가 seed 집합을 씁니다.
    domain = 20 if split == "validation" else 30
    env = make_environment(config, experiment)
    rows = []
    for opponent in opponents or config.eval_opponents:
        for pair in range(count):
            for team in (0, 1):
                row = episode(env, models, team, derived_seed(20261031, domain, pair),
                              derived_seed(20261031, domain+1, pair), 0.0, opponent, False)
                row.update(split=split, opponent=opponent, pair=pair)
                rows.append(row)
    return rows


def summarize(rows):
    """경기별 기록을 평균 점수·면적·사망 대기·기방문 비율 등으로 집계합니다."""
    fields = ("score", "win", "draw", "area_fraction", "mean_return", "deaths", "captures", "moves_failed")
    result = {name: float(np.mean([row[name] for row in rows])) for name in fields}
    result["games"] = len(rows)
    samples = sum(r["learner_samples"] for r in rows)
    result["dead_fraction"] = sum(r["dead_samples"] for r in rows) / max(1, samples)
    result["known_state_fraction"] = sum(r["known_samples"] for r in rows) / max(1, samples)
    result["learner_rejected"] = sum(r["learner_rejected"] for r in rows)
    return result
