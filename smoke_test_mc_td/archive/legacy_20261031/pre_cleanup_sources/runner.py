"""여섯 행동을 모두 모은 뒤 한 틱을 진행합니다. GUI나 실제 시간 대기는 없습니다.

기존 PettingZoo Parallel 어댑터를 사용하며 정책에 디버그 이벤트를 주지 않습니다.
"""
from dataclasses import asdict
from dataclasses import replace
import time
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
    """완결된 경기를 실행합니다. 모델 6개면 양 팀 모두 행동하고 학습합니다.

    team은 모델 첫 묶음이 맡는 진영입니다. 6개 모델도 진영 교대 시 묶음별로
    이동합니다. 모델 1/3개는 기존 고정 상대 모드를 유지합니다.
    """
    if len(models) not in (1, 3, 6) or team not in (0, 1):
        raise ValueError("모델 수는 1/3/6이고 team은 0/1이어야 합니다.")
    observations, _ = env.reset(seed=seed)
    learner_ids = ((AGENT_IDS[team*3:] + AGENT_IDS[:team*3]) if len(models) == 6
                   else AGENT_IDS[team*3:team*3+len(models)])
    assigned = dict(zip(learner_ids, models))
    rngs = {aid: np.random.default_rng(derived_seed(policy_seed, 1, i))
            for i, aid in enumerate(learner_ids)}
    bots = {aid: controller(opponent if i//3 != team else "expand", derived_seed(policy_seed, 2, i))
            for i, aid in enumerate(AGENT_IDS) if aid not in assigned}
    trajectories = {aid: [] for aid in assigned}
    counters = dict(rejected=0, learner_rejected=0, deaths=0, captures=0, moves_failed=0,
                    dead_samples=0, learner_samples=0, known_samples=0)
    returns = {aid: 0.0 for aid in AGENT_IDS}

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
        for aid in AGENT_IDS:
            returns[aid] += rewards[aid]
        # SARSA에서 선택한 다음 행동은 실제 다음 전이에도 그대로 사용합니다.
        # 독립 테이블의 값을 갱신하기 전에 모든 다음 행동을 선택합니다.
        next_keys, next_actions = ({}, {}) if result.terminated else choose(next_obs)
        for aid, model in assigned.items():
            reward = result.rewards[aid]
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
    focus_ids = [aid for aid in assigned if AGENT_IDS.index(aid)//3 == team]
    # net 보상은 양 팀을 평균하면 상쇄됩니다. 기존 mean_return은 첫 묶음 기준입니다.
    return dict(team=team, ticks=snapshot.tick, score=1.0 if area > other_area else 0.0 if area < other_area else 0.5,
                win=int(area > other_area), draw=int(area == other_area),
                area_fraction=area/sum(snapshot.areas), mean_return=float(np.mean([returns[a] for a in focus_ids])),
                red_area=snapshot.areas[0],blue_area=snapshot.areas[1],
                red_score=1.0 if snapshot.areas[0]>snapshot.areas[1] else 0.0 if snapshot.areas[0]<snapshot.areas[1] else 0.5,
                blue_score=1.0 if snapshot.areas[1]>snapshot.areas[0] else 0.0 if snapshot.areas[1]<snapshot.areas[0] else 0.5,
                red_mean_return=float(np.mean([returns[a] for a in AGENT_IDS[:3]])),
                blue_mean_return=float(np.mean([returns[a] for a in AGENT_IDS[3:]])),
                **{f"role_{i}_return": returns[aid] for i, aid in enumerate(learner_ids)},
                **counters)


def train(config, experiment, algorithm, seed, progress=None):
    """독립 Q-table을 만들고 진영을 교대하며 지정한 에피소드 수만큼 학습합니다."""
    models = [TabularControl(algorithm, config.alpha, config.gamma) for _ in range(config.learner_count)]
    env = make_environment(config, experiment)
    rows = []
    diagnostics = []
    models[0].training_diagnostics = diagnostics
    wall_start,cpu_start = time.perf_counter(),time.process_time()
    probes = None
    previous_probe = None
    previous_updates = 0
    previous_errors = previous_changes = 0.0

    def diagnose(completed):
        nonlocal previous_probe,previous_updates,previous_errors,previous_changes
        # 고정 validation bank의 작은 부분집합으로만 중간 평가합니다. test는 보지 않습니다.
        eval_start,eval_cpu = time.perf_counter(),time.process_time()
        held = evaluate(replace(config,validation_pairs=config.diagnostic_pairs),experiment,models,"validation")
        score = summarize(held)
        eval_wall,eval_cpu = time.perf_counter()-eval_start,time.process_time()-eval_cpu
        updates = sum(m.updates for m in models)
        errors,changes = sum(m.abs_error_sum for m in models),sum(m.abs_change_sum for m in models)
        count = max(1,updates-previous_updates)
        probe = np.concatenate([np.concatenate([m.values(k) for k in keys]) for m,keys in zip(models,probes)]) if probes else np.zeros(1)
        delta = np.abs(probe-(previous_probe if previous_probe is not None else np.zeros_like(probe)))
        role_visits = [list(m.pair_visits.values()) for m in models]
        sample_count = sum(sum(values) for values in role_visits)
        pair_count = sum(len(values) for values in role_visits)
        diagnostic = dict(episode=completed,validation_score=score["score"],validation_area=score["area_fraction"],
                          validation_known_fraction=score["known_state_fraction"],validation_games=len(held),
                          updates=updates,q_states=sum(len(m.q) for m in models),
                          state_action_pairs=pair_count,mean_pair_visits=sample_count/max(1,pair_count),
                          pairs_visited_at_least_twice=sum(v>=2 for vs in role_visits for v in vs),
                          pairs_visited_at_least_five=sum(v>=5 for vs in role_visits for v in vs),
                          mean_abs_target_error=errors/count if completed==0 else (errors-previous_errors)/count,
                          mean_abs_q_update=changes/count if completed==0 else (changes-previous_changes)/count,
                          max_abs_q_update=max(m.window_max_change for m in models),
                          probe_mean_abs_q_change=float(delta.mean()),probe_max_abs_q_change=float(delta.max()),
                          probe_value_count=0 if probes is None else len(probe),
                          elapsed_wall_seconds=time.perf_counter()-wall_start,elapsed_cpu_seconds=time.process_time()-cpu_start,
                          diagnostic_eval_wall_seconds=eval_wall,diagnostic_eval_cpu_seconds=eval_cpu)
        from .profile_runs import memory
        diagnostic.update(memory())
        diagnostics.append(diagnostic)
        if progress:
            progress(diagnostic)
        previous_probe = probe.copy() if probes else None
        previous_updates,previous_errors,previous_changes = updates,errors,changes
        for model in models:
            model.window_max_change = 0.0

    if config.diagnostic_interval:
        diagnose(0)
    for i in range(config.episodes):
        ep_wall,ep_cpu = time.perf_counter(),time.process_time()
        row = episode(env, models, i % 2, derived_seed(seed, 10, i), derived_seed(seed, 11, i),
                      config.epsilon(i), config.train_opponent, True)
        row.update(episode=i+1, epsilon=config.epsilon(i),
                   q_states=sum(len(m.q) for m in models), updates=sum(m.updates for m in models))
        row.update(training_wall_seconds=time.perf_counter()-ep_wall,training_cpu_seconds=time.process_time()-ep_cpu)
        for role, model in enumerate(models):
            row[f"role_{role}_updates"] = model.updates
            row[f"role_{role}_q_states"] = len(model.q)
        rows.append(row)
        if config.diagnostic_interval:
            if probes is None:
                # 첫 에피소드의 key를 고정하고 이후에도 같은 가치 성분의 변화를 비교합니다.
                probes = [tuple(list(m.q)[:128]) for m in models]
                models[0].probe_keys = probes
            if (i+1)%config.diagnostic_interval==0 or i+1==config.episodes:
                diagnose(i+1)
    return models, rows


def evaluate(config, experiment, models, split, opponents=None):
    """고정된 validation/test 경기에서 양쪽 진영을 평가하며 학습하지 않습니다."""
    if split not in ("validation", "test"):
        raise ValueError(split)
    count = config.validation_pairs if split == "validation" else config.test_pairs
    # 모든 알고리즘·조건·학습 seed에서 동일한 별도 평가 seed 집합을 씁니다.
    # 새 양 팀 학습에서는 이전 실험에서 이미 확인한 평가 bank를 재사용하지 않습니다.
    domain = ((40 if split == "validation" else 50) if config.learner_count == 6
              else (20 if split == "validation" else 30))
    env = make_environment(config, experiment)
    rows = []
    # 양 팀이 함께 바뀌므로 평가에서는 각 팀 묶음을 고정 상대와 따로 대결시킵니다.
    cohorts = [models[:3], models[3:]] if len(models) == 6 else [models]
    for cohort, evaluated_models in enumerate(cohorts):
        for opponent in opponents or config.eval_opponents:
            for pair in range(count):
                for team in (0, 1):
                    row = episode(env, evaluated_models, team, derived_seed(20261031, domain, pair),
                                  derived_seed(20261031, domain+1, pair), 0.0, opponent, False)
                    row.update(split=split, opponent=opponent, pair=pair, cohort=cohort)
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
