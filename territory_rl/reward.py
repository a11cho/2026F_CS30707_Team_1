from .models import AGENT_IDS


def calculate_rewards(experiment, before, after, events, terminated):
    """전후 면적과 점령 이벤트로 (ID별 보상, ID별 점령 기여도, ID별 보상 항) 사전을 반환한다.

    공동 점령의 기여도는 해당 타일을 획득한 공격자 수로 균등 분할한다.
    individual은 본인 점령 기여를 지급한다. 공통 시그널은 (면적, 격차, 종료) 마스크로
    선택한다. 면적/격차는 전이 후 현재 값의 1/3을 팀원에게 주고 종료 항은 종료 때만 준다.
    명시 마스크가 있으면 이전 team_signal/include_terminal_reward 옵션은 덮어쓴다.
    사망 대기 중인 팀원도 포함하며, 게임 상태나 난수 상태를 변경하지 않는다.
    """
    credits = dict.fromkeys(AGENT_IDS, 0.0)
    gains = [0, 0]
    for event in events:
        if event.kind == "capture":
            gains[event.new_owner] += 1
            for aid in event.contributors:
                credits[aid] += 1 / len(event.contributors)
    outcomes = (int(after[0] > after[1]) - int(after[0] < after[1]),
                int(after[1] > after[0]) - int(after[1] < after[0]))
    rewards, components = {}, {}
    area_on, gap_on, terminal_on = experiment.resolved_reward_mask
    for index, aid in enumerate(AGENT_IDS):
        team = index // 3
        capture = experiment.capture_weight * credits[aid] if experiment.reward_mode == "individual" else 0.0
        area = experiment.area_weight * after[team] / 3 if area_on else 0.0
        gap = experiment.gap_weight * (after[team] - after[1-team]) / 3 if gap_on else 0.0
        legacy_team = 0.0
        if experiment.reward_signal_mask is None and experiment.reward_mode == "team":
            if experiment.team_signal == "net":
                legacy_team = experiment.area_weight * (after[team] - before[team]) / 3
            elif experiment.team_signal == "gross":
                legacy_team = experiment.area_weight * gains[team] / 3
        outcome_reward = {1: experiment.win_reward, -1: experiment.loss_reward, 0: experiment.draw_reward}[outcomes[team]]
        terminal = float(outcome_reward if terminated and terminal_on else 0)
        components[aid] = {"capture": float(capture), "area": float(area), "gap": float(gap),
                           "team": float(legacy_team), "terminal": terminal}
        rewards[aid] = float(sum(components[aid].values()))
    return rewards, credits, components

