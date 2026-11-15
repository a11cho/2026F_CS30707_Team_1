from .models import AGENT_IDS


def calculate_rewards(experiment, before, after, events, terminated):
    """전후 면적과 점령 이벤트로 (ID별 보상, ID별 점령 기여도, ID별 보상 항) 사전을 반환한다.

    공동 점령의 기여도는 해당 타일을 획득한 공격자 수로 균등 분할한다.
    individual은 본인 기여도, team은 선택된 면적 신호의 1/3을 기본 보상으로 사용한다.
    개인 점령/팀 면적 항과 종료 승패 항은 명시 설정에 따라 선택하며 자동 혼합하지 않는다.
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
    for index, aid in enumerate(AGENT_IDS):
        team = index // 3
        if experiment.reward_mode == "individual":
            base = experiment.capture_weight * credits[aid]
        else:
            signals = {"net": after[team] - before[team], "gross": gains[team],
                       "area": after[team], "terminal": 0}
            base = experiment.area_weight * signals[experiment.team_signal] / 3
        terminal_enabled = experiment.include_terminal_reward or (
            experiment.reward_mode == "team" and experiment.team_signal == "terminal")
        outcome_reward = {1: experiment.win_reward, -1: experiment.loss_reward, 0: experiment.draw_reward}[outcomes[team]]
        terminal = float(outcome_reward if terminated and terminal_enabled else 0)
        components[aid] = {"capture": float(base) if experiment.reward_mode == "individual" else 0.0,
                           "team": float(base) if experiment.reward_mode == "team" else 0.0,
                           "terminal": terminal}
        rewards[aid] = float(base + terminal)
    return rewards, credits, components

