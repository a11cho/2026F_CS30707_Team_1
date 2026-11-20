from .models import AGENT_IDS


def calculate_rewards(experiment, areas, events, terminated):
    """전이 후 면적과 점령 이벤트로 ID별 보상, 점령 참여 횟수, 보상 항을 반환한다.

    점령에 참여한 공격자마다 점령 횟수를 1 올리고, individual에서는 각자 capture_weight를
    지급한다. 팀 시그널은 (면적, 격차, 종료) 마스크로 선택한다.
    면적과 격차는 각 값에 가중치를 곱해 팀원마다 지급한다. 종료 항은 종료 때만 준다.
    사망 대기 중인 팀원도 포함하며, 게임 상태나 난수 상태를 변경하지 않는다.
    """
    credits = dict.fromkeys(AGENT_IDS, 0.0)
    for event in events:
        if event.kind == "capture":
            for aid in event.contributors:
                credits[aid] += 1
    outcomes = (int(areas[0] > areas[1]) - int(areas[0] < areas[1]),
                int(areas[1] > areas[0]) - int(areas[1] < areas[0]))
    rewards, components = {}, {}
    area_on, gap_on, terminal_on = experiment.resolved_reward_mask
    for index, aid in enumerate(AGENT_IDS):
        team = index // 3
        capture = experiment.capture_weight if experiment.reward_mode == "individual" and credits[aid] else 0.0
        area = experiment.area_weight * areas[team] if area_on else 0.0
        gap = experiment.gap_weight * (areas[team] - areas[1-team]) if gap_on else 0.0
        outcome_reward = {1: experiment.win_reward, -1: experiment.loss_reward, 0: experiment.draw_reward}[outcomes[team]]
        terminal = float(outcome_reward if terminated and terminal_on else 0)
        components[aid] = {"capture": float(capture), "area": float(area),
                           "gap": float(gap), "terminal": terminal}
        rewards[aid] = float(sum(components[aid].values()))
    return rewards, credits, components

