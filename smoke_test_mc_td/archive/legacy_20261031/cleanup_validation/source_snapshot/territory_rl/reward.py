from .models import AGENT_IDS


def calculate_rewards(experiment, before, after, events, terminated):
    """전후 면적과 점령 이벤트로 (ID별 보상, ID별 점령 기여도) 사전을 반환한다.

    공동 점령의 기여도는 해당 타일을 획득한 공격자 수로 균등 분할한다.
    individual은 본인 기여도, team은 선택된 면적 신호의 1/3을 기본 보상으로 사용한다.
    종료 시에만 승/무/패의 +1/0/-1에 terminal_bonus를 곱해 각 팀원에게 더한다.
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
    rewards = {}
    for index, aid in enumerate(AGENT_IDS):
        team = index // 3
        if experiment.reward_mode == "individual":
            base = credits[aid]
        else:
            signals = {"net": after[team] - before[team], "gross": gains[team],
                       "area": after[team], "terminal": 0}
            base = signals[experiment.team_signal] / 3
        rewards[aid] = float(base + (experiment.terminal_bonus * outcomes[team] if terminated else 0))
    return rewards, credits

