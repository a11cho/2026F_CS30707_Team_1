from .models import AGENT_IDS


def calculate_rewards(experiment, before, after, events, terminated):
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

