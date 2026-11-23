"""학습 보상 외에 조건 간 비교에 필요한 게임 지표를 계산합니다."""

from statistics import mean


class GameMetrics:
    def __init__(self, agents, team_index):
        self.agents = set(agents)
        self.team_index = team_index
        self.ticks = 0
        self.area_sum = 0
        self.episode_return = 0
        self.deaths = 0
        self.captures = 0
        self.joint_captures = 0
        self.team_return = 0
        self.capture_return = 0

    def record(self, result, rewards, components):
        self.ticks += 1
        self.area_sum += result.area_after[self.team_index] / sum(result.area_after)
        self.episode_return += mean(float(rewards[a]) for a in sorted(self.agents))
        self.team_return += mean(sum(value for key,value in components[a].items() if key!='capture')
                                 for a in sorted(self.agents))
        self.capture_return += mean(components[a]['capture'] for a in sorted(self.agents))
        for event in result.events:
            if event.kind == "death" and event.agent_id in self.agents:
                self.deaths += 1
            if event.kind == "capture" and self.agents.intersection(event.contributors):
                self.captures += 1
                self.joint_captures += int(len(self.agents.intersection(event.contributors)) > 1)

    def finish(self, result):
        area = result.area_after[self.team_index]
        other = sum(result.area_after) - area
        score = 1.0 if area > other else 0.0 if area < other else 0.5
        return {
            "ticks": self.ticks, "score": score,
            "final_area_share": area / (area + other),
            "mean_area_share": self.area_sum / self.ticks,
            "mean_agent_return": self.episode_return,
            "team_return": self.team_return,
            "capture_return": self.capture_return,
            "deaths": self.deaths, "captures": self.captures,
            "joint_captures": self.joint_captures,
            "termination_reasons": "|".join(result.termination_reasons),
        }


def summarize_games(rows):
    fields = ("score", "ticks", "final_area_share", "mean_area_share",
              "mean_agent_return", "team_return", "capture_return",
              "deaths", "captures", "joint_captures")
    return {
        "games": len(rows),
        "wins": sum(row["score"] == 1 for row in rows),
        "draws": sum(row["score"] == 0.5 for row in rows),
        "losses": sum(row["score"] == 0 for row in rows),
        **{key: mean(row[key] for row in rows) for key in fields},
    }
