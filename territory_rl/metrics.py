from collections import Counter


class EpisodeMetrics:
    def __init__(self):
        self.counts = Counter()
        self.returns = Counter()
        self.credits = Counter()
        self.last_area = None

    def update(self, result):
        self.counts["ticks"] += 1
        self.returns.update(result.rewards)
        self.credits.update(result.credits)
        self.last_area = result.area_after
        for e in result.events:
            self.counts[e.kind] += 1
            if e.kind == "move" and e.reason != "success":
                self.counts["move_failed:" + e.reason] += 1
            if e.kind == "capture" and len(e.contributors) > 1:
                self.counts["shared_captures"] += 1

    def summary(self):
        area = self.last_area
        winner = None if area is None or area[0] == area[1] else ("red" if area[0] > area[1] else "blue")
        return {"counts": dict(self.counts), "returns": dict(self.returns), "credits": dict(self.credits),
                "areas": area, "winner": winner}

