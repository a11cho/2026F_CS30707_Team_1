from collections import Counter


class EpisodeMetrics:
    def __init__(self):
        """한 에피소드의 사건 수·누적 보상·누적 점령 기여도와 마지막 면적을 초기화한다."""
        self.counts = Counter()
        self.returns = Counter()
        self.credits = Counter()
        self.last_area = None

    def update(self, result):
        """StepResult 한 개의 보상과 사건을 집계기에 누적한다.
        이동 실패는 이유별로 세고 공동 점령도 구별한다. 게임 상태와 보상 계산식에는 영향을 주지 않는다.
        """
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
        """현재까지의 집계값·마지막 면적·면적상 우세 팀을 일반 사전으로 반환한다.
        아직 결과가 없거나 동률이면 winner는 None이다. 종료 전 호출의 winner는 최종 승자가 아니다.
        """
        area = self.last_area
        winner = None if area is None or area[0] == area[1] else ("red" if area[0] > area[1] else "blue")
        return {"counts": dict(self.counts), "returns": dict(self.returns), "credits": dict(self.credits),
                "areas": area, "winner": winner}

