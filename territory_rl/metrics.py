from collections import Counter
from .models import Action


class EpisodeMetrics:
    def __init__(self):
        """한 에피소드의 사건 수·누적 보상·누적 점령 기여도와 마지막 면적을 초기화한다."""
        self.counts = Counter()
        self.returns = Counter()
        self.credits = Counter()
        self.last_area = None
        self.area_sum = [0, 0]
        self.pending_returns = {}
        self.frontier_returns = []
        self.termination_reasons = ()

    def update(self, result, snapshot=None):
        """StepResult 한 개의 보상과 사건을 집계기에 누적한다.
        이동 실패는 이유별로 세고 공동 점령도 구별한다. 게임 상태와 보상 계산식에는 영향을 주지 않는다.
        """
        self.counts["ticks"] += 1
        self.returns.update(result.rewards)
        self.credits.update(result.credits)
        self.last_area = result.area_after
        self.area_sum = [x+y for x, y in zip(self.area_sum, result.area_after)]
        self.termination_reasons = result.termination_reasons
        for e in result.events:
            if e.kind == "death" and e.agent_id in self.pending_returns:
                self._finish_return(e.agent_id, result.tick, "died_before_frontier")
            if e.kind == "respawn":
                self.pending_returns[e.agent_id] = result.tick
            self.counts[e.kind] += 1
            if e.kind == "move" and e.reason != "success":
                self.counts["move_failed:" + e.reason] += 1
            if e.kind == "capture" and len(e.contributors) > 1:
                self.counts["shared_captures"] += 1

        if snapshot is not None:
            for a in snapshot.agents:
                if a.agent_id not in self.pending_returns or not a.alive:
                    continue
                for move in (v for v in Action if v.is_move):
                    p = move.target(a.position)
                    if (0 <= p[0] < snapshot.owner.shape[0] and 0 <= p[1] < snapshot.owner.shape[1]
                            and snapshot.owner[p] != a.team):
                        self._finish_return(a.agent_id, result.tick, "reached")
                        break
        if result.terminated:
            for aid in list(self.pending_returns):
                self._finish_return(aid, result.tick, "episode_ended")

    def _finish_return(self, aid, tick, status):
        """경계 복귀 시간을 기록한다. 미도달은 0이 아닌 None과 사유로 구분한다."""
        started = self.pending_returns.pop(aid)
        self.frontier_returns.append({"agent_id": aid, "respawn_tick": started,
                                      "ticks": tick-started if status == "reached" else None,
                                      "status": status})

    def summary(self):
        """현재까지의 집계값·마지막 면적·면적상 우세 팀을 일반 사전으로 반환한다.
        아직 결과가 없거나 동률이면 winner는 None이다. 종료 전 호출의 winner는 최종 승자가 아니다.
        """
        area = self.last_area
        winner = None if area is None or area[0] == area[1] else ("red" if area[0] > area[1] else "blue")
        return {"counts": dict(self.counts), "returns": dict(self.returns), "credits": dict(self.credits),
                "areas": area, "winner": winner, "area_sum": self.area_sum,
                "termination_reasons": self.termination_reasons, "frontier_returns": list(self.frontier_returns)}

