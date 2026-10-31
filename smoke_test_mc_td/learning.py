import gzip
import json
import numpy as np


class TabularControl:
    """관측 key마다 행동 9개의 가치를 저장하는 MC/SARSA 공통 학습자입니다."""
    def __init__(self, algorithm, alpha=0.1, gamma=0.99):
        if algorithm not in ("mc", "sarsa"):
            raise ValueError(algorithm)
        self.algorithm, self.alpha, self.gamma = algorithm, alpha, gamma
        self.q = {}
        self.updates = 0
        self.decisions = 0
        self.known_decisions = 0

    def values(self, key):
        """저장된 Q 배열을 조회하며 미방문 key는 저장하지 않고 0을 반환합니다."""
        return self.q.get(key, np.zeros(9, dtype=float))

    def act(self, key, mask, epsilon, rng, track=False):
        """허용된 행동에서 epsilon-greedy로 선택하고 최대값 동률은 무작위로 풉니다."""
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("선택 가능한 행동이 없는 마스크입니다.")
        if track:
            self.decisions += 1
            self.known_decisions += int(key in self.q)
        if rng.random() < epsilon:
            return int(rng.choice(legal))
        values = self.values(key)[legal]
        return int(rng.choice(legal[np.flatnonzero(values == values.max())]))

    def update(self, key, action, target):
        """지정한 한 상태·행동의 Q를 고정 학습률로 타깃 방향으로 갱신합니다."""
        if not np.isfinite(target):
            raise FloatingPointError("타깃에 NaN 또는 무한대가 있습니다.")
        values = self.q.setdefault(key, np.zeros(9, dtype=float))
        values[action] += self.alpha * (target - values[action])
        if not np.isfinite(values[action]):
            raise FloatingPointError("Q값에 NaN 또는 무한대가 있습니다.")
        self.updates += 1

    def finish_episode(self, trajectory):
        """시간순 (관측 key, 행동, 보상) 경험열에 first-visit MC를 적용합니다.

        고정 alpha를 사용하는 MC 제어입니다. 표본평균 추정과 구분합니다.
        에피소드 수집 중 정책은 고정하며 사망으로 경험열을 나누지 않습니다.
        """
        returns = [0.0] * len(trajectory)
        total = 0.0
        for i in range(len(trajectory)-1, -1, -1):
            total = trajectory[i][2] + self.gamma * total
            returns[i] = total
        seen = set()
        for (key, action, _), target in zip(trajectory, returns):
            if (key, action) not in seen:
                self.update(key, action, target)
                seen.add((key, action))

    def sarsa_step(self, key, action, reward, next_key=None, next_action=None, terminated=False):
        """실제 다음 행동의 Q로 bootstrap하며 실제 종료에서는 마지막 보상만 씁니다."""
        target = float(reward)
        if not terminated:
            if next_key is None or next_action is None:
                raise ValueError("SARSA에는 실제로 선택한 다음 행동이 필요합니다.")
            target += self.gamma * self.values(next_key)[next_action]
        self.update(key, action, target)


def save_models(path, models, metadata):
    """학습자들의 Q-table과 재현용 설정을 gzip JSON 파일로 저장합니다."""
    payload = {"metadata": metadata, "models": [
        {"algorithm": m.algorithm, "alpha": m.alpha, "gamma": m.gamma, "updates": m.updates,
         "q": [[list(key), values.tolist()] for key, values in sorted(m.q.items())]}
        for m in models]}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)


def load_models(path):
    """체크포인트에서 학습자 목록과 설정 메타데이터를 복원합니다."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    models = []
    for item in payload["models"]:
        model = TabularControl(item["algorithm"], item["alpha"], item["gamma"])
        model.q = {tuple(key): np.array(value, dtype=float) for key, value in item["q"]}
        model.updates = item["updates"]
        models.append(model)
    return models, payload["metadata"]
