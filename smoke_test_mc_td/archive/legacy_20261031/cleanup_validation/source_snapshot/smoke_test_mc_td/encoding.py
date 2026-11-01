"""정보를 축약하는 이산 관측 표현 compact_v1입니다.

정책이 볼 수 있는 관측만 사용하며 완전한 Markov state를 복원하지 않습니다.
두 모드의 key 길이는 같으며 aware 정보도 key에 실제 반영합니다.
"""
import numpy as np


def bucket(value, bins=4):
    """정규화된 수치를 bins개의 정수 구간으로 바꿉니다."""
    return min(bins - 1, max(0, int(float(value) * bins)))


def encode(obs):
    """에이전트에게 공개된 관측만 읽어 Q-table 조회용 불변 튜플로 요약합니다."""
    grid = obs["local_grid"]
    r = grid.shape[1] // 2
    features = obs["self_features"]
    key = [int(features[2]), int(features[3]), bucket(features[4]), bucket(features[5]),
           bucket(features[0]), bucket(features[1])]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        key.extend(int(v) for v in grid[[0, 1, 2, 4, 5], r+dy, r+dx])
    # 시야 안에서 가장 가까운 적 영역과 적 에이전트의 대략적인 방향입니다.
    for channel in (1, 5):
        cells = np.argwhere(grid[channel] > 0)
        if len(cells):
            distances = np.abs(cells - r).sum(axis=1)
            y, x = cells[int(np.argmin(distances))] - r
            key.extend([int(np.sign(y)), int(np.sign(x)), min(3, int(abs(y)+abs(x)))])
        else:
            key.extend([0, 0, -1])
    for values, mask in zip(obs["teammate_features"], obs["teammate_feature_mask"]):
        key.extend([bucket(values[0]), bucket(values[1]), int(values[2]), int(values[3]),
                    bucket(values[4]), int(np.argmax(values[5:])) if mask[5:].any() else -1])
        key.extend(int(v) for v in mask)
    key.extend(int(v) for v in obs["action_mask"])
    return tuple(key)
