import numpy as np


class TileMap:
    def __init__(self, height, width, owner=None):
        """타일 소유권 배열을 생성한다. owner가 없으면 좌우를 팀 0/1로 절반씩 나눈다.
        직접 전달한 owner는 크기와 값(0 또는 1)을 검증한 뒤 복사하여 외부 배열과 분리한다.
        """
        if owner is None:
            owner = np.zeros((height, width), dtype=np.int8)
            owner[:, width // 2:] = 1
        raw = np.asarray(owner)
        if raw.shape != (height, width) or not np.isin(raw, [0, 1]).all():
            raise ValueError("owner must have map shape and only team IDs 0/1")
        self.owner = raw.astype(np.int8, copy=True)

    def contains(self, p):
        """(행, 열) 좌표 p가 맵 경계 안에 있는지 반환한다. 소유권·점유 여부는 검사하지 않는다."""
        return 0 <= p[0] < self.owner.shape[0] and 0 <= p[1] < self.owner.shape[1]

    def areas(self):
        """현재 팀 0과 팀 1이 소유한 타일 수를 (팀0 면적, 팀1 면적) 튜플로 반환한다."""
        return tuple(int(np.count_nonzero(self.owner == k)) for k in range(2))

