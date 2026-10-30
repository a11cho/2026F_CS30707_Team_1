import numpy as np


class TileMap:
    def __init__(self, height, width, owner=None):
        if owner is None:
            owner = np.zeros((height, width), dtype=np.int8)
            owner[:, width // 2:] = 1
        raw = np.asarray(owner)
        if raw.shape != (height, width) or not np.isin(raw, [0, 1]).all():
            raise ValueError("owner must have map shape and only team IDs 0/1")
        self.owner = raw.astype(np.int8, copy=True)

    def contains(self, p):
        return 0 <= p[0] < self.owner.shape[0] and 0 <= p[1] < self.owner.shape[1]

    def areas(self):
        return tuple(int(np.count_nonzero(self.owner == k)) for k in range(2))

