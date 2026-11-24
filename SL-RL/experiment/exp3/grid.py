"""공통 설정에서 마스킹 칸 수·잠재 차원별 실행 조건을 만듭니다."""
from storage import validate

BASELINES = {(0, 16): "exp2", (9, 16): "exp1"}
SIZES = (1, 3)


def label(key):
    masked, latent = key
    return f"mask{masked:02d}_z{latent}"


def key_of(cfg):
    cells = cfg["game"]["height"] * cfg["game"]["width"]
    return round(cells * cfg["ssl"]["mask_fraction"]), cfg["ssl"]["latent"]


def conditions(cfg):
    grid = cfg["ae_grid"]
    cells = cfg["game"]["height"] * cfg["game"]["width"]
    for field, low, high in (("masked_cells", 0, cells-1), ("latent_sizes", 1, None), ("seeds", 0, None)):
        vals = grid[field]
        if (not vals or any(type(v) is not int or v < low or (high is not None and v > high) for v in vals)
                or len(set(vals)) != len(vals)):
            raise ValueError(f"ae_grid.{field}의 범위와 중복을 확인하세요.")
    if type(grid["reuse_latent16"]) is not bool:
        raise ValueError("reuse_latent16에는 true 또는 false를 지정하세요.")
    base = {k: v for k, v in cfg.items() if k != "ae_grid"}
    result = {}
    for masked in grid["masked_cells"]:
        for latent in grid["latent_sizes"]:
            ssl = dict(base["ssl"], mask_fraction=masked/cells, latent=latent)
            variant = dict(base, ssl=ssl, representation="masked" if masked else "ae")
            validate(variant)
            result[masked, latent] = variant
    return result


def copied_source(cfg, key):
    return BASELINES.get(key) if cfg["ae_grid"]["reuse_latent16"] else None


def compatible(actual, expected):
    """과거 exp1의 저장 주기는 달랐으므로 학습 조건 비교에서 별도로 취급합니다."""
    def signature(cfg):
        ppo = {k: v for k, v in cfg["ppo"].items() if k != "checkpoint_every"}
        return cfg.get("representation", "masked"), cfg["game"], cfg["ssl"], ppo, cfg["evaluation"]
    return signature(actual) == signature(expected)
