"""실험별 번호 할당 및 체크포인트 기록입니다."""
import csv
from contextlib import contextmanager
import json
import os
import tomllib
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiment"


def experiment(name):
    if not name or name in (".", "..") or Path(name).name != name or ":" in name:
        raise ValueError("실험 폴더 이름을 지정하세요.")
    path = EXPERIMENTS / name
    with (path / "config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    validate(cfg)
    return path, cfg


def validate(cfg):
    """학습을 시작하기 전에 반복 횟수와 비율의 입력 오류를 찾습니다."""
    from territory_rl.config import GameConfig
    GameConfig(**cfg["game"], initial_placement="frontier")
    ssl, ppo = cfg["ssl"], cfg["ppo"]
    for opt, keys in ((ssl, ("latent","hidden","episodes","frames_per_episode","epochs","batch_size")),
                      (ppo, ("hidden","entity_hidden","updates","episodes_per_update","epochs","batch_size","checkpoint_every"))):
        for key in keys:
            if type(opt[key]) is not int or opt[key] < 1:
                raise ValueError(f"{key} 값은 양의 정수여야 합니다.")
    if ssl["episodes"] < 2 or not 0 <= ssl["mask_fraction"] < 1 or not 0 < ssl["validation_fraction"] < 1:
        raise ValueError("SSL 경기 수와 마스킹·검증 비율을 확인하세요.")
    mode = cfg.get("representation", "masked")
    if mode not in ("masked", "ae", "raw") or (mode == "masked") != (ssl["mask_fraction"] > 0):
        raise ValueError("입력 방식과 사전학습 마스크 비율을 확인하세요.")
    if "online_validation" in cfg:
        games = cfg["online_validation"]["games"]
        if type(games) is not int or games < 2 or games % 2:
            raise ValueError("학습 중 검증 경기 수는 양의 짝수여야 합니다.")
    if not all(0 <= ppo[k] <= 1 for k in ("gamma","gae_lambda")) or not 0 < ppo["clip"] < 1:
        raise ValueError("PPO 할인·GAE·clip 값을 확인하세요.")
    if min(ssl["learning_rate"], ppo["learning_rate"], ppo["max_grad_norm"]) <= 0:
        raise ValueError("학습률과 그래디언트 상한은 양수여야 합니다.")


def allocate(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000000):
        path = parent / f"{prefix}{index}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("실행 폴더 번호를 할당할 수 없습니다.")


def save_pt(path, obj):
    temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def load_pt(path):
    # 이 실험에서 직접 생성한 로컬 체크포인트만 읽습니다.
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, obj):
    temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def run_lock(run):
    """동일 run에 동시에 쓰지 않도록 운영체제 잠금을 잡습니다. 종료하면 자동 해제됩니다."""
    with (run/".training.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"이 run은 다른 학습이 사용 중입니다: {run}") from exc
        yield
