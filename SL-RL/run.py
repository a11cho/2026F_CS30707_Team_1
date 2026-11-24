"""실험 실행 진입점. 저장소 어디에서 실행해도 동일한 실험 경로를 사용합니다."""
import argparse
import re
import sys
from pathlib import Path

# 원본 패키지는 저장소 루트에 있습니다. 설치나 원본 수정 없이 가져옵니다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation import evaluate
from pretrain import pretrain
from checkpoint_selection import DEFAULT_GAMES, keep_last_and_best
from storage import experiment, run_lock, validate
from training import train


def numbered(prefix):
    def parse(value):
        if not re.fullmatch(prefix+r"[1-9][0-9]*", value):
            raise argparse.ArgumentTypeError(f"{prefix} 뒤에 양의 정수를 붙이세요.")
        return value
    return parse


def main():
    parser = argparse.ArgumentParser(description="점령 지도 자기지도 사전학습과 팀별 PPO")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("pretrain", "train", "resume", "evaluate", "duel", "prune-checkpoints"):
        sub = subs.add_parser(name)
        sub.add_argument("--experiment", default="exp1")
        sub.add_argument("--condition", help="AE 그리드 조건 이름 (예: mask09_z8)")
        if name in ("pretrain","train"):
            sub.add_argument("--size", type=int, choices=(1,3), required=True)
        if name == "train":
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--representation", choices=("ae", "raw"), help="exp2의 지도 입력 방식")
            sub.add_argument("--fit", type=numbered("fit"))
        if name in ("resume","evaluate","duel"):
            sub.add_argument("--run", type=numbered("run"), required=True)
        if name == "prune-checkpoints":
            sub.add_argument("--run", nargs="+", type=numbered("run"), required=True)
            sub.add_argument("--at-update", type=int, required=True, help="정리할 최신 체크포인트의 업데이트 번호")
        if name in ("train","resume","prune-checkpoints"):
            sub.add_argument("--validation-games", type=int, help="학습 팀·상대별 검증 경기 수(짝수)")
        if name in ("train","resume"):
            sub.add_argument("--keep-last-and-best", action="store_true", help="완료 후 최종 및 검증 최고 체크포인트 보관")
            sub.add_argument("--updates", type=int, help="추가 횟수가 아닌 최종 목표 업데이트")
        if name in ("evaluate","duel"):
            sub.add_argument("--games", type=int)
            sub.add_argument("--checkpoint", default="latest.pt")
            sub.add_argument("--replay", action="store_true")
        if name == "evaluate":
            sub.add_argument("--opponents", nargs="+")
        if name == "duel":
            sub.add_argument("--versus", type=numbered("run"), required=True)
    a = parser.parse_args()
    path, cfg = experiment(a.experiment)
    if "ae_grid" in cfg:
        from experiment.exp3.grid import conditions, copied_source, key_of, label
        import json
        if getattr(a, "representation", None):
            parser.error("그리드 실험에서는 --condition으로 입력을 선택하세요.")
        if not a.condition and a.command in ("train", "pretrain"):
            parser.error("그리드 단일 학습에는 --condition이 필요합니다. 일괄 실행은 launch.py를 사용하세요.")
        if a.condition:
            options = {label(k): (k, v) for k, v in conditions(cfg).items()}
            if a.condition not in options:
                parser.error(f"가능한 조건: {sorted(options)}")
            key, variant = options[a.condition]
            source = copied_source(cfg, key)
            if source:
                if a.command not in ("evaluate", "duel"):
                    parser.error("16차원 복사본은 평가에 사용합니다.")
                path = path/"copied"/source
            cfg = variant
            if a.command in ("evaluate", "duel", "resume"):
                names = [a.run] + ([a.versus] if a.command == "duel" else [])
                for name in names:
                    settings = json.loads((path/name/"settings.json").read_text(encoding="utf-8"))
                    if key_of(settings["config"]) != key:
                        parser.error(f"{name}의 학습 조건이 {a.condition}와 다릅니다.")
    elif a.condition:
        parser.error("--condition은 AE 그리드 실험에서 사용하세요.")
    torch.set_num_threads(1)
    if getattr(a,"updates",None) is not None and a.updates < 1:
        parser.error("목표 업데이트는 양수여야 합니다.")
    select = a.command == "prune-checkpoints" or getattr(a, "keep_last_and_best", False)
    games = getattr(a, "validation_games", None)
    if games is not None and (not select or games < 2 or games % 2):
        parser.error("--validation-games는 최고 모델 선정 시에만 2 이상의 짝수로 지정하세요.")
    validation_games = (games or DEFAULT_GAMES) if select else None
    if a.command == "prune-checkpoints":
        for name in dict.fromkeys(a.run):
            run = path/name
            with run_lock(run):
                keep_last_and_best(run, a.at_update, validation_games)
    elif a.command == "pretrain":
        pretrain(path, cfg, a.size)
    elif a.command == "train":
        if a.representation:
            if "representation" not in cfg:
                parser.error("이 실험 설정에는 입력 방식 비교가 없습니다.")
            cfg = dict(cfg, representation=a.representation)
            validate(cfg)
        train(path, cfg, a.size, a.seed, fit_name=a.fit, updates=a.updates, validation_games=validation_games)
    elif a.command == "resume":
        train(path, cfg, None, None, resume=a.run, updates=a.updates, validation_games=validation_games)
    else:
        if not re.fullmatch(r"(latest|best|update_[0-9]+)\.pt", a.checkpoint):
            parser.error("체크포인트 파일명을 확인하세요.")
        evaluate(path, cfg, a.run, a.checkpoint, getattr(a,"versus",None),
                 cfg["evaluation"]["games"] if a.games is None else a.games,
                 getattr(a,"opponents",None) or cfg["evaluation"]["opponents"], a.replay)


if __name__ == "__main__":
    main()
