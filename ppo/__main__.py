import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch

from territory_rl.config import ExperimentConfig, GameConfig

from .evaluation import evaluate
from .policy import policy_from_checkpoint
from .rollout import OPPONENTS
from .sender import SenderActionPredictor
from .trainer import TrainingConfig, train


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the feedback PPO team.")
    parser.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    defaults = TrainingConfig()
    training.add_argument("--output", type=Path, required=True, help="new run directory")
    training.add_argument("--seed", type=int, default=defaults.seed)
    training.add_argument("--num-envs", type=positive_int, default=defaults.num_envs)
    training.add_argument("--rollout-ticks", type=positive_int, default=defaults.rollout_ticks)
    training.add_argument("--updates", type=positive_int, default=defaults.updates)
    training.add_argument("--eval-every", type=positive_int, default=defaults.eval_every)
    training.add_argument("--validation-pairs", type=positive_int, default=defaults.validation_pairs)
    training.add_argument("--reward", choices=("team", "individual"), default="team")
    training.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=defaults.opponents)
    training.add_argument("--hide-teammates", action="store_true")
    training.add_argument("--map-size", type=positive_int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(8, 8))
    training.add_argument("--canvas-size", type=positive_int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    training.add_argument("--no-respawn", action="store_true", help="keep dead agents out until the match ends")
    training.add_argument("--sender-prediction-coef", type=nonnegative_float,
                          default=defaults.sender_prediction_coef, help="weight of the sender action prediction loss")
    training.add_argument("--backpropagate-messages", action="store_true",
                          help="route receiver policy gradients to its teammates' proposal heads")

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=OPPONENTS)
    evaluation.add_argument("--seed", type=int, default=200_000)
    evaluation.add_argument("--pairs", type=positive_int, default=8, help="seed count per opponent; play both sides")
    evaluation.add_argument("--map-size", type=positive_int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    evaluation.add_argument("--respawn", choices=("immediate", "none"))
    evaluation.add_argument("--cooperation-scenes", action="store_true",
                            help="measure all seven team-defined cooperation scenes")

    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("--opponents must be unique")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "train":
        config = TrainingConfig(
            seed=args.seed, num_envs=args.num_envs, rollout_ticks=args.rollout_ticks,
            updates=args.updates, eval_every=args.eval_every, validation_pairs=args.validation_pairs,
            hide_teammates=args.hide_teammates, opponents=tuple(args.opponents),
            sender_prediction_coef=args.sender_prediction_coef,
            backpropagate_messages=args.backpropagate_messages,
        )
        experiment = ExperimentConfig(decision_strategy="feedback", reward_mode=args.reward,
                                      team_signal="terminal", loss_reward=-1.0)
        game = GameConfig(height=args.map_size[0], width=args.map_size[1])
        if args.no_respawn:
            game = replace(game, respawn_delay=game.max_ticks)
        train(game, experiment, config, args.output,
              canvas_shape=tuple(args.canvas_size) if args.canvas_size else None)
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training_game = GameConfig(**checkpoint["game"])
    game = training_game
    if args.map_size:
        game = replace(game, height=args.map_size[0], width=args.map_size[1])
    if args.respawn:
        game = replace(game, respawn_delay=game.max_ticks if args.respawn == "none" else 0)
    experiment = ExperimentConfig(**checkpoint["experiment"])
    policy = policy_from_checkpoint(checkpoint)
    sender_predictor = SenderActionPredictor.from_checkpoint(checkpoint)
    if game.height > policy.canvas_shape[0] or game.width > policy.canvas_shape[1]:
        raise SystemExit(f"evaluation map {(game.height, game.width)} exceeds the saved canvas {policy.canvas_shape}")
    hidden = checkpoint["training"]["hide_teammates"]
    report = {
        "checkpoint": str(args.checkpoint),
        "training_game": asdict(training_game),
        "evaluation_game": asdict(game),
        "canvas_shape": policy.canvas_shape,
        "update": checkpoint["update"],
        "seed": args.seed,
        "training_seed": checkpoint["training"]["seed"],
        "sender_prediction_coef": checkpoint["training"].get("sender_prediction_coef", 0.0),
        "backpropagate_messages": checkpoint["training"].get("backpropagate_messages", False),
        "pairs_per_opponent": args.pairs,
        "training_mode": evaluate(policy, game, experiment, args.opponents, args.seed, args.pairs, hidden,
                                  measure_cooperation=True, sender_predictor=sender_predictor,
                                  measure_cooperation_scenes=args.cooperation_scenes),
    }
    if not hidden:
        report["without_teammate_proposals"] = evaluate(
            policy, game, experiment, args.opponents, args.seed, args.pairs, hide_teammates=True,
            measure_cooperation=True, sender_predictor=sender_predictor,
            measure_cooperation_scenes=args.cooperation_scenes,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
