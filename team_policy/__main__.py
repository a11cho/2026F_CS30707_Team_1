import argparse
import math
from pathlib import Path

from .curriculum import LESSONS, LESSONS_WITH_INTERMEDIATE_SEARCH
from .experiment import evaluate_checkpoint, train
from .fixed_scenes import SPLITS
from .match import OPPONENTS, SCRIPTED
from .search import SEARCH_TEACHERS
from .state import RULES_PROMPTS


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def opponent_list(value):
    names = tuple(value.split(","))
    if any(name not in OPPONENTS for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"choose distinct opponents from {', '.join(OPPONENTS)}")
    return names


def weights(value):
    values = tuple(map(float, value.split(",")))
    total = sum(values)
    if any(weight < 0 for weight in values) or not math.isfinite(total) or total <= 0:
        raise argparse.ArgumentTypeError("opponent weights must be finite and nonnegative with a positive sum")
    return values


def main():
    parser = argparse.ArgumentParser(description="Centralized 3v3 territory policy")
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train", help="initialize and tune a centralized team policy")
    evaluator = commands.add_parser("evaluate", help="evaluate a frozen policy in paired fixtures")
    for command in (trainer, evaluator):
        command.add_argument("--device", default="cpu")
        command.add_argument("--base", type=Path, help="local snapshot of the pinned Qwen base")
        command.add_argument("--force-entry", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--rules-prompt", action=argparse.BooleanOptionalAction, default=None,
                             help="include rules and ally cover guidance in Qwen input (defaults to the checkpoint setting)")
        command.add_argument("--rules-prompt-version", choices=tuple(RULES_PROMPTS), type=int,
                             help="choose the rules text when enabled (defaults to the checkpoint version or 1)")
        command.add_argument("--tactical-metrics", action="store_true", help="measure unsafe entry and defendable losses in games")
        command.add_argument("--inference-batch", type=positive, default=4)
        command.add_argument("--eval-opponents", type=opponent_list, default=("expand",))
        command.add_argument("--eval-seed", type=nonnegative, default=610000)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--backbone", choices=("cnn", "bf16", "nf4"), default="cnn")
    trainer.add_argument("--additive-only", action="store_true")
    trainer.add_argument("--seed", type=nonnegative, default=7)
    trainer.add_argument("--algorithm", choices=("ppo", "grpo", "dapo", "sdpo"), default="ppo")
    trainer.add_argument("--group-size", type=positive, default=4)
    trainer.add_argument("--terminal-margin-weight", type=float, default=0.0,
                         help="add this coefficient times the final normalized area gap to the training reward")
    trainer.add_argument("--opponents", type=opponent_list, default=SCRIPTED)
    trainer.add_argument("--opponent-weights", type=weights)
    trainer.add_argument("--frozen-opponents", nargs="+", default=[])
    trainer.add_argument("--frozen-weight", type=float, default=0.4)
    trainer.add_argument("--initialize", type=Path)
    trainer.add_argument("--resume-optimizer", action="store_true",
                         help="restore AdamW state from the online checkpoint passed to --initialize")
    trainer.add_argument("--keep-snapshots", action="store_true")
    trainer.add_argument("--ticks", type=nonnegative, default=1024)
    trainer.add_argument("--warmup-ticks", type=nonnegative, default=128)
    trainer.add_argument("--warmup-envs", type=positive, default=1)
    trainer.add_argument("--warmup-epochs", type=positive, default=2)
    trainer.add_argument("--teacher-temperature", type=positive_float, default=0.04)
    trainer.add_argument("--student-warmup-fraction", type=float, default=0.0,
                         help="fraction of teacher-labeled warm-up episodes driven by the initial policy")
    trainer.add_argument("--envs", type=positive, default=8)
    trainer.add_argument("--rollout-steps", type=positive, default=128)
    trainer.add_argument("--batch-size", type=positive, default=32)
    trainer.add_argument("--microbatch", type=positive, default=4)
    trainer.add_argument("--epochs", type=positive, default=2)
    trainer.add_argument("--head-lr", type=positive_float, default=3e-4)
    trainer.add_argument("--lora-lr", type=positive_float, default=1e-5)
    trainer.add_argument("--lora-lr-warmup-steps", type=nonnegative, default=0,
                         help="ramp the LoRA learning rate to --lora-lr over this many PPO optimizer steps")
    trainer.add_argument("--activation-checkpointing", action="store_true")
    trainer.add_argument("--freeze-text-backbone", action="store_true",
                         help="cache Qwen features in the rollout and tune the projections, CNN, and heads")
    trainer.add_argument("--curriculum", action="store_true")
    trainer.add_argument("--curriculum-start", type=nonnegative, default=0)
    trainer.add_argument("--curriculum-intermediate-search", action="store_true",
                         help="insert an 80-tick intermediate search lesson")
    trainer.add_argument("--curriculum-fine-search", action="store_true",
                         help="gradually raise the intermediate training opponent's search rate while keeping its exam fixed")
    trainer.add_argument("--curriculum-fine-tactical", action="store_true",
                         help="ramp the tactical lesson's training search budgets while keeping its exam fixed")
    trainer.add_argument("--curriculum-leaf-budget", action="store_true",
                         help="adapt the tactical lesson's candidate limit to paired scores for a fixed-stage run")
    trainer.add_argument("--curriculum-tactical-exploration", action="store_true",
                         help="adapt the tactical lesson's random-action probability to paired scores")
    trainer.add_argument("--curriculum-exploration-schedule", type=int, choices=range(55, 96, 5), nargs="+",
                         help="replay one random-action percentage per tactical lesson check interval")
    trainer.add_argument("--tactical-exercises", action="store_true",
                         help="mix short tactical exercises into the intermediate lesson")
    trainer.add_argument("--keep-curriculum-checkpoints", action="store_true",
                         help="save the checkpoint evaluated at each curriculum exam")
    trainer.add_argument("--curriculum-interval", type=positive, default=4096)
    trainer.add_argument("--curriculum-pairs", type=positive, default=8)
    trainer.add_argument("--curriculum-target", type=positive_float, default=0.625)
    trainer.add_argument("--curriculum-seed", type=nonnegative, default=640000)
    trainer.add_argument("--search-imitation-weight", type=float, default=0.0,
                         help="add search imitation to PPO on 25%% of eligible learner states")
    trainer.add_argument("--search-teacher", choices=SEARCH_TEACHERS,
                         help="search policy for PPO labels (defaults to the checkpoint or search_strong)")
    trainer.add_argument("--eval-pairs", type=nonnegative, default=0)
    evaluator.add_argument("checkpoint", type=Path)
    evaluator.add_argument("--explicit-map-prompt", action="store_true",
                           help="label Qwen map coordinates and list the current agent positions")
    evaluator.add_argument("--eval-pairs", type=nonnegative, default=1)
    evaluator.add_argument("--eval-max-ticks", type=positive, help="use a lesson's game length for paired evaluation")
    evaluator.add_argument("--eval-timeline", action="store_true", help="record areas, captures, and deaths for each game tick")
    evaluator.add_argument("--scene-pairs", type=nonnegative, default=0)
    evaluator.add_argument("--tactical-probe-pairs", type=nonnegative, default=0,
                           help="run paired one-tick probes on separate tactical exercise seeds")
    evaluator.add_argument("--tactical-probe-actions", action="store_true",
                           help="record the starting board, commands, and engine events for each probe")
    evaluator.add_argument("--scene-split", choices=SPLITS, default="validation")
    args = parser.parse_args()
    if args.command == "train":
        if not (args.ticks or args.warmup_ticks):
            parser.error("train requires warm-up or online ticks")
        if args.algorithm in ("grpo", "dapo") and not 2 <= args.group_size <= args.envs:
            parser.error("group learning needs 2 to --envs episodes per group")
        if args.algorithm == "sdpo" and args.backbone == "cnn":
            parser.error("SDPO requires a text backbone for engine feedback")
        if not math.isfinite(args.terminal_margin_weight) or args.terminal_margin_weight < 0:
            parser.error("--terminal-margin-weight must be finite and nonnegative")
        if not 0 <= args.student_warmup_fraction <= 1:
            parser.error("--student-warmup-fraction must be in [0, 1]")
        if args.student_warmup_fraction and (args.initialize is None or not args.warmup_ticks):
            parser.error("student-driven warm-up requires --initialize and --warmup-ticks")
        if args.resume_optimizer and (args.initialize is None or args.warmup_ticks or not args.ticks):
            parser.error("--resume-optimizer requires --initialize, --warmup-ticks 0, and online training")
        if args.lora_lr_warmup_steps and (
            args.algorithm != "ppo" or args.backbone == "cnn" or args.freeze_text_backbone
            or args.resume_optimizer or not args.ticks
        ):
            parser.error("LoRA learning-rate warm-up requires online Qwen PPO with a new, unfrozen optimizer")
        if args.activation_checkpointing and args.backbone == "cnn":
            parser.error("activation checkpointing requires a Qwen backbone")
        if args.freeze_text_backbone and (args.backbone == "cnn" or args.algorithm != "ppo"):
            parser.error("cached text-backbone training requires Qwen PPO")
        if args.freeze_text_backbone and args.activation_checkpointing:
            parser.error("choose cached text features or activation checkpointing")
        if args.curriculum and args.algorithm != "ppo":
            parser.error("the curriculum runner uses PPO")
        if args.curriculum and args.curriculum_target > 1:
            parser.error("the curriculum target score must be at most 1")
        if args.curriculum_fine_search and not (args.curriculum and args.curriculum_intermediate_search):
            parser.error("--curriculum-fine-search requires --curriculum and --curriculum-intermediate-search")
        if args.curriculum_fine_tactical and not args.curriculum:
            parser.error("--curriculum-fine-tactical requires --curriculum")
        if args.curriculum_leaf_budget and (not args.curriculum or args.curriculum_fine_tactical):
            parser.error("--curriculum-leaf-budget requires --curriculum and its own tactical schedule")
        if args.curriculum_tactical_exploration and (
            not args.curriculum or args.curriculum_leaf_budget or args.curriculum_fine_tactical
        ):
            parser.error("--curriculum-tactical-exploration requires --curriculum and its own tactical schedule")
        if args.tactical_exercises and not (args.curriculum and args.curriculum_intermediate_search and args.envs >= 4):
            parser.error("--tactical-exercises requires the intermediate curriculum and at least four environments")
        if args.keep_curriculum_checkpoints and not args.curriculum:
            parser.error("--keep-curriculum-checkpoints requires --curriculum")
        lessons = LESSONS_WITH_INTERMEDIATE_SEARCH if args.curriculum_intermediate_search else LESSONS
        if args.curriculum and args.curriculum_start >= len(lessons):
            parser.error("--curriculum-start is outside the selected lesson schedule")
        if args.curriculum_exploration_schedule:
            if not args.curriculum_tactical_exploration or lessons[args.curriculum_start].opponent != "search_tactical":
                parser.error("--curriculum-exploration-schedule requires starting in the tactical exploration lesson")
            intervals = (args.ticks + args.curriculum_interval - 1) // args.curriculum_interval
            if len(args.curriculum_exploration_schedule) != intervals:
                parser.error("--curriculum-exploration-schedule requires one percentage per training interval")
        if args.curriculum and args.frozen_opponents:
            parser.error("curriculum opponents are supplied by the lesson schedule")
        if not math.isfinite(args.search_imitation_weight) or args.search_imitation_weight < 0:
            parser.error("--search-imitation-weight must be finite and nonnegative")
        if args.search_imitation_weight and args.algorithm != "ppo":
            parser.error("search imitation is implemented for PPO")
        if args.opponent_weights is not None and len(args.opponents) != len(args.opponent_weights):
            parser.error("--opponent-weights must match --opponents")
        if not 0 <= args.frozen_weight < 1:
            parser.error("--frozen-weight must be in [0, 1)")
        train(args)
    else:
        if not (args.eval_pairs or args.scene_pairs or args.tactical_probe_pairs):
            parser.error("evaluate requires full-match, fixed-scene, or tactical probe pairs")
        if args.tactical_probe_actions and not args.tactical_probe_pairs:
            parser.error("--tactical-probe-actions requires --tactical-probe-pairs")
        if args.eval_timeline and not args.eval_pairs:
            parser.error("--eval-timeline requires --eval-pairs")
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
