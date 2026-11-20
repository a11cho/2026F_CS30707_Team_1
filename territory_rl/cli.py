import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from .config import ExperimentConfig, GameConfig
from .controller import FrontierController, RandomController
from .environment import TerritoryEnvironment
from .manager import EpisodeManager
from .models import AGENT_IDS


def replay(path):
    """JSONL의 reset 설정과 행동열을 다시 실행해 저장된 결과와 정확히 일치하는지 검사한다.
    이벤트·면적·보상 등 어느 필드라도 다르면 해당 틱의 ValueError를 발생시킨다.
    성공 시 검증한 전이 수를 출력한다. 모델 학습이나 정책 갱신은 수행하지 않는다.
    """
    env = None
    count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["type"] == "reset":
            env = TerritoryEnvironment(GameConfig(**item["game"]), ExperimentConfig(**item["experiment"]))
            env.reset(item["seed"], {"observation_rotation": item["observation_rotation"]})
        elif item["type"] == "step":
            if env is None:
                raise ValueError("trace is missing its reset record")
            if "decision" in item:
                env.begin_decision()
                result = None
                rounds = item["decision"]
                expected_rounds = 2 if env.decision_strategy == "feedback" else 1
                if len(rounds) != expected_rounds:
                    raise ValueError("incorrect decision round count")
                for entry in rounds:
                    if entry["round"] != env.decision_round or set(entry["requested"]) != set(AGENT_IDS):
                        raise ValueError("invalid decision trace")
                    for aid in AGENT_IDS:
                        team, number = aid.rsplit("_", 1)
                        env.agent_session(team, int(number)).submit_action(entry["observation_requested"][aid])
                    result = env.finish_decision_round()
                if env.last_decision != rounds or item["actions"] != rounds[-1]["requested"]:
                    raise ValueError("decision trace mismatch")
            else:
                if env.decision_strategy != "independent":
                    raise ValueError("strategy replay requires decision rounds")
                result = env.step(item["actions"])
            actual = json.loads(json.dumps(result.to_dict()))
            expected = {k: item[k] for k in actual}
            if actual != expected:
                raise ValueError(f"replay mismatch at tick {actual['tick']}")
            count += 1
    print(json.dumps({"replayed_transitions": count, "verified": True}))


def main(argv=None):
    """명령행 옵션을 해석해 keyboard·headless·replay 중 선택한 실행 모드를 시작한다.

    argv가 None이면 프로세스 인자를 사용한다. headless는 random/scripted 컨트롤러로
    경기를 실행하고 지표를 출력하며, --log 지정 시 행동 재생용 JSONL을 기록한다.
    keyboard 모드에서만 GUI를 불러온다. 어떤 모드도 학습을 실행하지 않는다.
    """
    defaults, exp_defaults = GameConfig(), ExperimentConfig()
    parser = argparse.ArgumentParser(description="Territory Control functional test runner")
    parser.add_argument("--mode", choices=["keyboard", "headless", "replay"], default="keyboard")
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--ticks", type=int, default=defaults.max_ticks)
    parser.add_argument("--respawn-delay", type=int, default=defaults.respawn_delay)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observation-window-size", type=int, default=exp_defaults.observation_window_size)
    parser.add_argument("--randomize-observation-rotation", action="store_true",
                        help="use one shared random 0/90/180/270-degree observation rotation per episode")
    parser.add_argument("--strategy", choices=["independent", "sequential", "feedback"], default=exp_defaults.decision_strategy)
    parser.add_argument("--reward", choices=["individual", "team"], default="team")
    parser.add_argument("--team-signal", choices=["area", "gap", "terminal"], default=exp_defaults.team_signal)
    parser.add_argument("--reward-mask", nargs=3, type=int, choices=(0, 1), metavar=("AREA", "GAP", "TERMINAL"),
                        help="0/1 mask; overrides team-signal/include-terminal-reward; enabled terms are summed")
    parser.add_argument("--capture-weight", type=float, default=exp_defaults.capture_weight)
    parser.add_argument("--area-weight", type=float, default=exp_defaults.area_weight)
    parser.add_argument("--gap-weight", type=float, default=exp_defaults.gap_weight)
    parser.add_argument("--win-reward", type=float, default=exp_defaults.win_reward)
    parser.add_argument("--loss-reward", type=float, default=exp_defaults.loss_reward)
    parser.add_argument("--draw-reward", type=float, default=exp_defaults.draw_reward)
    parser.add_argument("--include-terminal-reward", action="store_true")
    parser.add_argument("--manual-all", action="store_true")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--opponent", choices=["random", "expand", "aggressive", "defend"], default="aggressive")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--log", type=Path, help="JSONL trace output; input in replay mode")
    args = parser.parse_args(argv)
    if args.mode == "replay":
        if args.log is None:
            parser.error("replay requires --log")
        replay(args.log)
        return
    if args.interval_ms < 1 or args.episodes < 1:
        parser.error("interval-ms and episodes must be positive")
    config = GameConfig(height=args.height, width=args.width, max_ticks=args.ticks, respawn_delay=args.respawn_delay)
    experiment = ExperimentConfig(decision_strategy=args.strategy, reward_mode=args.reward,
                                  team_signal=args.team_signal, capture_weight=args.capture_weight,
                                  area_weight=args.area_weight, gap_weight=args.gap_weight,
                                  reward_signal_mask=args.reward_mask, win_reward=args.win_reward,
                                  loss_reward=args.loss_reward, draw_reward=args.draw_reward,
                                  observation_window_size=args.observation_window_size,
                                  randomize_observation_rotation=args.randomize_observation_rotation,
                                  include_terminal_reward=args.include_terminal_reward)
    if args.mode == "keyboard":
        from .keyboard_demo import KeyboardDemo
        KeyboardDemo(config, experiment, args.seed, args.manual_all, args.interval_ms, log_path=args.log).run()
        return
    stream = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log.open("w", encoding="utf-8")
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            controllers = {aid: (RandomController(seed+i) if args.opponent == "random" else
                                  FrontierController(seed+i, args.opponent, experiment.tile_encoding)) for i, aid in enumerate(AGENT_IDS)}
            manager = EpisodeManager(TerritoryEnvironment(config, experiment), controllers)
            manager.reset(seed)
            if stream:
                stream.write(json.dumps({"type": "reset", "seed": seed, "game": asdict(config),
                                         "experiment": asdict(experiment),
                                         "observation_rotation": manager.environment.observation_rotation}) + "\n")
            start = time.perf_counter()
            while not manager.environment.terminated:
                result = manager.step()
                if stream:
                    record = result.to_dict()
                    record.update(type="step", actions={a: int(v) for a, v in manager.last_actions.items()},
                                  decision=manager.environment.last_decision)
                    stream.write(json.dumps(record) + "\n")
            summary = manager.metrics.summary()
            summary.update(seed=seed, steps_per_second=manager.environment.tick / (time.perf_counter()-start))
            print(json.dumps(summary))
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()

