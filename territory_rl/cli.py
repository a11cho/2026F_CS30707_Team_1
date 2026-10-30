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
    env = None
    count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["type"] == "reset":
            env = TerritoryEnvironment(GameConfig(**item["game"]), ExperimentConfig(**item["experiment"]))
            env.reset(item["seed"])
        elif item["type"] == "step":
            if env is None:
                raise ValueError("trace is missing its reset record")
            actual = json.loads(json.dumps(env.step(item["actions"]).to_dict()))
            expected = {k: item[k] for k in actual}
            if actual != expected:
                raise ValueError(f"replay mismatch at tick {actual['tick']}")
            count += 1
    print(json.dumps({"replayed_transitions": count, "verified": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Territory Control functional test runner")
    parser.add_argument("--mode", choices=["keyboard", "headless", "replay"], default="keyboard")
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--ticks", type=int, default=256)
    parser.add_argument("--respawn-delay", type=int, default=2)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observation", choices=["local", "aware"], default="local")
    parser.add_argument("--reward", choices=["individual", "team"], default="team")
    parser.add_argument("--team-signal", choices=["net", "gross", "area", "terminal"], default="net")
    parser.add_argument("--terminal-bonus", type=float)
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
    config = GameConfig(args.height, args.width, args.ticks, args.respawn_delay, args.radius)
    experiment = ExperimentConfig(args.observation, args.reward, args.team_signal,
                                  args.terminal_bonus if args.terminal_bonus is not None else float(args.reward == "team"))
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
                                  FrontierController(seed+i, args.opponent)) for i, aid in enumerate(AGENT_IDS)}
            manager = EpisodeManager(TerritoryEnvironment(config, experiment), controllers)
            manager.reset(seed)
            if stream:
                stream.write(json.dumps({"type": "reset", "seed": seed, "game": asdict(config), "experiment": asdict(experiment)}) + "\n")
            start = time.perf_counter()
            while not manager.environment.terminated:
                result = manager.step()
                if stream:
                    record = result.to_dict()
                    record.update(type="step", actions={a: int(v) for a, v in manager.last_actions.items()})
                    stream.write(json.dumps(record) + "\n")
            summary = manager.metrics.summary()
            summary.update(seed=seed, steps_per_second=config.max_ticks / (time.perf_counter()-start))
            print(json.dumps(summary))
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()

