from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GameConfig:
    height: int = 12
    width: int = 16
    max_ticks: int = 256
    respawn_delay: int = 2
    observation_radius: int = 3
    initial_placement: str = "frontier"
    rules_version: str = "territory_v0"

    def __post_init__(self):
        for name in ("height", "width", "max_ticks", "respawn_delay", "observation_radius"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.height < 3 or self.width < 2 or self.width % 2:
            raise ValueError("height >= 3 and even width >= 2 are required")
        if self.max_ticks < 1 or self.respawn_delay < 0 or self.observation_radius < 1:
            raise ValueError("max_ticks/radius must be positive; respawn_delay >= 0")
        if self.initial_placement not in ("frontier", "random"):
            raise ValueError("initial_placement must be frontier or random")
        if self.rules_version != "territory_v0":
            raise ValueError("unsupported rules_version")


@dataclass(frozen=True)
class ExperimentConfig:
    observation_mode: str = "local"
    reward_mode: str = "team"
    team_signal: str = "net"
    terminal_bonus: float = 1.0

    def __post_init__(self):
        if self.observation_mode not in ("local", "aware"):
            raise ValueError("observation_mode must be local or aware")
        if self.reward_mode not in ("individual", "team"):
            raise ValueError("reward_mode must be individual or team")
        if self.team_signal not in ("net", "gross", "area", "terminal"):
            raise ValueError("unknown team_signal")
        if not math.isfinite(self.terminal_bonus) or self.terminal_bonus < 0:
            raise ValueError("terminal_bonus must be finite and nonnegative")

