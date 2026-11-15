from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GameConfig:
    height: int = 8
    width: int = 8
    max_ticks: int = 128
    respawn_delay: int = 0
    initial_placement: str = "frontier"
    rules_version: str = "territory_v1"

    def __post_init__(self):
        """맵·시간·부활 설정을 검증한다. 딜레이 0은 사망 틱 후반 부활이다."""
        for name in ("height", "width", "max_ticks", "respawn_delay"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.height < 3 or self.width < 2 or self.width % 2:
            raise ValueError("height >= 3 and even width >= 2 are required")
        if self.max_ticks < 1 or self.respawn_delay < 0:
            raise ValueError("max_ticks must be positive; respawn_delay >= 0")
        if self.initial_placement not in ("frontier", "random"):
            raise ValueError("initial_placement must be frontier or random")
        if self.rules_version != "territory_v1":
            raise ValueError("unsupported rules_version; v0 action logs require migration")


@dataclass(frozen=True)
class ExperimentConfig:
    decision_strategy: str = "independent"
    reward_mode: str = "team"
    team_signal: str = "area"
    capture_weight: float = 1.0
    area_weight: float = 1.0
    win_reward: float = 1.0
    loss_reward: float = 0.0
    draw_reward: float = 0.0
    include_terminal_reward: bool = False

    def __post_init__(self):
        """전략과 보상을 독립적으로 검증한다. 실행 기본값은 실험 결과나 확정 계수를 뜻하지 않는다."""
        if self.decision_strategy not in ("independent", "sequential", "feedback"):
            raise ValueError("unknown decision_strategy")
        if self.reward_mode not in ("individual", "team"):
            raise ValueError("reward_mode must be individual or team")
        if self.team_signal not in ("net", "gross", "area", "terminal"):
            raise ValueError("unknown team_signal")
        for name in ("capture_weight", "area_weight", "win_reward", "loss_reward", "draw_reward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.capture_weight <= 0 or self.area_weight < 0 or self.win_reward < 0 or self.loss_reward > 0:
            raise ValueError("capture_weight > 0; area_weight/win_reward >= 0; loss_reward <= 0")
        if type(self.include_terminal_reward) is not bool:
            raise ValueError("include_terminal_reward must be a bool")
