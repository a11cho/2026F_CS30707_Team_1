from dataclasses import asdict, dataclass
import itertools


@dataclass(frozen=True)
class RunConfig:
    """실험 예산·환경·보상/관측 그리드·평가 조건을 한곳에서 정의합니다."""
    algorithms: tuple = ("mc", "sarsa")
    seeds: tuple = (7, 19, 31, 43, 59)
    episodes: int = 80
    ticks: int = 32
    height: int = 6
    width: int = 6
    radius: int = 2
    respawn_delay: int = 2
    learner_count: int = 1
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    validation_pairs: int = 10
    test_pairs: int = 30
    train_opponent: str = "random"
    eval_opponents: tuple = ("random", "expand", "aggressive", "defend")
    observation_modes: tuple = ("local", "aware")
    reward_specs: tuple = (("individual", "gross", 0.0), ("individual", "gross", 1.0),
                           ("team", "net", 0.0), ("team", "net", 1.0),
                           ("team", "gross", 0.0), ("team", "gross", 1.0),
                           ("team", "area", 0.0), ("team", "area", 1.0),
                           ("team", "terminal", 1.0))

    def validate(self):
        """환경 생성과 학습 전에 설정의 범위와 지원 여부를 검사합니다."""
        import math
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("중복되지 않는 학습 seed를 지정하세요.")
        for name in ("episodes", "ticks", "validation_pairs", "test_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 값은 양의 정수여야 합니다.")
        if self.learner_count not in (1, 3):
            raise ValueError("learner_count는 1 또는 3이어야 합니다.")
        if not all(math.isfinite(v) for v in (self.alpha, self.gamma, self.epsilon_start, self.epsilon_end)):
            raise ValueError("학습 파라미터에 NaN 또는 무한대가 있습니다.")
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("0 < alpha <= 1, 0 <= gamma <= 1이어야 합니다.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("0 <= epsilon_end <= epsilon_start <= 1이어야 합니다.")
        if not self.algorithms or set(self.algorithms) - {"mc", "sarsa"}:
            raise ValueError("지원하지 않는 알고리즘입니다.")
        if not self.observation_modes or set(self.observation_modes) - {"local", "aware"}:
            raise ValueError("지원하지 않는 관측 모드입니다.")
        styles = {"random", "expand", "aggressive", "defend", "noop"}
        if self.train_opponent not in styles or not self.eval_opponents or set(self.eval_opponents) - styles:
            raise ValueError("지원하지 않는 상대 정책입니다.")
        if not self.reward_specs:
            raise ValueError("보상 그리드가 비어 있습니다.")
        from territory_rl.config import ExperimentConfig, GameConfig
        GameConfig(self.height, self.width, self.ticks, self.respawn_delay, self.radius)
        for reward, signal, bonus in self.reward_specs:
            ExperimentConfig("local", reward, signal, bonus)
            if reward == "team" and signal == "terminal" and bonus == 0:
                raise ValueError("항상 0인 terminal 보상 설정은 제외합니다.")
        return self

    def conditions(self):
        """중복을 제거한 관측·보상 조합을 ExperimentConfig로 순서대로 생성합니다."""
        from territory_rl.config import ExperimentConfig
        seen = set()
        for obs, (reward, signal, bonus) in itertools.product(self.observation_modes, self.reward_specs):
            signal = "gross" if reward == "individual" else signal
            key = (obs, reward, signal, bonus)
            if key not in seen:
                seen.add(key)
                yield ExperimentConfig(*key)

    def epsilon(self, episode):
        """0부터 시작하는 에피소드 번호에 해당하는 선형 탐험률을 반환합니다."""
        fraction = episode / max(1, self.episodes - 1)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def condition_id(experiment):
    return f"{experiment.observation_mode}_{experiment.reward_mode}_{experiment.team_signal}_b{experiment.terminal_bonus:g}"
