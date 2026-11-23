from dataclasses import dataclass, replace

from .search import FINE_SEARCH_BUDGETS, FINE_TACTICAL_BUDGETS, TACTICAL_EXPLORATION_BUDGETS, TACTICAL_LEAF_BUDGETS


@dataclass(frozen=True)
class Lesson:
    opponent: str
    max_ticks: int

    def game(self, base):
        return replace(base, max_ticks=self.max_ticks)


LESSONS = (
    Lesson("noop", 32),
    Lesson("random", 48),
    Lesson("search_opening", 64),
    Lesson("search_intro", 64),
    Lesson("search_tactical", 96),
    Lesson("search_strong", 128),
)
LESSONS_WITH_INTERMEDIATE_SEARCH = (*LESSONS[:4], Lesson("search_intermediate", 80), *LESSONS[4:])
INTERMEDIATE_TRAINING_OPPONENTS = (*FINE_SEARCH_BUDGETS, "search_intermediate")
TACTICAL_TRAINING_OPPONENTS = (*FINE_TACTICAL_BUDGETS, "search_tactical")
TACTICAL_SCORE_FLOOR = 0.25


class Curriculum:
    def __init__(self, level, interval, target, lessons=LESSONS, *, fine_search=False, fine_tactical=False,
                 leaf_budget=False, tactical_exploration=False, exploration_schedule=()):
        self.lessons = lessons
        self.level = level
        self.interval = interval
        self.target = target
        self.fine_search = fine_search
        self.fine_tactical = fine_tactical
        self.tactical_opponents = ()
        self.calibration_opponents = ()
        if tactical_exploration:
            self.tactical_opponents = tuple(TACTICAL_EXPLORATION_BUDGETS)
            self.calibration_opponents = self.tactical_opponents[::2]
        elif leaf_budget:
            self.tactical_opponents = self.calibration_opponents = tuple(TACTICAL_LEAF_BUDGETS)
        self.tactical_schedule = tuple(f"search_tactical_random_{percent}" for percent in exploration_schedule)
        self.tactical_level = self.tactical_opponents.index(self.tactical_schedule[0]) if self.tactical_schedule else 0
        self.stage_ticks = 0
        self.next_check = interval

    @property
    def lesson(self):
        return self.lessons[self.level]

    @property
    def uses_adaptive_tactical(self):
        return bool(self.tactical_opponents) and self.lesson.opponent == "search_tactical"

    @property
    def tactical_opponent(self):
        return self.tactical_opponents[self.tactical_level]

    def select_tactical_level(self, scores):
        tested = [(i, scores[name]) for i, name in enumerate(self.tactical_opponents) if name in scores]
        candidates = [(i, score) for i, score in tested if TACTICAL_SCORE_FLOOR <= score < self.target]
        if candidates:
            self.tactical_level = min(candidates, key=lambda row: abs(row[1] - 0.5))[0]
        else:
            self.tactical_level = next((i for i, score in tested if score < self.target), tested[-1][0])

    def adjust_tactical_level(self, score):
        if self.tactical_schedule:
            interval = min(self.stage_ticks // self.interval, len(self.tactical_schedule) - 1)
            self.tactical_level = self.tactical_opponents.index(self.tactical_schedule[interval])
        elif score >= self.target:
            self.tactical_level = min(self.tactical_level + 1, len(self.tactical_opponents) - 1)
        elif score < TACTICAL_SCORE_FLOOR:
            self.tactical_level = max(0, self.tactical_level - 1)

    def opponents(self):
        opponent = self.lesson.opponent
        if self.level == 0:
            return (opponent,), (1.0,)
        if self.uses_adaptive_tactical:
            return (self.tactical_opponent,), (1.0,)
        if self.fine_search and opponent == "search_intermediate":
            opponent = self.training_opponent(INTERMEDIATE_TRAINING_OPPONENTS)
        if self.fine_tactical and opponent == "search_tactical":
            opponent = self.training_opponent(TACTICAL_TRAINING_OPPONENTS)
        return (opponent, self.lessons[self.level - 1].opponent), (0.8, 0.2)

    def training_opponent(self, tiers):
        return tiers[min(self.stage_ticks // self.interval, len(tiers) - 1)]

    def check_due(self):
        return self.stage_ticks >= self.next_check

    def check(self, score):
        """Move by one lesson when the current checkpoint meets the paired evaluation target."""
        self.next_check = self.stage_ticks + self.interval
        if score < self.target or self.level == len(self.lessons) - 1 or self.uses_adaptive_tactical:
            return False
        self.level += 1
        self.stage_ticks = 0
        self.next_check = self.interval
        return True
