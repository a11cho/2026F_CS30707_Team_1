"""실험별 TOML 설정과 인용부호가 필요 없는 명령행 덮어쓰기."""

import copy
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


OPPONENTS = ('random', 'expand', 'aggressive', 'defend', 'noop')
RESUME_FIELDS = {'train.updates', 'train.checkpoint_every', 'runtime.device', 'runtime.threads'}


@dataclass
class RunConfig:
    game: dict = field(default_factory=lambda: dict(height=3, width=10, max_ticks=128))
    reward: dict = field(default_factory=lambda: dict(
        mode='A-T', signal='area_share', team_weight=1.0, capture_weight=1.0))
    model: dict = field(default_factory=lambda: dict(hidden_size=128, previous_actions=False))
    train: dict = field(default_factory=lambda: dict(seed=7, updates=100, episodes_per_update=8, checkpoint_every=5))
    ppo: dict = field(default_factory=lambda: dict(
        learning_rate=0.0003, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
        entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, epochs=4, minibatch_size=256))
    evaluation: dict = field(default_factory=lambda: dict(
        seed=20261120, opponents={'expand': 80, 'random': 80}, deterministic=True,
        mirror_to_training_side=True))
    runtime: dict = field(default_factory=lambda: dict(device='cpu', threads=1))

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=(), *, base=None, resume=False):
    values = copy.deepcopy(base) if base is not None else RunConfig().to_dict()
    if base is not None:
        # 이전 체크포인트는 면적 변화량으로 학습했습니다.
        values['reward'].setdefault('signal','gap_delta')
        values['evaluation'].setdefault('mirror_to_training_side',True)
    if path:
        with Path(path).open('rb') as stream:
            sections = tomllib.load(stream)
        # 기존 실험의 설정 파일을 새 보상으로 묵시적으로 변경하지 않습니다.
        if base is None and 'reward' in sections and 'signal' not in sections['reward']:
            values['reward']['signal']='gap_delta'
        for section, fields in sections.items():
            if section not in values or not isinstance(fields, dict):
                raise ValueError(f'알 수 없는 설정 구역: {section}')
            for key, value in fields.items():
                _set(values, f'{section}.{key}', value)
    for text in overrides:
        name, separator, raw = text.partition('=')
        if not separator:
            raise ValueError('--set은 구역.항목=값 형식입니다.')
        if resume and name not in RESUME_FIELDS:
            raise ValueError(f'이어 학습에서 변경할 수 없는 설정: {name}')
        try:
            value = tomllib.loads(f'value = {raw}')['value']
        except tomllib.TOMLDecodeError:
            value = raw
        _set(values, name, value)
    cfg = RunConfig(**values)
    validate(cfg)
    return cfg


def _set(values, name, value):
    section, dot, key = name.partition('.')
    if not dot or section not in values or '.' in key or not key:
        raise ValueError(f'알 수 없는 설정: {name}')
    if section != 'game' and key not in values[section]:
        raise ValueError(f'알 수 없는 설정: {name}')
    values[section][key] = value


def validate(cfg):
    for fields, minima in (
        (cfg.model, {'hidden_size': 1}),
        (cfg.train, {'seed': 0, 'updates': 1, 'episodes_per_update': 1, 'checkpoint_every': 1}),
        (cfg.ppo, {'epochs': 1, 'minibatch_size': 1}),
        (cfg.evaluation, {'seed': 0}), (cfg.runtime, {'threads': 1}),
    ):
        for key, minimum in minima.items():
            if type(fields[key]) is not int or fields[key] < minimum:
                raise ValueError(f'{key}는 {minimum} 이상의 정수여야 합니다.')
    for fields, names in (
        (cfg.reward, ('team_weight', 'capture_weight')),
        (cfg.ppo, ('learning_rate', 'gamma', 'gae_lambda', 'clip_ratio', 'entropy_coef',
                   'value_coef', 'max_grad_norm')),
    ):
        for key in names:
            value = fields[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key}는 유한한 음이 아닌 값이어야 합니다.')
    for key in ('learning_rate', 'clip_ratio', 'max_grad_norm'):
        if cfg.ppo[key] <= 0:
            raise ValueError(f'{key}는 양수여야 합니다.')
    if cfg.ppo['gamma'] > 1 or cfg.ppo['gae_lambda'] > 1:
        raise ValueError('gamma와 gae_lambda는 0~1이어야 합니다.')
    if cfg.reward['mode'] not in ('A-T', 'A-S'):
        raise ValueError('reward.mode는 A-T 또는 A-S입니다.')
    if cfg.reward['signal'] not in ('area_share','gap_delta'):
        raise ValueError('reward.signal은 area_share 또는 gap_delta입니다.')
    for fields, key in ((cfg.model, 'previous_actions'), (cfg.evaluation, 'deterministic'),
                        (cfg.evaluation,'mirror_to_training_side')):
        if type(fields[key]) is not bool:
            raise ValueError(f'{key}는 true/false여야 합니다.')
    validate_opponents(cfg.evaluation['opponents'])


def validate_opponents(opponents):
    if not isinstance(opponents, dict) or not opponents:
        raise ValueError('평가 상대와 경기 수가 필요합니다.')
    for name, count in opponents.items():
        if name not in OPPONENTS or type(count) is not int or count < 2 or count % 2:
            raise ValueError(f'평가 상대는 {OPPONENTS}, 경기 수는 2 이상의 짝수여야 합니다: {name}={count}')


def parse_opponents(items):
    result = {}
    for text in items:
        name, separator, raw = text.partition('=')
        if not separator or name in result:
            raise ValueError(f'중복 없는 --opponent 이름=경기수 형식이 필요합니다: {text}')
        try:
            result[name] = int(raw)
        except ValueError as exc:
            raise ValueError(f'경기 수는 정수여야 합니다: {text}') from exc
    validate_opponents(result)
    return result
