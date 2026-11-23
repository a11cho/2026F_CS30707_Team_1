"""원본 환경의 고정 상대 컨트롤러를 에피소드별로 초기화합니다."""

import numpy as np
from territory_rl.controller import FrontierController, NoopController, RandomController


def episode_seed(master, *indices):
    return int(np.random.SeedSequence([master, *indices]).generate_state(1)[0])


def make_opponents(style, agents, seed):
    controllers = {}
    for slot, agent in enumerate(agents):
        policy_seed = episode_seed(seed, 1, slot)
        if style == "noop":
            controller = NoopController()
        elif style == "random":
            controller = RandomController(policy_seed)
        else:
            controller = FrontierController(policy_seed, style)
        controllers[agent] = controller
    return controllers
