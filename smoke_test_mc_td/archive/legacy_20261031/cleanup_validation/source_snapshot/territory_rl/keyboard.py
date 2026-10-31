from .models import Action


class KeyboardController:
    """GUI-independent temporary agent. A frontend feeds keys, manager consumes one action.

    Keys stage the NEXT action (latest key wins before a tick). No held-key polling,
    no blocking input, and no direct environment access. Multiple instances can be
    connected to independent keyboard/network frontends.
    """
    KEYMAP = {"w": Action.UP, "s": Action.DOWN, "a": Action.LEFT, "d": Action.RIGHT,
              "up": Action.ATTACK_UP, "down": Action.ATTACK_DOWN,
              "left": Action.ATTACK_LEFT, "right": Action.ATTACK_RIGHT,
              "i": Action.ATTACK_UP, "k": Action.ATTACK_DOWN,
              "j": Action.ATTACK_LEFT, "l": Action.ATTACK_RIGHT,
              "space": Action.NOOP, " ": Action.NOOP}

    def __init__(self, keymap=None):
        """키 이름→Action 매핑을 복사하고 다음 틱의 예약 행동을 비운다.
        keymap을 생략하면 WASD 이동·방향키/IJKL 공격·Space 정지를 사용한다.
        """
        self.keymap = dict(self.KEYMAP if keymap is None else keymap)
        self.pending = None

    def feed_key(self, key):
        """키 이름을 소문자로 정규화해 다음 행동으로 예약하고, 처리한 키이면 True를 반환한다.
        알 수 없는 키는 False를 반환한다. 한 틱 전에 여러 키가 오면 마지막 유효 키가 덮어쓴다.
        행동을 예약할 뿐 게임 틱이나 위치는 바꾸지 않는다.
        """
        key = key.lower()
        if key not in self.keymap:
            return False
        self.pending = Action.parse(self.keymap[key])
        return True

    def clear(self):
        """예약 행동을 버린다. reset이나 조작 대상 전환 시 이전 입력이 남지 않게 한다."""
        self.pending = None

    def act(self, observation):
        """예약 행동을 한 번 반환한 뒤 비우며, 예약이 없으면 NOOP을 반환한다.
        관측은 공통 인터페이스를 위해 받는다. 금지 행동도 코어가 거부 이벤트로 기록하도록 그대로 전달한다.
        """
        action = self.pending if self.pending is not None else Action.NOOP
        self.clear()
        # Let the core report rejected actions, including cooldown/death/boundary.
        return action

