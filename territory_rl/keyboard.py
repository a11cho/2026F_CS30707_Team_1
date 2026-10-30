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
        self.keymap = dict(self.KEYMAP if keymap is None else keymap)
        self.pending = None

    def feed_key(self, key):
        key = key.lower()
        if key not in self.keymap:
            return False
        self.pending = Action.parse(self.keymap[key])
        return True

    def clear(self):
        self.pending = None

    def act(self, observation):
        action = self.pending if self.pending is not None else Action.NOOP
        self.clear()
        # Let the core report rejected actions, including cooldown/death/boundary.
        return action

