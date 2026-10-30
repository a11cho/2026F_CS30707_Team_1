from dataclasses import asdict
import json
from .controller import FrontierController
from .environment import TerritoryEnvironment
from .keyboard import KeyboardController
from .manager import EpisodeManager
from .models import AGENT_IDS, Action
from .visualization import TkRenderer


class KeyboardDemo:
    def __init__(self, config=None, experiment=None, seed=7, manual_all=False, interval_ms=250, root=None, log_path=None):
        import tkinter as tk
        self.seed, self.manual_all, self.interval_ms = seed, manual_all, interval_ms
        self.keyboards = {aid: KeyboardController() for aid in AGENT_IDS}
        self.selected = AGENT_IDS[0]
        self.running, self.closed = False, False
        self.after_id = None
        self.renderer = TkRenderer(root)
        self.root = self.renderer.root
        self.env = TerritoryEnvironment(config, experiment)
        self.manager = EpisodeManager(self.env, self._controllers())
        self.manager.reset(seed)
        self.last_result = None
        self.log = None
        if log_path is not None:
            from pathlib import Path
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log = path.open("w", encoding="utf-8")
        panel = tk.Frame(self.root, bg="#121826")
        panel.pack(side="right", fill="both", padx=(0, 16), pady=14)
        def label(text, **kwargs):
            widget = tk.Label(panel, text=text, bg="#121826", fg="#dce6f5", justify="left", anchor="w", **kwargs)
            widget.pack(fill="x", pady=5)
            return widget
        label("TERRITORY CONTROL", font=("Segoe UI", 17, "bold"))
        label("Debug view: full map / simultaneous 3v3", font=("Segoe UI", 10))
        self.status = label("", font=("Consolas", 11), width=36)
        self.roster = label("", font=("Consolas", 10), width=36)
        label("W A S D   Move\nArrows / I J K L   Attack\nSpace   Queue NOOP\n1-6 / Tab   Select agent\nEnter   Step one tick\nP   Play / pause   |   R   Reset\nEsc   Close", font=("Segoe UI", 11))
        for name, callback in (("Step  [Enter]", self.step), ("Play / Pause  [P]", self.toggle), ("Reset  [R]", self.reset)):
            tk.Button(panel, text=name, command=callback, takefocus=False, bg="#26374f", fg="white", relief="flat").pack(fill="x", pady=3)
        self.events_label = label("", font=("Consolas", 9), wraplength=340, height=8)
        self.root.bind("<KeyPress>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._write_reset()
        self.refresh()

    def _controllers(self):
        bots = {aid: FrontierController(self.seed+i, "aggressive" if i >= 3 else "expand") for i, aid in enumerate(AGENT_IDS)}
        return {aid: self.keyboards[aid] if self.manual_all or aid == self.selected else bots[aid] for aid in AGENT_IDS}

    def _write_reset(self):
        if self.log:
            self.log.write(json.dumps({"type": "reset", "seed": self.seed, "game": asdict(self.env.config),
                                       "experiment": asdict(self.env.experiment)}) + "\n")
            self.log.flush()

    def on_key(self, event):
        key = event.keysym.lower()
        if key == "escape":
            self.close()
            return "break"
        if key in ("return", "kp_enter"):
            self.step()
        elif key == "p":
            self.toggle()
        elif key == "r":
            self.reset()
        elif key in ("1", "2", "3", "4", "5", "6", "tab"):
            previous = self.selected
            self.selected = AGENT_IDS[(AGENT_IDS.index(previous)+1) % 6] if key == "tab" else AGENT_IDS[int(key)-1]
            if not self.manual_all:
                self.keyboards[previous].clear()
                self.manager.controllers[previous] = FrontierController(self.seed+AGENT_IDS.index(previous))
                self.manager.controllers[self.selected] = self.keyboards[self.selected]
        else:
            self.keyboards[self.selected].feed_key(key)
        if not self.closed:
            self.refresh()
        return "break"

    def step(self):
        if self.closed or self.env.terminated:
            self.pause()
            return
        self.last_result = self.manager.step()
        if self.log:
            record = self.last_result.to_dict()
            record.update(type="step", actions={a: int(v) for a, v in self.manager.last_actions.items()})
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()
        if self.env.terminated:
            self.pause()
        self.refresh()

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle(self):
        if self.running:
            self.pause()
        elif not self.env.terminated:
            self.running = True
            self.after_id = self.root.after(self.interval_ms, self._timer)
        self.refresh()

    def _timer(self):
        self.after_id = None
        if self.running and not self.closed:
            self.step()
            if self.running:
                self.after_id = self.root.after(self.interval_ms, self._timer)

    def reset(self):
        self.pause()
        for keyboard in self.keyboards.values():
            keyboard.clear()
        self.manager.controllers = self._controllers()
        self.manager.reset(self.seed)
        self.last_result = None
        self._write_reset()
        self.refresh()

    def refresh(self):
        snapshot = self.env.snapshot()
        self.renderer.draw(snapshot, self.selected)
        pending = self.keyboards[self.selected].pending
        mode = "FINISHED" if snapshot.terminated else ("RUNNING" if self.running else "PAUSED")
        self.status.config(text=f"{mode}   Tick {snapshot.tick}/{snapshot.max_ticks}\nRED {snapshot.areas[0]}   BLUE {snapshot.areas[1]}\nSelected: {self.selected}\nQueued: {pending.name if pending is not None else 'NOOP (default)'}\nMode: {'all manual' if self.manual_all else 'selected human + 5 bots'}")
        lines = []
        for index, a in enumerate(snapshot.agents):
            state = str(a.position) if a.alive else f"dead; due {a.respawn_at}"
            ready = bool(self.env.action_mask(a.agent_id)[5:].any())
            queued = self.keyboards[a.agent_id].pending
            short_actions = ("wait", "up", "down", "left", "right", "atkU", "atkD", "atkL", "atkR")
            suffix = f" [{short_actions[int(queued)]}]" if queued is not None else ""
            lines.append(f"{index+1} {a.agent_id:6} {state:15} {'ATK' if ready else '---'}{suffix}")
        self.roster.config(text="\n".join(lines))
        recent = [] if self.last_result is None else [e for e in self.last_result.events if e.kind != "move" or e.reason != "success"]
        self.events_label.config(text="Last tick events\n" + ("\n".join(f"{e.kind}: {e.agent_id or e.position} {e.reason}" for e in recent[-7:]) or "No events"))

    def close(self):
        if not self.closed:
            self.pause()
            self.closed = True
            if self.log:
                self.log.close()
            self.renderer.close()

    def run(self):
        self.root.mainloop()

