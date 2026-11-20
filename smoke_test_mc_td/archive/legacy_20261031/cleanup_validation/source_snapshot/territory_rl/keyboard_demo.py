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
        """키보드 컨트롤러·환경·manager·Tk 테스트 화면을 연결해 일시정지 상태로 시작한다.

        manual_all이면 여섯 명 모두 수동, 아니면 선택한 한 명만 수동이고 나머지는 scripted다.
        interval_ms는 자동 진행 간격이며, log_path가 있으면 JSONL 파일을 쓰기 모드로 연다.
        생성은 초기화까지만 수행하고 이벤트 루프는 run에서 시작한다.
        """
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
            """공통 배경·글자 스타일을 가진 패널 Label을 생성·배치하고 위젯을 반환한다."""
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
        """현재 선택·수동 모드에 맞는 ID별 컨트롤러 사전을 만든다.
        키보드 인스턴스는 재사용하고 scripted 컨트롤러는 지정 시드로 새로 생성한다.
        """
        bots = {aid: FrontierController(self.seed+i, "aggressive" if i >= 3 else "expand") for i, aid in enumerate(AGENT_IDS)}
        return {aid: self.keyboards[aid] if self.manual_all or aid == self.selected else bots[aid] for aid in AGENT_IDS}

    def _write_reset(self):
        """로그가 열려 있으면 재생에 필요한 seed·게임 설정·실험 설정을 reset 레코드로 기록한다.
        즉시 flush하여 이후 행동 로그의 시작 상태를 보존한다.
        """
        if self.log:
            self.log.write(json.dumps({"type": "reset", "seed": self.seed, "game": asdict(self.env.config),
                                       "experiment": asdict(self.env.experiment)}) + "\n")
            self.log.flush()

    def on_key(self, event):
        """Tk KeyPress 이벤트를 행동 예약 또는 진행·선택·초기화·종료 명령으로 해석한다.

        이동/공격 키는 예약만 하며 Enter가 한 틱을 진행한다. 한 명 수동 모드에서
        조작 대상을 바꾸면 이전 대상을 scripted로 되돌리고 남은 예약 키를 비운다.
        반환값 'break'는 Tk의 후속 기본 키 처리를 막는다.
        """
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
        """manager를 한 번 실행하고 행동·결과 로그와 화면을 갱신한다.
        경기가 끝났거나 창이 닫혔으면 전이하지 않는다. 종료 틱에 도달하면 자동 진행을 멈춘다.
        """
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
        """자동 진행 플래그를 끄고 예약된 Tk 타이머가 있으면 취소한다. 게임 상태는 유지한다."""
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle(self):
        """자동 진행과 일시정지를 전환하고 상태 표시를 갱신한다. 종료한 경기는 재시작하지 않는다."""
        if self.running:
            self.pause()
        elif not self.env.terminated:
            self.running = True
            self.after_id = self.root.after(self.interval_ms, self._timer)
        self.refresh()

    def _timer(self):
        """Tk 타이머 호출 시 한 틱을 진행하고 계속 실행 중일 때만 다음 타이머를 예약한다.
        처리 시작 시 기존 타이머 ID를 비워 취소된 작업이나 중복 예약이 남지 않게 한다.
        """
        self.after_id = None
        if self.running and not self.closed:
            self.step()
            if self.running:
                self.after_id = self.root.after(self.interval_ms, self._timer)

    def reset(self):
        """자동 진행과 예약 입력을 비우고 같은 시드로 환경·scripted 컨트롤러를 다시 초기화한다.
        새 reset 로그를 기록하고 초기 화면을 표시한다.
        """
        self.pause()
        for keyboard in self.keyboards.values():
            keyboard.clear()
        self.manager.controllers = self._controllers()
        self.manager.reset(self.seed)
        self.last_result = None
        self._write_reset()
        self.refresh()

    def refresh(self):
        """현재 맵·선택 대상·예약 행동·생존/공격 상태·최근 사건을 화면에 표시한다.
        전역 맵을 보여 주는 디버그 화면이며 에이전트 관측 범위와는 다르다. 게임 틱은 진행하지 않는다.
        """
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
        """타이머를 중단하고 로그 파일과 창을 닫는다. 여러 번 호출해도 종료 처리는 한 번만 한다."""
        if not self.closed:
            self.pause()
            self.closed = True
            if self.log:
                self.log.close()
            self.renderer.close()

    def run(self):
        """Tk 이벤트 루프에 진입해 키보드·버튼·타이머 입력을 처리한다. 창이 닫히면 반환한다."""
        self.root.mainloop()

