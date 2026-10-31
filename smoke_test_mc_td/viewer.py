"""저장된 Q-table로 경기를 관전합니다. 학습·체크포인트 저장 기능은 호출하지 않습니다."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
from territory_rl import ExperimentConfig, GameConfig, TerritoryEnvironment
from territory_rl.controller import FrontierController, NoopController, RandomController
from territory_rl.models import AGENT_IDS, Action
from .encoding import encode
from .learning import load_models

ACTION_NAMES = ("정지", "위 이동", "아래 이동", "왼쪽 이동", "오른쪽 이동",
                "위 공격", "아래 공격", "왼쪽 공격", "오른쪽 공격")
MODE_NAMES = {"자동": "auto", "양 팀 모델": "both", "모델 팀 대 고정 상대": "fixed"}
OPPONENTS = ("random", "expand", "aggressive", "defend", "noop")


def _seed(master, domain, index=0):
    return int(np.random.SeedSequence([master, domain, index]).generate_state(1)[0])


class FrozenPolicy:
    """추론 전용 정책입니다. Q 배열과 사전을 읽기 전용으로 만들며 갱신 메서드가 없습니다."""
    def __init__(self, model):
        values = {}
        for key, array in model.q.items():
            if array.shape != (9,) or not np.isfinite(array).all():
                raise ValueError("체크포인트의 Q 배열 크기 또는 값이 올바르지 않습니다.")
            frozen = array.copy()
            frozen.flags.writeable = False
            values[key] = frozen
        self.q = MappingProxyType(values)
        self.algorithm = model.algorithm
        self.training_updates = model.updates

    def act(self, observation, rng):
        """local/aware 관측만 사용합니다. 최대값 동률과 미방문 key는 허용 행동에서 추첨합니다."""
        key = encode(observation)
        legal = np.flatnonzero(observation["action_mask"])
        if not len(legal):
            raise ValueError("허용 행동이 없는 관측입니다.")
        q = self.q.get(key)
        scores = np.zeros(len(legal)) if q is None else q[legal]
        best = legal[np.flatnonzero(scores == scores.max())]
        # 기존 TabularControl.act(epsilon=0)의 난수 소비 순서와 맞춥니다.
        rng.random()
        action = Action(int(rng.choice(best)))
        return action, dict(known=key in self.q, ties=len(best))


class ModelSimulation:
    """GUI와 분리된 추론 시뮬레이션입니다. 로딩과 reset 시에도 파일을 쓰지 않습니다."""
    def __init__(self, checkpoint, mode="auto", cohort=0, team=0, opponent="expand", seed=7, policy_seed=None):
        if mode not in ("auto", "both", "fixed") or cohort not in (0, 1) or team not in (0, 1):
            raise ValueError("관전 모드·팀 묶음·진영 설정을 확인하세요.")
        if opponent not in OPPONENTS:
            raise ValueError("지원하지 않는 고정 상대입니다.")
        self.checkpoint = Path(checkpoint).resolve()
        models, self.metadata = load_models(self.checkpoint)
        if len(models) not in (1, 3, 6):
            raise ValueError("모델이 1개, 3개 또는 6개인 체크포인트가 필요합니다.")
        if self.metadata.get("encoding") != "compact_v1":
            raise ValueError("compact_v1 관측 표현의 체크포인트만 지원합니다.")
        cfg = self.metadata["config"]
        self.game = GameConfig(height=cfg["height"], width=cfg["width"], max_ticks=cfg["ticks"],
                               respawn_delay=cfg["respawn_delay"], observation_radius=cfg["radius"])
        self.experiment = ExperimentConfig(**self.metadata["experiment"])
        self.policies = tuple(FrozenPolicy(model) for model in models)
        self.mode = ("both" if len(models) == 6 else "fixed") if mode == "auto" else mode
        if self.mode == "both" and len(models) != 6:
            raise ValueError("양 팀 모델 모드에는 모델 6개가 필요합니다. 이 파일은 한쪽 팀 학습 결과입니다.")
        if cohort == 1 and len(models) != 6:
            raise ValueError("이 파일에는 두 번째 팀 묶음이 없습니다.")
        self.cohort, self.team, self.opponent = cohort, team, opponent
        self.env = TerritoryEnvironment(self.game, self.experiment)
        self.reset(seed, policy_seed)
        key_size = len(encode(self.observations[AGENT_IDS[0]]))
        if any(len(key) != key_size for policy in self.policies for key in policy.q):
            raise ValueError("현재 인코딩과 체크포인트의 key 길이가 다릅니다.")

    def reset(self, seed=None, policy_seed=None):
        """같은 두 seed로 초기화하면 부활과 행동 동률 처리까지 동일하게 재생합니다."""
        if seed is None and policy_seed is None:
            policy_seed = getattr(self, "policy_seed", None)
        seed = getattr(self, "seed", 7) if seed is None else seed
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("환경 seed는 0 이상 2^32 미만의 정수여야 합니다.")
        if policy_seed is not None and (type(policy_seed) is not int or not 0 <= policy_seed < 2**32):
            raise ValueError("정책 seed는 0 이상 2^32 미만의 정수여야 합니다.")
        self.seed = seed
        self.policy_seed = _seed(seed, 11) if policy_seed is None else policy_seed
        self.env.reset(seed=self.seed)
        ordered_ids = AGENT_IDS[self.team*3:] + AGENT_IDS[:self.team*3]
        if self.mode == "both":
            selected = tuple(enumerate(self.policies))
            ids = ordered_ids
        else:
            start = self.cohort*3 if len(self.policies) == 6 else 0
            selected = tuple(enumerate(self.policies[start:start+3], start))
            ids = AGENT_IDS[self.team*3:self.team*3+len(selected)]
        self.assigned = {aid: policy for aid, (_, policy) in zip(ids, selected)}
        self.roles = {aid: role for aid, (role, _) in zip(ids, selected)}
        self.rngs = {aid: np.random.default_rng(_seed(self.policy_seed, 1, i)) for i, aid in enumerate(ids)}
        self.bots = {}
        for i, aid in enumerate(AGENT_IDS):
            if aid in self.assigned:
                continue
            style = self.opponent if i//3 != self.team else "expand"
            bot_seed = _seed(self.policy_seed, 2, i)
            self.bots[aid] = (RandomController(bot_seed) if style == "random" else NoopController() if style == "noop"
                              else FrontierController(bot_seed, style))
        self.observations = {aid: self.env.observe(aid) for aid in AGENT_IDS}
        self.last_actions, self.last_decisions = {}, {}
        self.last_result = None
        self.returns = dict.fromkeys(AGENT_IDS, 0.0)
        self.counts = Counter()
        return self.env.snapshot()

    def step(self):
        """여섯 행동을 모아 환경을 한 틱만 진행합니다. 학습은 하지 않습니다."""
        if self.env.terminated:
            return None
        actions, decisions = {}, {}
        for aid, policy in self.assigned.items():
            actions[aid], decisions[aid] = policy.act(self.observations[aid], self.rngs[aid])
        for aid, bot in self.bots.items():
            actions[aid] = bot.act(self.observations[aid])
        if set(actions) != set(AGENT_IDS) or any(not self.observations[a]["action_mask"][v] for a, v in actions.items()):
            raise RuntimeError("행동 마스크 또는 에이전트 목록과 일치하지 않는 행동입니다.")
        result = self.env.step(actions)
        self.last_result, self.last_actions, self.last_decisions = result, actions, decisions
        for aid, reward in result.rewards.items():
            self.returns[aid] += reward
        for event in result.events:
            self.counts[event.kind] += 1
        self.observations = {aid: self.env.observe(aid) for aid in AGENT_IDS}
        return result

    def summary(self):
        snapshot = self.env.snapshot()
        return dict(checkpoint=str(self.checkpoint), mode=self.mode, seed=self.seed, policy_seed=self.policy_seed,
                    tick=snapshot.tick, max_ticks=snapshot.max_ticks, areas=list(snapshot.areas),
                    terminated=snapshot.terminated, rewards=dict(self.returns), events=dict(self.counts),
                    roles=dict(self.roles), training=False)


class SimulationViewer:
    """기존 TkRenderer를 재사용하는 관전 창입니다. 타이머만 화면 재생 속도를 결정합니다."""
    def __init__(self, simulation=None, root=None, interval_ms=250):
        import tkinter as tk
        from tkinter import ttk
        from territory_rl.visualization import TkRenderer
        if not 20 <= interval_ms <= 2000:
            raise ValueError("재생 간격은 20~2000ms여야 합니다.")
        self.root = root if root is not None else tk.Tk()
        self.renderer = TkRenderer(self.root, tile_size=64)
        self.root.title("학습 모델 관전 · Q-table 업데이트 없음")
        self.simulation = simulation
        self.running, self.closed, self.after_id = False, False, None
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(side="right", fill="both", expand=True)
        ttk.Label(panel, text="학습 모델 시뮬레이션", font=("맑은 고딕", 15, "bold")).pack(anchor="w")
        ttk.Label(panel, text="전체 맵은 관전자 화면입니다. 모델은 자신의 local/aware 관측만 받습니다.", wraplength=520).pack(anchor="w", pady=5)
        ttk.Button(panel, text="체크포인트 열기…", command=self.open_file).pack(fill="x")
        self.source = ttk.Label(panel, text="model.json.gz 파일을 선택하세요.", wraplength=520)
        self.source.pack(anchor="w", pady=5)
        controls = ttk.Frame(panel)
        controls.pack(fill="x")
        ttk.Button(controls, text="재생 / 일시정지 [Space]", command=self.toggle).pack(side="left")
        ttk.Button(controls, text="한 틱 [Enter]", command=self.step_once).pack(side="left", padx=4)
        ttk.Button(controls, text="같은 seed로 초기화 [R]", command=self.reset).pack(side="left")
        self.interval = tk.DoubleVar(value=interval_ms)
        ttk.Label(panel, text="틱 간격(ms): 작을수록 빠르게 재생합니다.").pack(anchor="w", pady=(7, 0))
        self.speed_label = ttk.Label(panel)
        self.speed_label.pack(anchor="w")
        ttk.Scale(panel, from_=20, to=2000, variable=self.interval, command=self.change_speed).pack(fill="x")
        settings = ttk.LabelFrame(panel, text="관전 설정 · 변경 후 적용 버튼을 누르세요", padding=6)
        settings.pack(fill="x", pady=6)
        self.mode = tk.StringVar(value="자동")
        self.cohort = tk.StringVar(value="0")
        self.team = tk.StringVar(value="빨강")
        self.opponent = tk.StringVar(value="expand")
        self.seed = tk.StringVar(value="7")
        self.policy_seed = tk.StringVar(value="")
        options = [("대결 모드",self.mode,tuple(MODE_NAMES)),("고정 상대 모드의 모델 묶음",self.cohort,("0","1")),
                   ("첫 모델 묶음 진영",self.team,("빨강","파랑")),("고정 상대",self.opponent,OPPONENTS)]
        for row,(name,variable,values) in enumerate(options):
            ttk.Label(settings,text=name).grid(row=row,column=0,sticky="w")
            ttk.Combobox(settings,textvariable=variable,values=values,state="readonly",width=24).grid(row=row,column=1,sticky="ew")
        for row,(name,variable) in enumerate((("환경 seed",self.seed),("정책 seed (빈칸이면 자동)",self.policy_seed)),len(options)):
            ttk.Label(settings,text=name).grid(row=row,column=0,sticky="w")
            ttk.Entry(settings,textvariable=variable,width=26).grid(row=row,column=1,sticky="ew")
        ttk.Button(settings,text="설정 적용 · 처음부터",command=self.apply_settings).grid(row=6,column=0,columnspan=2,sticky="ew",pady=4)
        self.status = ttk.Label(panel, wraplength=520)
        self.status.pack(anchor="w", pady=5)
        from tkinter.scrolledtext import ScrolledText
        self.roster = ScrolledText(panel, height=12, width=70, font=("맑은 고딕", 9), wrap="word", state="disabled")
        self.roster.pack(fill="x")
        self.events = ScrolledText(panel, height=5, width=70, font=("맑은 고딕", 9), wrap="word", state="disabled")
        self.events.pack(fill="x", pady=(6, 0))
        self.root.bind("<space>",lambda event:self.shortcut(event,self.toggle))
        self.root.bind("<Return>",lambda event:self.shortcut(event,self.step_once))
        self.root.bind("<Key-r>",lambda event:self.shortcut(event,self.reset))
        self.root.bind("<Escape>",lambda event:self.close())
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        if simulation is not None:
            self.sync_settings()
        self.change_speed()
        self.refresh()

    def shortcut(self, event, callback):
        """입력칸·콤보박스·텍스트에서 타이핑할 때 관전 단축키가 실행되지 않게 합니다."""
        if event.widget.winfo_class() not in ("Entry","TEntry","TCombobox","Text"):
            callback()
            return "break"

    def sync_settings(self):
        sim = self.simulation
        self.mode.set("양 팀 모델" if sim.mode == "both" else "모델 팀 대 고정 상대")
        self.cohort.set(str(sim.cohort))
        self.team.set("빨강" if sim.team == 0 else "파랑")
        self.opponent.set(sim.opponent)
        self.seed.set(str(sim.seed))
        self.policy_seed.set(str(sim.policy_seed))

    @staticmethod
    def _text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0","end")
        widget.insert("1.0",value)
        widget.configure(state="disabled")

    def open_file(self):
        from tkinter import filedialog, messagebox
        self.pause()
        path = filedialog.askopenfilename(parent=self.root,title="학습 체크포인트 선택",filetypes=[("압축 JSON 모델","*.json.gz"),("모든 파일","*")])
        if not path:
            return
        try:
            new_sim = ModelSimulation(path,seed=int(self.seed.get()),
                                      policy_seed=int(self.policy_seed.get()) if self.policy_seed.get().strip() else None)
        except Exception as exc:
            messagebox.showerror("모델을 불러오지 못했습니다",str(exc),parent=self.root)
            return
        self.simulation = new_sim
        self.sync_settings()
        self.refresh()

    def apply_settings(self):
        from tkinter import messagebox
        self.pause()
        if self.simulation is None:
            return
        try:
            sim = ModelSimulation(self.simulation.checkpoint,mode=MODE_NAMES[self.mode.get()],
                                  cohort=int(self.cohort.get()),team=0 if self.team.get()=="빨강" else 1,
                                  opponent=self.opponent.get(),seed=int(self.seed.get()),
                                  policy_seed=int(self.policy_seed.get()) if self.policy_seed.get().strip() else None)
        except Exception as exc:
            messagebox.showerror("관전 설정 오류",str(exc),parent=self.root)
            return
        self.simulation = sim
        self.sync_settings()
        self.refresh()

    def reset(self):
        self.pause()
        if self.simulation is not None:
            self.simulation.reset(self.simulation.seed,self.simulation.policy_seed)
            self.refresh()

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle(self):
        if self.closed or self.simulation is None:
            return
        if self.running:
            self.pause()
        elif not self.simulation.env.terminated:
            self.running = True
            self.after_id = self.root.after(int(self.interval.get()),self._timer)
        self.refresh()

    def _timer(self):
        self.after_id = None
        if self.closed or not self.running:
            return
        self._advance()
        if self.running:
            self.after_id = self.root.after(int(self.interval.get()),self._timer)

    def _advance(self):
        if self.simulation is None or self.closed:
            return
        try:
            self.simulation.step()
        except Exception as exc:
            from tkinter import messagebox
            self.pause()
            messagebox.showerror("시뮬레이션 오류",str(exc),parent=self.root)
            return
        if self.simulation.env.terminated:
            self.pause()
        self.refresh()

    def step_once(self):
        self.pause()
        self._advance()

    def change_speed(self, value=None):
        self.speed_label.configure(text=f"{int(self.interval.get())} ms / 틱")
        if self.running:
            self.pause()
            self.toggle()

    def refresh(self):
        if self.closed or self.simulation is None:
            return
        sim = self.simulation
        snap = sim.env.snapshot()
        self.renderer.draw(snap)
        self.source.configure(text=f"{sim.checkpoint}\n{sim.policies[0].algorithm.upper()} · {sim.experiment.observation_mode} · "
                              f"{sim.experiment.reward_mode}/{sim.experiment.team_signal} · 종료계수 {sim.experiment.terminal_bonus:g}")
        state = "종료" if snap.terminated else "재생 중" if self.running else "일시정지"
        outcome = ("무승부" if snap.areas[0]==snap.areas[1] else "빨강 승리" if snap.areas[0]>snap.areas[1] else "파랑 승리") if snap.terminated else ""
        self.status.configure(text=f"{state} {outcome} · 틱 {snap.tick}/{snap.max_ticks} · 맵 {sim.game.height}×{sim.game.width}\n"
                              f"빨강 면적 {snap.areas[0]} / 파랑 면적 {snap.areas[1]} · 환경 seed {sim.seed} / 정책 seed {sim.policy_seed}\n"
                              f"{'양 팀 모두 저장 모델' if sim.mode=='both' else '선택 모델 묶음 대 고정 상대'} · 학습 업데이트 없음")
        rows=[]
        for agent in snap.agents:
            aid=agent.agent_id
            owner=f"모델 role {sim.roles[aid]}" if aid in sim.roles else "고정 정책"
            position=str(agent.position) if agent.alive else f"부활 대기 (예정 틱 {agent.respawn_at})"
            action=ACTION_NAMES[int(sim.last_actions[aid])] if aid in sim.last_actions else "아직 행동 없음"
            info=sim.last_decisions.get(aid)
            diagnostic=(f"{'기방문' if info['known'] else '미방문'} / 최댓값 {info['ties']}개") if info else ""
            ready="공격 가능" if sim.env.action_mask(aid)[5:].any() else "공격 불가"
            rows.append(f"{aid} [{owner}] {position} · {ready}\n  직전: {action} · 누적 보상 {sim.returns[aid]:.2f} · {diagnostic}")
        self._text(self.roster,"\n".join(rows))
        names={"capture":"점령","attack":"공격","death":"사망","respawn":"부활","respawn_wait":"부활 지연","move":"이동 실패","rejected":"행동 거부"}
        events=[] if sim.last_result is None else [e for e in sim.last_result.events if e.kind!="move" or e.reason!="success"]
        self._text(self.events,"직전 틱 사건\n"+("\n".join(f"{names.get(e.kind,e.kind)} · {e.agent_id or e.position} · {e.reason}" for e in events[-10:]) or "없음"))

    def close(self):
        if self.closed:
            return
        self.pause()
        self.closed = True
        self.renderer.close()

    def run(self):
        self.root.mainloop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",nargs="?",type=Path,help="생략하면 창에서 파일을 선택합니다.")
    parser.add_argument("--mode",choices=("auto","both","fixed"),default="auto")
    parser.add_argument("--cohort",type=int,choices=(0,1),default=0)
    parser.add_argument("--team",choices=("red","blue"),default="red",help="첫 모델 묶음의 진영")
    parser.add_argument("--opponent",choices=OPPONENTS,default="expand")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--policy-seed",type=int)
    parser.add_argument("--interval-ms",type=int,default=250)
    parser.add_argument("--headless",action="store_true",help="창 없이 끝까지 추론하고 JSON 요약만 출력합니다.")
    args=parser.parse_args()
    if args.headless and args.checkpoint is None:
        parser.error("headless 모드에서는 체크포인트 경로가 필요합니다.")
    if not 20<=args.interval_ms<=2000:
        parser.error("재생 간격은 20~2000ms여야 합니다.")
    sim=None
    if args.checkpoint is not None:
        try:
            sim=ModelSimulation(args.checkpoint,args.mode,args.cohort,0 if args.team=="red" else 1,
                                args.opponent,args.seed,args.policy_seed)
        except Exception as exc:
            parser.error(f"모델 로딩 실패: {exc}")
    if args.headless:
        while not sim.env.terminated:
            sim.step()
        print(json.dumps(sim.summary(),ensure_ascii=False))
        return
    SimulationViewer(simulation=sim,interval_ms=args.interval_ms).run()


if __name__=="__main__":
    main()
