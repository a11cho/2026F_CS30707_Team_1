"""기록된 수치에서 PPO·SSL 곡선, 승률 표와 맞대결 보고서를 만듭니다."""
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from analysis import read_rows
from storage import write_csv

FONT_NAMES = {f.name for f in font_manager.fontManager.ttflist}
for family in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
    if family in FONT_NAMES:
        plt.rcParams["font.family"] = family
        break
plt.rcParams.update({"axes.unicode_minus":False, "axes.spines.top":False, "axes.spines.right":False,
                     "axes.grid":True, "grid.alpha":0.18, "axes.titleweight":"bold"})
FIELDS = {"actor_loss":"Actor loss", "critic_loss":"Critic loss", "entropy":"정책 엔트로피", "approx_kl":"근사 KL"}
COLORS = {"red":"#BE4733", "blue":"#186EAC"}


def values(rows, key):
    return np.array([float(r[key]) for r in rows])


def save(fig, path):
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def training_plots(root, out, runs, update):
    out.mkdir(parents=True, exist_ok=True)
    summary, ssl_summary = [], []
    by_size = defaultdict(list)
    fits = {}
    for r in runs:
        by_size[r.size].append(r)
        if not r.settings.get("encoder"):
            continue
        rel = Path(r.settings["encoder"].replace(chr(92), "/"))
        fits[root/rel.parent/"metrics.csv"] = f"{r.size}vs{r.size} / {rel.parent.name}"
    for size, group in sorted(by_size.items()):
        fig, axes = plt.subplots(len(group),4,figsize=(14,2.45*len(group)+1.1), squeeze=False,
                                 sharex=True, sharey="col", layout="constrained")
        for i, run in enumerate(group):
            rows = [x for x in read_rows(run.path/"training.csv") if int(x["update"])<=update]
            for team, color in COLORS.items():
                d = sorted((x for x in rows if x["team"]==team), key=lambda x:int(x["update"]))
                if [int(x["update"]) for x in d] != list(range(1,update+1)):
                    plt.close(fig)
                    raise ValueError(f"{run.name}/{team}: 업데이트 1~{update} 학습 로그를 확인하세요.")
                width = min(15,len(d))
                for j,(key,title) in enumerate(FIELDS.items()):
                    ax = axes[i,j]
                    x, y = values(d,"episodes"), values(d,key)
                    ax.plot(x,y,color=color,alpha=.20,lw=.7)
                    ax.plot(x[width-1:],np.convolve(y,np.ones(width)/width,"valid"),color=color,lw=1.6,label=team)
                    if key=="critic_loss" and np.all(y>0):
                        ax.set_yscale("log")
                    if i==0:
                        ax.set_title(title)
                    if j==0:
                        ax.set_ylabel(f"{run.name} · 시드 {run.seed}")
                for band,part in (("처음",d[:25]),("마지막",d[-25:])):
                    summary.append({"run":run.name,"size":size,"seed":run.seed,"team":team,"window":band,
                                    "updates":len(part), **{k:float(values(part,k).mean()) for k in (*FIELDS,"per_agent_return","ticks")}})
        axes[0,-1].legend(fontsize=8)
        fig.suptitle(f"{size}vs{size} PPO · 업데이트 원자료와 최근 {min(update,15)}회 평균",weight="bold")
        fig.supxlabel("누적 학습 경기")
        save(fig,out/f"ppo_{size}v{size}.png")
    write_csv(out/"training_windows.csv",summary)
    if not fits:
        return summary, []
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for path,label in sorted(fits.items()):
        d = read_rows(path)
        if not d:
            raise ValueError(f"빈 SSL 학습 곡선: {path}")
        x = values(d,"epoch")
        line, = axes[0].plot(x,values(d,"validation_loss"),label=label+" 검증")
        color = line.get_color()
        axes[0].plot(x,values(d,"train_loss"),"--",color=color,label=label+" 학습")
        front = np.array([float(r["encoder_frontier"] or "nan")*100 for r in d])
        axes[1].plot(x,front,color=color,label=label+" 인코더")
        if d[0]["neighbour_frontier"]:
            axes[1].axhline(100*float(d[0]["neighbour_frontier"]),color=color,ls=":",label=label+" 이웃")
        best = min(d,key=lambda r:float(r["validation_loss"]))
        ssl_summary.append({"encoder":label,"source":path.relative_to(root).as_posix(),**best})
    scope = "전체" if runs[0].settings["config"]["ssl"]["mask_fraction"] == 0 else "가려진"
    for ax,title in zip(axes,(f"{scope} 칸의 교차엔트로피",f"{scope} 전선 칸의 정확도 (%)")):
        ax.set_title(title)
        ax.set_xlabel("사전학습 epoch")
        ax.legend(fontsize=8,ncol=2)
    save(fig,out/"ssl.png")
    write_csv(out/"ssl_validation.csv",ssl_summary)
    return summary, ssl_summary


def aggregate(games):
    n = len(games)
    w,d,l = (sum(int(r[k]) for r in games) for k in ("win","draw","loss"))
    return dict(games=n,wins=w,draws=d,losses=l,final_gap=float(values(games,"final_gap").mean()),
                per_agent_return=float(values(games,"per_agent_return").mean()),
                score=(w+0.5*d)/n,win_rate=w/n)


def table(headers, rows):
    def line(cells):
        return "| "+" | ".join(map(str,cells))+" |"
    return "\n".join([line(headers),line(["---"]*len(headers)),*(line(r) for r in rows)])


def link(path, base):
    return Path(os.path.relpath(path,base)).as_posix()


def report(root, out, dest, runs, update, data, opts, games):
    training, ssl = training_plots(root,out,runs,update)
    bots = opts["opponents"]
    fixed, by_size, duels = {}, defaultdict(list), []
    details, sources = [], []
    for evaluation in data:
        run, rival = evaluation.run, evaluation.rival
        sources.append([run.name, rival.name if rival else "고정 상대",len(evaluation.games),
                        f"[경기별 CSV]({link(evaluation.path/'games.csv',dest)})"])
        if rival:
            duels.append((f"{run.name} / {rival.name}",run.seed,aggregate(evaluation.games)))
        else:
            for bot in bots:
                subset = [r for r in evaluation.games if r["opponent"]==bot]
                fixed[run.name,bot] = aggregate(subset)
                by_size[run.size,bot].extend(subset)
        groups = defaultdict(list)
        for r in evaluation.games:
            groups[r["opponent"],r["trained_team"],r["opponent_team"],r["field"]].append(r)
        for (opponent,team,other,field),rows in groups.items():
            details.append(dict(run=run.name,representation=run.settings["config"].get("representation","masked"),size=run.size,train_seed=run.seed,trained_team=team,
                                opponent=opponent,opponent_team=other,field=field,**aggregate(rows)))
    write_csv(dest/"evaluation_groups.csv",details)
    cols = 2 if duels else 1
    fig,axs=plt.subplots(1,cols,figsize=(6.7*cols,max(4.2,.6*len(runs)+1.5)),squeeze=False,layout="constrained")
    ax=axs[0,0]
    rates=np.array([[fixed[r.name,b]["win_rate"] for b in bots] for r in runs])
    cmap=LinearSegmentedColormap.from_list("승률",["#F5DBC7","#FAF9EF","#6DB6AA","#194F53"])
    ax.imshow(rates,aspect="auto",vmin=0,vmax=1,cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(bots)),bots)
    ax.set_yticks(range(len(runs)),[f"{r.name} · {r.size}v{r.size} · 시드 {r.seed}" for r in runs])
    for (i,j),v in np.ndenumerate(rates):
        ax.text(j,i,f"{v:.1%}",ha="center",va="center",color="white" if v>.78 else "#142F35")
    ax.set_title("고정 상대별 승률 · 빨강/파랑 학습 팀 합계")
    if duels:
        ax=axs[0,1]
        left=np.zeros(len(duels))
        for key,label,color in (("wins","복제 팀 승","#286B78"),("draws","무승부","#D8DEE1"),("losses","직접 팀 승","#B06B51")):
            counts=np.array([s[key] for _,_,s in duels])
            widths=100*counts/[s["games"] for _,_,s in duels]
            ax.barh(range(len(duels)),widths,left=left,label=label,color=color)
            for i,width in enumerate(widths):
                if width>=5:
                    ax.text(left[i]+width/2,i,str(counts[i]),va="center",ha="center",color="#263C41" if key=="draws" else "white")
            left+=widths
        ax.set_yticks(range(len(duels)),[f"시드 {s} · {name}" for name,s,_ in duels])
        ax.set_xlim(0,100)
        ax.invert_yaxis()
        ax.set_xlabel("경기 비율 (%) · 막대 안 숫자는 경기 수")
        ax.set_title("1vs1 복제 팀과 3vs3 직접 학습 팀")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False)
    save(fig,dest/"evaluation.png")
    body=[f"# {root.name} · 업데이트 {update} 학습 및 3vs3 평가",
          f"평가 시드 {opts['seed']} · 학습 팀·상대 조합별 {games}경기 · 양 진영 절반씩. 승률은 승수/전체 경기 수입니다.",
          "## 자기대전 학습"]
    for size in sorted({r.size for r in runs}):
        body.append(f"![{size}vs{size} 학습 곡선]({link(out/f'ppo_{size}v{size}.png',dest)})")
    bands=[]
    for size in sorted({r.size for r in runs}):
        for band in ("처음","마지막"):
            group=[x for x in training if x["size"]==size and x["window"]==band]
            avg=lambda k:float(np.mean([x[k] for x in group]))
            # 양 팀을 같은 가중치로 평균하면 틱당 팀 점유율 보상은 0.5입니다.
            bands.append([size,band,int(avg("updates")),*[f"{avg(k):.5g}" for k in FIELDS],
                          f"{avg('per_agent_return'):.2f}",f"{avg('per_agent_return')-avg('ticks')/2:.2f}"])
    body.append(table(["팀 인원","구간","업데이트 수",*FIELDS,"평균 보상","개인 점령 보상분"],bands))
    body.append("양 팀을 동일 가중치로 평균한 구간입니다. 개인 점령 보상분은 평균 에이전트 보상에서 평균 경기 길이의 절반(점유율 보상)을 뺀 값입니다.")
    if ssl:
        body.extend(["## 인코더 검증",f"![SSL 검증]({link(out/'ssl.png',dest)})",
                 table(["인코더","최저 검증 loss epoch","검증 loss","전선 정확도","이웃 다수결"],
                       [[r["encoder"],r["epoch"],f"{float(r['validation_loss']):.4f}",r["encoder_frontier"] or "—",r["neighbour_frontier"] or "—"] for r in ssl])])
    body.extend(["## 고정 상대","![평가 승률](evaluation.png)"])
    stats=[]
    for (size,bot),rows in sorted(by_size.items()):
        s=aggregate(rows)
        stats.append([size,bot,s["games"],s["wins"],s["draws"],s["losses"],f"{s['final_gap']:+.2f}",f"{s['per_agent_return']:.2f}",f"{s['win_rate']:.2%}"])
    body.append(table(["팀당 학습 인원","상대","경기","승","무","패","최종 격차","에이전트당 보상","승률"],stats))
    if duels:
        all_duels=[row for e in data if e.rival for row in e.games]
        duels.append(("합계","전체",aggregate(all_duels)))
        body+=["## 맞대결 · 복제 팀 관점",table(["run 쌍","학습 시드","승","무","패","점령 격차","무승부 절반 점수","승률"],
                [[name,seed,s["wins"],s["draws"],s["losses"],f"{s['final_gap']:+.2f}",f"{s['score']:.2%}",f"{s['win_rate']:.2%}"] for name,seed,s in duels])]
    body+=["## 원자료",table(["실행","상대","경기","기록"],sources),"[학습 팀·실제 진영별 전체 평가표](evaluation_groups.csv)"]
    (dest/"REPORT.md").write_text("\n\n".join(body)+"\n",encoding="utf-8")
