"""수집된 자료만으로 논문식 그래프와 한국어 Markdown 보고서를 생성합니다."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics as st
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import ticker

COLORS={"mc":"#0072B2","sarsa":"#D55E00"}
NAMES={"mc":"MC","sarsa":"SARSA(0)"}
CONDITIONS=("local_individual_gross_b0","local_team_terminal_b1","aware_individual_gross_b0","aware_team_terminal_b1")
LABELS={"local_individual_gross_b0":"개인 · local","local_team_terminal_b1":"팀 종료 · local",
        "aware_individual_gross_b0":"개인 · aware","aware_team_terminal_b1":"팀 종료 · aware"}


def table(headers,rows):
    def cell(value):
        return str(value).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(cell(v) for v in row)+" |" for row in rows])


def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(root,final_root):
    data=json.loads((root/"data/paper_data.json").read_text(encoding="utf-8"))
    cases=data["cases"]
    get=lambda condition,algorithm:next(c for c in cases if c["selected"]["condition"]==condition and c["selected"]["algorithm"]==algorithm)
    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    plt.rcParams.update({"font.family":"Malgun Gothic","font.size":10,"axes.titlesize":12,
                         "axes.labelsize":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.unicode_minus":False,"svg.fonttype":"none","figure.facecolor":"white",
                         "savefig.facecolor":"white","legend.frameon":False})
    figures=root/"figures"
    figures.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figures/f"{name}.png",dpi=300,bbox_inches="tight")
        fig.savefig(figures/f"{name}.svg",bbox_inches="tight")
        plt.close(fig)
    def band(ax,x,values,color,label):
        values=np.asarray(values,float)
        for line in values:
            ax.plot(x,line,color=color,alpha=.20,lw=.8)
        average=values.mean(axis=0)
        sd=values.std(axis=0,ddof=1)
        ax.plot(x,average,color=color,label=label,lw=2,marker="o",ms=3)
        ax.fill_between(x,average-sd,average+sd,color=color,alpha=.13,lw=0)
    def grid(ax):
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    x=np.arange(0,1001,100)
    diagnostic_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout="constrained")
    for condition,ax in zip(CONDITIONS,axes.flat):
        for algorithm in ("mc","sarsa"):
            case=get(condition,algorithm)
            values=[[r["validation_score"]*100 for r in case["diagnostics"][str(seed)]] for seed in (7,19,31)]
            band(ax,x,values,COLORS[algorithm],NAMES[algorithm])
            for seed in (7,19,31):
                for row in case["diagnostics"][str(seed)]:
                    diagnostic_rows.append(dict(condition=condition,algorithm=algorithm,seed=seed,**row))
        ax.axhline(6.25,color="#777777",ls="--",lw=1,label="학습 전: 같은 소규모 평가")
        ax.set(title=LABELS[condition],xlabel="학습 에피소드 (에피소드당 64틱)",ylabel="고정 validation 진단 점수 (%)",ylim=(0,42))
        grid(ax)
    axes[0,0].legend(fontsize=9,loc="upper left")
    fig.suptitle("그림 1. 학습 중 고정 평가 추이 · 얇은 선: seed별, 굵은 선: 평균, 음영: ±1 SD (n=3)",fontsize=13)
    save(fig,"fig01_learning_curves")
    save_csv(root/"data/diagnostics_long.csv",diagnostic_rows)

    baseline=np.mean([r["score"] for r in data["baseline_test"]])
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True,layout="constrained")
    centers=np.arange(4)
    for algorithm,offset in (("mc",-.18),("sarsa",.18)):
        selected=[get(c,algorithm) for c in CONDITIONS]
        means=[100*c["group"]["validation_score_mean"] for c in selected]
        errors=[100*c["group"]["validation_score_std"] for c in selected]
        axes[0].bar(centers+offset,means,width=.32,color=COLORS[algorithm],alpha=.85,label=NAMES[algorithm],yerr=errors,capsize=4)
        axes[1].bar(centers+offset,[100*c["selected"]["test"]["score"] for c in selected],width=.32,color=COLORS[algorithm],alpha=.85,label=NAMES[algorithm])
    axes[0].axhline(18.125,color="#555555",ls="--",label="Q=0 validation 기준선")
    axes[1].axhline(100*baseline,color="#555555",ls="--",label="Q=0 test 기준선")
    for ax,title in zip(axes,("Validation: 학습 seed 3개 평균 ± SD","Test: validation으로 선택한 seed 한 개")):
        ax.set(title=title,ylabel="경기 점수 (%)",ylim=(0,35),xticks=centers,xticklabels=[LABELS[c] for c in CONDITIONS])
        ax.tick_params(axis="x",labelrotation=18)
        grid(ax)
        ax.legend(fontsize=9)
    fig.suptitle("그림 2. 최종 성능 · 승리 1 / 무승부 0.5 / 패배 0",fontsize=13)
    save(fig,"fig02_final_performance")

    # 환경 pair를 블록으로 두어 두 팀 묶음·상대·진영의 상관을 보존합니다.
    def key(row):
        return int(row["pair"]),row["opponent"],int(row["team"]),int(row["cohort"])
    base_map={key(r):r["score"] for r in data["baseline_test"]}
    rng=np.random.default_rng(20261031)
    draws=rng.integers(0,30,size=(10000,30))
    paired=[]
    for condition in CONDITIONS:
        for algorithm in ("mc","sarsa"):
            case=get(condition,algorithm)
            blocks=defaultdict(list)
            for row in case["test"]:
                blocks[int(row["pair"])].append(row["score"]-base_map[key(row)])
            assert len(blocks)==30 and all(len(rows)==16 for rows in blocks.values())
            block_values=np.array([np.mean(blocks[p]) for p in range(30)])
            bootstrap=block_values[draws].mean(axis=1)
            low,high=np.quantile(bootstrap,[.025,.975])
            paired.append(dict(condition=condition,algorithm=algorithm,seed=case["selected"]["seed"],
                               difference_pp=100*block_values.mean(),ci_low_pp=100*low,ci_high_pp=100*high))
    fig,ax=plt.subplots(figsize=(10,6),layout="constrained")
    for i,row in enumerate(paired):
        ax.errorbar(row["difference_pp"],i,xerr=[[row["difference_pp"]-row["ci_low_pp"]],[row["ci_high_pp"]-row["difference_pp"]]],
                    fmt="o",color=COLORS[row["algorithm"]],capsize=4,lw=1.5)
    ax.axvline(0,color="#555555",ls="--",lw=1)
    ax.set(yticks=np.arange(8),yticklabels=[LABELS[r["condition"]]+" / "+NAMES[r["algorithm"]] for r in paired],
           xlabel="학습 전 Q=0 대비 test 점수 차이 (퍼센트포인트)",title="그림 3. 선택 모델의 대응 점수 차이와 환경 pair 블록 bootstrap 95% 구간")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.18)
    save(fig,"fig03_paired_test_differences")
    save_csv(root/"data/paired_bootstrap.csv",paired)

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for obs,col in (("local",0),("aware",1)):
        for reward,style in (("individual_gross_b0","-"),("team_terminal_b1","--")):
            condition=obs+"_"+reward
            for algorithm in ("mc","sarsa"):
                case=get(condition,algorithm)
                rows=[case["diagnostics"][str(seed)] for seed in (7,19,31)]
                states=np.array([[r["q_states"] for r in rs] for rs in rows])/1000
                visits=np.array([[r["mean_pair_visits"] for r in rs] for rs in rows])
                label=NAMES[algorithm]+(" 개인" if reward.startswith("individual") else " 팀 종료")
                for ax,values in ((axes[0,col],states),(axes[1,col],visits)):
                    # 방문 경험이 0인 초기점의 평균 방문 수는 정의하지 않고 제외합니다.
                    xx,values=(x[1:],values[:,1:]) if ax is axes[1,col] else (x,values)
                    ax.plot(xx,values.mean(axis=0),color=COLORS[algorithm],ls=style,lw=2,label=label)
                    sd=values.std(axis=0,ddof=1)
                    ax.fill_between(xx,values.mean(axis=0)-sd,values.mean(axis=0)+sd,color=COLORS[algorithm],alpha=.08)
                    grid(ax)
        axes[0,col].set(title=obs,ylabel="여섯 모델의 저장 key 수 합계 (천 개)")
        axes[1,col].set(xlabel="학습 에피소드",ylabel="상태·행동 쌍당 평균 방문 횟수")
        axes[0,col].legend(fontsize=8)
    fig.suptitle("그림 4. 표현 크기와 경험 재사용 · 보상 조건과 무관하게 aware의 key가 계속 증가",fontsize=13)
    save(fig,"fig04_state_reuse")

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for condition,ax in zip(CONDITIONS,axes.flat):
        for algorithm in ("mc","sarsa"):
            case=get(condition,algorithm)
            values=np.array([[r["mean_abs_q_update"] for r in case["diagnostics"][str(seed)][1:]] for seed in (7,19,31)])
            probe=np.array([[r["probe_mean_abs_q_change"] for r in case["diagnostics"][str(seed)][1:]] for seed in (7,19,31)])
            average=values.mean(axis=0)
            ax.plot(x[1:],np.maximum(average,1e-8),color=COLORS[algorithm],label=NAMES[algorithm]+" 1회 갱신",lw=2)
            ax.plot(x[1:],np.maximum(probe.mean(axis=0),1e-8),color=COLORS[algorithm],ls=":",label=NAMES[algorithm]+" 고정 probe",lw=1.8)
        ax.set(title=LABELS[condition],xlabel="학습 에피소드",ylabel="평균 절댓값 (로그 축)",yscale="log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,position:f"{value:.0e}"))
        grid(ax)
        ax.legend(fontsize=8)
    fig.suptitle("그림 5. Q 안정화 진단 · 점선은 고정 key의 100에피소드 간 Q 변화 (0은 표시용 1e-8)",fontsize=13)
    save(fig,"fig05_q_diagnostics")

    opponents=("random","expand","aggressive","defend")
    heat=[]
    labels=[]
    for condition in CONDITIONS:
        for algorithm in ("mc","sarsa"):
            rows=get(condition,algorithm)["test"]
            heat.append([100*np.mean([r["score"] for r in rows if r["opponent"]==op]) for op in opponents])
            labels.append(LABELS[condition]+" / "+NAMES[algorithm])
    heat.append([100*np.mean([r["score"] for r in data["baseline_test"] if r["opponent"]==op]) for op in opponents])
    labels.append("학습 전 Q=0")
    fig,axes=plt.subplots(1,2,figsize=(14,7),width_ratios=(1.5,1),layout="constrained")
    im=axes[0].imshow(heat,vmin=0,vmax=100,cmap="Blues",aspect="auto")
    axes[0].set(xticks=np.arange(4),xticklabels=opponents,yticks=np.arange(9),yticklabels=labels,title="상대별 test 점수 (%) · 상대당 120경기")
    for i,row in enumerate(heat):
        for j,value in enumerate(row):
            axes[0].text(j,i,f"{value:.1f}",ha="center",va="center",color="white" if value>55 else "#222222",fontsize=9)
    fig.colorbar(im,ax=axes[0],shrink=.7,label="점수 (%)")
    for i,(condition,algorithm) in enumerate((c,a) for c in CONDITIONS for a in ("mc","sarsa")):
        selected=get(condition,algorithm)["selected"]
        a,b=(selected["test_by_cohort"][str(k)]["score"]*100 for k in (0,1))
        axes[1].scatter(a,b,color=COLORS[algorithm],s=60,marker="o" if condition.startswith("local") else "s")
        axes[1].annotate(str(i+1),(a,b),xytext=(5,3),textcoords="offset points",fontsize=9)
    axes[1].plot([0,35],[0,35],color="#888888",ls="--")
    axes[1].set(xlabel="모델 묶음 0의 test 점수 (%)",ylabel="모델 묶음 1의 test 점수 (%)",xlim=(0,35),ylim=(0,35),title="학습된 두 팀의 성적 · 번호는 왼쪽 행 1–8")
    grid(axes[1])
    fig.suptitle("그림 6. 고정 상대 일반화와 팀 묶음 간 편차",fontsize=13)
    save(fig,"fig06_opponents_and_cohorts")

    resource=data["resources"]
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout="constrained")
    stages=("training_including_diagnostics","validation","untrained_validation","selected_test")
    stage_names=("학습+중간 진단","최종 validation","학습 전 기준선","선택 모델 test")
    phase_colors=("#0072B2","#56B4E9","#999999","#E69F00")
    cases_order=[get(c,a) for c in CONDITIONS for a in ("mc","sarsa")]
    bottom=np.zeros(8)
    for phase,title,color in zip(stages,stage_names,phase_colors):
        values=np.array([sum(r["wall_seconds"] for r in resource if r["condition"]==case["selected"]["condition"] and r["algorithm"]==case["selected"]["algorithm"] and r["phase"]==phase)/60 for case in cases_order])
        axes[0].bar(np.arange(8),values,bottom=bottom,color=color,label=title)
        bottom+=values
    axes[0].set(xticks=np.arange(8),xticklabels=[f"{i+1}" for i in range(8)],xlabel="조건/방법 번호 (그림 6의 행 순서)",ylabel="세 seed와 선택 test의 합산 경과 시간 (분)",title="기록된 phase 시간 · 저장/로딩 등의 간격 제외")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(1,len(resource)+1),[r["working_set_bytes"]/2**30 for r in resource],label="phase 종료 시 상주 메모리",color="#0072B2")
    axes[1].plot(np.arange(1,len(resource)+1),[r["peak_working_set_bytes"]/2**30 for r in resource],label="프로세스 수명 peak",color="#D55E00",ls="--")
    axes[1].set(xlabel="순차 phase 번호 (시간 축 아님)",ylabel="프로세스 메모리 (GiB)",title="학습·평가·체크포인트 로딩을 포함한 프로세스")
    axes[1].legend(fontsize=8)
    for ax in axes: grid(ax)
    fig.suptitle("그림 7. 실제 실행 자원 기록 · 순차 실행이므로 조건 간 시간 차이를 벤치마크로 해석하지 않음",fontsize=13)
    save(fig,"fig07_resources")

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for condition,ax in zip(CONDITIONS,axes.flat):
        for algorithm in ("mc","sarsa"):
            case=get(condition,algorithm)
            values=[]
            for seed in (7,19,31):
                rows=case["training"][str(seed)]
                # 겹치지 않는 50에피소드 창입니다. 첫 팀 묶음의 보상만 그려 상쇄를 피합니다.
                values.append([np.mean([r["mean_return"] for r in rows[start:start+50]]) for start in range(0,1000,50)])
            band(ax,np.arange(25,1000,50),values,COLORS[algorithm],NAMES[algorithm])
        ax.set(title=LABELS[condition],xlabel="학습 에피소드 (50개 비중첩 구간 평균)",ylabel="첫 모델 묶음의 에이전트당 비할인 보상")
        grid(ax)
        ax.legend(fontsize=9)
    fig.suptitle("그림 8. 자기대전 학습 보상 · 상대 정책과 epsilon도 바뀌므로 성능 향상으로 단정하지 않음",fontsize=13)
    save(fig,"fig08_training_returns")

    write_markdown(root,final_root,data,paired)
    print("그래프 8개(PNG/SVG)와 보고서 작성 완료:",root,flush=True)


def write_markdown(root,final_root,data,paired):
    cases=data["cases"]
    manifest=data["manifest"]
    cfg=manifest["config"]
    baseline=float(np.mean([r["score"] for r in data["baseline_test"]]))
    sections=[]
    def add(text): sections.append(text.strip())
    def image(name,caption):
        add(f"![{caption}](<figures/{name}>)\n\n{caption}")
    def case_label(case): return LABELS[case["selected"]["condition"]]+" / "+NAMES[case["selected"]["algorithm"]]
    add("# 양 팀 동시 학습 MC / SARSA(0) 스모크테스트 보고서\n\n**실행 ID:** `20261031T112320430780Z` · **원본 실행일:** 2026-10-31 · **학습 seed:** 7, 19, 31\n\n이 문서는 학습을 다시 수행하지 않고 원본 로그·최종 체크포인트·추가 추론으로 작성했습니다. 그래프는 300dpi PNG와 편집 가능한 SVG로 함께 제공하며, 그래프 데이터와 생성 코드도 동봉했습니다.")
    add("## 초록")
    add(f"4×4 영역 점령 환경에서 3대3 여섯 에이전트가 독립 Q-table로 동시에 학습했다. 개인 점령/팀 종료 승패 보상과 local/aware 관측의 네 조건에 MC 제어와 SARSA(0)를 적용하고, 각 조건을 세 개의 seed로 1,000에피소드씩 학습했다. 후보는 총 24개이며 validation으로 선택한 8개 모델 묶음을 고정 상대에 평가했다. **개인·local의 선택 모델은 MC 23.13%, SARSA 25.83%의 test 경기 점수로 학습 전 기준선 {baseline*100:.2f}%를 상회했다.** 나머지 조건에서는 뚜렷한 개선을 확인하기 어려웠다. local은 높은 관측 재방문률을 얻었지만 aware는 대부분의 key를 한 번 정도만 경험했다. 관측된 개선은 최적 정책 수렴이나 알고리즘의 일반적인 우열을 입증하지 않는다. 실제 전체 실행은 {manifest['total_wall_seconds']/60:.2f}분이었다.")
    add("## 1. 이전 실행: 짧은 배경")
    previous=data["previous"]
    old_start=datetime.strptime("20261031T093149737642Z","%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    old_seconds=previous["manifest_mtime_utc"]-old_start.timestamp()
    old_models=previous["results"]["representatives"]
    add(table(["항목","이전 초기 실행","이번 실행"],[
        ["맵 / 틱 / 에피소드","6×6 / 32 / 80","4×4 / 64 / 1,000"],
        ["실제 학습자","한 명; 아군 expand 2명, 상대 random 3명","양 팀 여섯 명 모두 학습"],
        ["관측 반경 / 학습 seed","2 / 7,19,31,43,59","1 / 7,19,31"],
        ["alpha / gamma / epsilon","0.1 / 0.99 / 1.0→0.1","0.1 / 0.99 / 0.8→0.05"],
        ["조건·seed / 총 후보","관측·보상 18조건 × 알고리즘 2 × seed 5 = 180","4조건 × 알고리즘 2 × seed 3 = 24"],
        ["전체 경과 시간",f"약 {old_seconds/60:.2f}분; 파일 시각 기반 추정",f"{manifest['total_wall_seconds']/60:.2f}분; 실행 중 계측"],
        ["Validation 선정 대표의 Test",f"MC {100*old_models['mc']['test']['score']:.2f}%, SARSA {100*old_models['sarsa']['test']['score']:.2f}%","MC 23.13%, SARSA 25.83% (모두 개인·local)"],
        ["진단","학습 경기가 전부 팀 승리; 재방문 부족, 개선 미확인","개인·local에서 기준선 대비 개선; 다른 조건과 수렴은 미확인"]]))
    add("이전 실행은 알고리즘 자체의 실패를 입증한 실험이라기보다 학습 구성·표현·기록의 문제를 발견한 초기 스모크테스트였다. 두 실행은 학습자 수·맵·상대 구성·보상 조건이 달라 **절대 점수나 시간만으로 개선/악화를 비교할 수 없다.** 이번에는 고정 아군 도움 없이 두 팀이 동시에 학습한다.")
    add("## 2. 실험 방법과 재현 설정")
    add(table(["구분","실제 설정"],[
        ["맵","height=4, width=4, 총 16타일, 시작 팀별 8타일"],
        ["인원·초기화","3대3 총 6명, frontier 대칭 초기 배치, 에피소드마다 두 모델 묶음의 진영 교대"],
        ["학습자","전원 학습, 독립 Q-table 6개, 파라미터 공유 없음, 학습 중 고정 봇 없음"],
        ["지평·부활","max_ticks=64, respawn_delay=2, 사망 대기도 경험에 포함; 종료 때 bootstrap=0"],
        ["관측","radius=1; local_grid=(6,3,3), self_features=(6,), teammate_features=(2,14), teammate_feature_mask=(2,14), action_mask=(9,)"],
        ["행동","정지 1 + 이동 4 + 공격 4, 마스크 안에서만 선택"],
        ["인코딩","compact_v1: 인접 타일/점유, 적 방향, 본인 좌표·상태·시간 구간, 팀원 상태/직전 행동과 유효 마스크; 부분 관측의 손실 압축"],
        ["MC","on-policy first-visit Q 제어; 에피소드 종료 후 고정 alpha로 갱신"],
        ["SARSA","on-policy 1-step SARSA(0); 실제 다음 행동을 선택·유지하여 매 틱 갱신"],
        ["학습 파라미터","alpha=0.1, gamma=0.99, 초기 Q=0, epsilon=0.8→0.05 선형 감소"],
        ["예산","seed당 1,000에피소드 × 64틱; 환경 틱 64,000, 여섯 명 전이 384,000"],
        ["seed","7,19,31; 독립 초기화, 환경·정책·봇 RNG 분리"],
        ["중간 평가","학습 전 + 100에피소드 간격; 고정 validation 부분집합 32경기, Q 갱신 없음"],
        ["최종 validation","후보당 160경기 = 두 팀 묶음 × 상대 4종 × seed 10 × 진영 2"],
        ["선정","최종 validation 점수→면적 비율→seed 목록 순서; 중간 최고 checkpoint나 test로 선정하지 않음"],
        ["최종 test","선택 모델당 480경기 = 두 팀 묶음 × 상대 4종 × seed 30 × 진영 2"],
        ["고정 평가 상대","random / expand / aggressive / defend"],
        ["평가 정책","epsilon=0, 최대 Q 동률은 seeded 무작위; 미방문 key도 허용 행동에서 무작위"],
        ["예산 총합","24,000 학습 경기·1,536,000 환경 틱·9,216,000 학습자 전이; MC/SARSA의 실제 갱신 수는 다름"],
        ["소프트웨어",f"{manifest['python']}; NumPy {manifest['numpy']}, Gymnasium {manifest['gymnasium']}, PettingZoo {manifest['pettingzoo']}"],
        ["하드웨어","Intel Core Ultra 7 258V, 물리/논리 코어 8/8, OS 보고 RAM 31.51GiB; Windows 11; GPU 학습 없음"]]))
    add(table(["네 조건","중간 보상","종료 보너스","관측"],[
        ["개인·local","본인 점령 credit; 공동 점령 균등 분할","0","주변 + 본인"],
        ["개인·aware","동일","0","local + 시야 밖 포함 팀원 상태·직전 행동"],
        ["팀 종료·local","0","팀 승/무/패 +1/0/−1을 각 팀원에게","주변 + 본인"],
        ["팀 종료·aware","0","동일","local + 팀원 정보"]]))
    add("개인/팀 조건은 **귀속과 보상 시점·목적이 함께 달라지는 비교**이다. 보상 공유만의 효과를 분리한 실험이 아니다. 관전자 영상의 전체 맵은 정책 입력으로 사용하지 않는다. config의 train_opponent=random 필드는 6명 학습 모드에서는 사용되지 않으며, 고정 상대는 평가에만 등장한다.")
    add(r"""### 업데이트 식

각 에이전트가 받은 관측을 압축한 key를 \(x_t\)라 두면, MC는 완결된 에피소드의 return
\(G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}\)을 사용해 첫 방문 상태·행동 쌍만 갱신한다.

\[
Q(x_t,a_t)\leftarrow Q(x_t,a_t)+\alpha\big[G_t-Q(x_t,a_t)\big].
\]

SARSA(0)는 실제로 선택하고 다음 틱에 실행하는 행동 \(a_{t+1}\)으로 매 틱 갱신한다.

\[
Q(x_t,a_t)\leftarrow Q(x_t,a_t)+\alpha\big[r_{t+1}+\gamma Q(x_{t+1},a_{t+1})-Q(x_t,a_t)\big].
\]

실제 종료에서는 다음 Q항을 0으로 하고, 일시 사망은 종료로 보지 않는다.
완전한 Markov state 대신 부분 관측을 사용한다는 점 때문에 표준 수렴 정리를 그대로 적용하지 않는다.""")
    add("## 3. 학습 iteration별 성능과 보상 추이")
    image("fig01_learning_curves.png","그림 1. 0,100,…,1000에피소드의 고정 validation 진단 점수. 얇은 선은 seed별 곡선, 굵은 선은 평균, 음영은 ±1 표본 SD(n=3)이다. 점수는 승 1·무 0.5의 평균이다. 진단은 32경기로, 최종 validation 160경기와 seed 범위가 다르다. 따라서 초기 진단 기준선 6.25%와 최종 기준선 18.125%를 같은 평가값으로 연결하지 않는다.")
    add("**중간 checkpoint의 의미:** 저장된 것은 시점별 진단 통계이며 매 100에피소드의 모델 가중치 파일은 보존하지 않았다. 그림 1은 당시 실제 측정값으로 복원한 곡선이다. 중간 최고 모델을 다시 로드하거나 그 시점의 영상을 만드는 것은 이 자료만으로 불가능하다. 최종 model.json.gz만 관전할 수 있다.")
    image("fig08_training_returns.png","그림 8. 자기대전 중 첫 모델 묶음(role 0–2)의 에이전트당 비할인 보상. 50개 비중첩 에피소드 구간 평균과 seed 간 ±1 SD. 두 팀 보상을 합치면 terminal 조건에서 상쇄되므로 첫 묶음만 사용했다. 학습 상대와 epsilon도 함께 바뀌므로 이 곡선만으로 실력 향상이나 수렴을 판단하지 않는다.")
    add("## 4. 최종 성능과 통계적 불확실성")
    rows=[]
    for case in cases:
        s,g=case["selected"],case["group"]
        rows.append([case_label(case),s["seed"],f"{100*g['validation_score_mean']:.2f} ± {100*g['validation_score_std']:.2f}",
                     f"{100*s['validation']['score']:.2f}",f"{100*s['test']['score']:.2f}",f"{100*s['test']['win']:.2f}",
                     f"{100*s['test']['draw']:.2f}",f"{100*s['test']['area_fraction']:.2f}"])
    add(table(["조건 / 방법","선택 seed","Val 평균±SD (3seed)","선택 Val","선택 Test 점수","Test 승률","무승부율","면적 비율"],rows))
    add(f"같은 test seed에서 추가 추론한 학습 전 Q=0 기준선은 **{baseline*100:.2f}%**이다. 이 기준선의 모델은 팀원 셋 모두 무작위 선택하며, 이전 실행처럼 scripted 아군 두 명의 지원이 없다. 최종 test는 seed 평균이 아니라 validation으로 선정된 한 개의 6모델 묶음에 대한 성적이다.")
    image("fig02_final_performance.png","그림 2. 왼쪽은 전체 학습 seed 3개의 최종 validation 평균±SD, 오른쪽은 선정 모델의 test 점수다. 서로 다른 통계량이므로 좌우 막대 차이를 과적합 정도의 직접 추정치로 해석하지 않는다.")
    image("fig03_paired_test_differences.png","그림 3. 각 선택 모델과 Q=0 기준선의 대응 test 점수 차이. 환경 pair 30개를 블록으로 묶어 두 팀 묶음·상대·진영을 함께 재표집했다(bootstrap 10,000회, 통계 seed=20261031, percentile 95% 구간). 모델과 기준선에 같은 재표집 인덱스를 사용했다.")
    add(table(["조건 / 방법","기준선 대비 차이 pp","환경 pair 95% 구간 pp"],[[LABELS[r["condition"]]+" / "+NAMES[r["algorithm"]],f"{r['difference_pp']:+.2f}",f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"] for r in paired]))
    add("**불확실성 해석:** 그림 3의 구간은 고정된 선택 모델에 대한 평가 환경 변동만 반영한다. 학습 seed·최고 seed 선택·조건 선정의 불확실성을 포함하지 않으며 8개 비교의 다중비교 보정도 하지 않았다. 모든 seed의 test를 평가한 알고리즘 성능 CI나 일반적 유의성 결론으로 사용할 수 없다. seed 3개는 스모크테스트 기술통계에 가깝다.")
    add("## 5. 표현 크기·반복 경험·Q 안정화")
    image("fig04_state_reuse.png","그림 4. 같은 경험 예산에서 저장 key 수와 상태·행동 쌍 평균 방문 수. 여섯 독립 테이블의 개수를 합산했다. mean_pair_visits에는 사망 대기 NOOP도 포함되어 생존 전술 상태의 재방문률과 같지 않다.")
    rows=[]
    for case in cases:
        last=[case["diagnostics"][str(seed)][-1] for seed in (7,19,31)]
        model=case["model_statistics"]
        states=sum(r["states"] for r in model)
        q_count=sum(r["q_scalars"] for r in model)
        nz=sum(r["nonzero"] for r in model)
        rows.append([case_label(case),f"{np.mean([r['q_states'] for r in last]):,.0f}",f"{np.mean([r['mean_pair_visits'] for r in last]):.2f}",
                     f"{states:,}",f"{q_count:,}",f"{100*nz/q_count:.2f}",f"{100*case['selected']['test']['known_state_fraction']:.2f}",
                     f"{case['checkpoint_bytes']/2**20:.2f}"])
    add(table(["조건 / 방법","최종 key 평균(3seed)","쌍당 방문 평균","선택 모델 key","저장 Q 스칼라 수","0 아닌 Q %","Test 기방문 %","압축 모델 MiB"],rows))
    add("Q 스칼라 수는 저장된 key 수 × 9이며 신경망의 파라미터 수가 아니다. 불법 행동의 0값도 포함한다. 0이 아닌 Q가 많다는 것이 좋은 행동을 구별했다는 뜻은 아니다. 이번 local 개인 조건은 높은 재방문률과 성능 개선이 함께 나타났지만, aware는 정보가 추가되면서 상태 표현이 세분화되어 새 key가 계속 증가했다. 이는 정보 공유 자체가 나쁘다는 증거가 아니라 **현재 표 기반 표현·예산에서 정보를 재사용하기 어렵다**는 진단이다.")
    image("fig05_q_diagnostics.png","그림 5. 굵은 실선은 구간당 평균 1회 Q 업데이트 절댓값, 점선은 첫 에피소드의 고정 key에서 본 구간 간 Q 변화다. probe key는 seed·모델마다 별도로 선정되어 seed 사이에 동일한 집합은 아니며, 각 실행 안에서만 고정된다. 첫 구간 probe는 Q=0과 비교한다. MC의 타깃 오차와 SARSA TD error는 의미가 다르며 보상 스케일도 달라 크기로 우열을 비교하지 않는다. 로그 축의 10⁻⁸ 표시는 실제 0을 표시하기 위한 바닥값이다.")
    add("**수렴 판단:** 모든 조건에서 엄밀 수렴은 확인하지 못했다. local 개인 보상은 행동 성능 개선의 근거가 있지만 Q 변화와 자기대전 상대 변화가 남아 있다. aware·팀 종료 SARSA의 평평한 점수는 수렴의 증거가 아니라 초기와 비슷한 행동 또는 드문 재방문으로 인한 가치 전파 부족일 수 있다. 고정 alpha·부분 관측·key 압축·동시 학습은 표준 고정 MDP 수렴 정리의 전제와 다르다. probe는 첫 에피소드의 한정된 key라 전체 상태 공간의 안정화를 대표하지 않는다.")
    add("## 6. 고정 상대 일반화와 학습된 두 팀의 차이")
    image("fig06_opponents_and_cohorts.png","그림 6. 왼쪽은 선택 모델의 상대별 점수(상대당 120경기), 오른쪽은 독립적으로 학습된 두 팀 묶음의 성적이다. 묶음 0과 1은 서로 다른 Q-table 세 개씩을 사용하며 양쪽 진영으로 평가했다. 평균 점수만으로 한 팀의 약한 성적이 가려질 수 있어 따로 표시했다.")
    rows=[]
    for case in cases:
        selected=case["selected"]
        by_op=[100*np.mean([r["score"] for r in case["test"] if r["opponent"]==op]) for op in ("random","expand","aggressive","defend")]
        rows.append([case_label(case),*[f"{v:.2f}" for v in by_op],f"{selected['test']['deaths']:.2f}",
                     f"{100*selected['test']['dead_fraction']:.2f}",f"{selected['test']['captures']:.2f}"])
    add(table(["조건 / 방법","random","expand","aggressive","defend","사망/경기","대기 전이 %","팀 점령/경기"],rows))
    add("평가에서는 선택된 세 명 모두 학습 모델이고 상대 세 명은 고정 정책이다. 사망과 대기는 평가 모델 세 명 합계/전이 기준이며, 점령 사건은 평가 팀 전체 기준이다. 같은 타일 재점령도 포함하므로 점령 횟수가 최종 면적 증가와 같지 않다. 높은 사망·대기 비율은 실제로 행동 가능한 경험이 전체 틱보다 적다는 점을 보여준다.")
    add("**개선이 발생한 상대:** 개인·local의 점수 향상은 주로 random 상대에서 나타났다. Q=0 기준선의 random 점수는 56.67%인데 MC는 90.83%, SARSA는 96.67%였다. expand와 aggressive 상대 점수는 두 모델 모두 0%이고, defend는 각각 1.67%, 6.67%에 그쳤다. 따라서 일반적인 강한 대전 정책을 얻었다기보다는 쉬운 상대에서 더 나은 점령 행동을 얻은 스모크테스트로 해석하는 편이 맞다.")
    add("**활동 양상의 변화:** 개인·local SARSA는 팀 점령 사건이 기준선의 경기당 약 7.46회에서 22.62회로 늘고, 사망 대기 전이 비율은 약 63.38%에서 45.04%로 줄었다. 반면 사망 횟수는 약 14.55회에서 20.29회로 늘었다. 더 자주 죽었다는 사실만으로 더 나쁜 정책이라고 해석할 수 없으며, 활동 가능한 시간과 점령 상호작용이 함께 늘어난 양상이다. 방어·협력 전략을 의도적으로 배웠다는 인과적 결론은 이 집계만으로 내리지 않는다.")
    add("## 7. 실제 시간·CPU·메모리")
    resources=data["resources"]
    peak=max(r["peak_working_set_bytes"] for r in resources)/2**30
    pure_train=sum(r["training_wall_seconds"] for c in cases for rs in c["training"].values() for r in rs)
    pure_cpu=sum(r["training_cpu_seconds"] for c in cases for rs in c["training"].values() for r in rs)
    mid_eval=sum(r["diagnostic_eval_wall_seconds"] for c in cases for rs in c["diagnostics"].values() for r in rs)
    stage_total=sum(r["wall_seconds"] for r in resources)
    total=manifest["total_wall_seconds"]
    add(table(["항목","측정값"],[
        ["전체 경과 시간",f"{total:.2f}초 / {total/60:.2f}분"],
        ["전체 프로세스 CPU 시간",f"{manifest['total_cpu_seconds']:.2f}초 / {manifest['total_cpu_seconds']/60:.2f}분"],
        ["에피소드 내부 학습 시간 합",f"벽시계 {pure_train/60:.2f}분 / CPU {pure_cpu/60:.2f}분; 중간 평가 제외"],
        ["중간 진단 평가 합",f"{mid_eval/60:.2f}분; 구간 집계·출력 비용은 제외"],
        ["학습 phase 합(진단 포함)",f"{sum(r['wall_seconds'] for r in resources if r['phase']=='training_including_diagnostics')/60:.2f}분"],
        ["최종 validation + Q=0 평가",f"{sum(r['wall_seconds'] for r in resources if r['phase'] in ('validation','untrained_validation'))/60:.2f}분"],
        ["선택 모델 test",f"{sum(r['wall_seconds'] for r in resources if r['phase']=='selected_test')/60:.2f}분"],
        ["phase 밖 잔여 시간",f"{(total-stage_total)/60:.2f}분; 주로 직렬화·로딩·CSV/JSON·소스 복사 등 포함, 개별 원인으로 분해하지 않음"],
        ["프로세스 peak working set",f"{peak:.3f}GiB; 프로세스 시작 후 최대값"],
        ["전체 CPU초/경과초",f"{100*manifest['total_cpu_seconds']/total:.1f}% (1코어 기준); 논리 8코어 전체 대비 약 {100*manifest['total_cpu_seconds']/total/8:.1f}% 환산"],
        ["추가 분석·영상 시간","위 63분에 포함하지 않음; 기존 학습 종료 이후 별도로 수행"]]))
    image("fig07_resources.png","그림 7. 실제 순차 실행에서 기록한 phase 시간과 메모리. 최대 메모리는 모델 학습뿐 아니라 큰 Q-table 직렬화·로드 중에도 발생할 수 있다. working set은 실제 상주 페이지이며 private commit은 같은 뜻이 아니다. phase 번호는 시간 간격을 의미하지 않는다.")
    add("메모리의 약 1.24GiB peak는 80에피소드 소규모 속도 검사 때의 수십 MiB와 다른 예산의 실측이다. 반복 횟수·key 수 증가와 직렬화 비용을 함께 고려해야 한다. 실행 중 다른 작업·전원 상태·CPU 스케줄링을 통제하지 않았고 조건을 무작위 순서로 실행하지 않았으므로, 조건별 벽시계 차이를 알고리즘 속도 우열로 해석하지 않는다.")
    add("## 8. 최종 모델 관전 영상")
    add("영상은 기존 viewer.py의 **ModelSimulation 추론 엔진**으로 만든 동일한 상태 전이의 시각화다. Tk 창을 녹화한 것이 아니라 headless 추론 상태를 영상으로 렌더링했다. 모델 업데이트·checkpoint 변경은 없으며 각 장면은 최종 체크포인트의 양 팀 대결이다. 환경 seed=7, 정책 seed=1247478191, epsilon=0, 64틱으로 고정했다. seed를 바꿔 좋은 장면을 고르지 않았다. 이는 한 장면의 정성 자료이며 고정 상대 test 결과나 중간 학습 모델을 보여주는 것은 아니다.")
    add(f"### 네 조건 SARSA(0) 동시 관전\n\n![네 조건 SARSA 관전](<videos/sarsa_four_conditions.gif>)\n\n같은 초기 상태에서 개인/local, 팀 종료/local, 개인/aware, 팀 종료/aware의 선택 SARSA 모델을 나란히 보여준다. 금색 테두리는 직전 틱 공격 목표다. [영상 파일 AVI](videos/sarsa_four_conditions.avi)")
    add(f"### 개인·local MC와 SARSA 비교\n\n![개인 local MC SARSA 관전](<videos/local_individual_mc_sarsa.gif>)\n\nMC seed=19와 SARSA seed=31의 각 양 팀 모델 대결이다. 서로 알고리즘을 맞붙인 cross-play가 아니다. [영상 파일 AVI](videos/local_individual_mc_sarsa.avi)")
    add("AVI는 로컬 재생용 MJPEG이며 GIF는 문서 내 미리보기용이다. 영상과 trace의 모델 seed·추론 seed·checkpoint SHA256은 videos/video_manifest.json과 data/simulation_traces.json에 기록했다. 화면의 전체 맵 정보는 관전자용이며 정책에는 원래 local/aware 관측만 전달됐다.")
    add("## 9. 해석과 다음 토의")
    add("1. **학습 가능한 경로는 확인됐다.** 개인·local에서 validation과 별도 test의 개선이 함께 나타났고, 관측 재방문률도 높아졌다. 다만 최고 seed 선택 후의 결과이며 최적 정책이나 보편적 수렴의 증거는 아니다.\n2. **숫자만 늘려도 모든 조건이 학습되지는 않았다.** aware의 조합적 key 증가와 terminal 보상의 긴 가치 전파는 남아 있다. 다음 예산을 늘리기 전 이 표현이 주는 정보와 재사용 비용을 함께 토의할 필요가 있다.\n3. **개인/팀 비교에는 목적·보상 시점 차이가 있다.** 팀 terminal의 낮은 성적은 팀 보상 자체가 나쁘다는 결론이 아니다.\n4. **자기대전과 고정 상대 일반화를 분리해야 한다.** 학습 중 상대도 바뀌어 Q 변화가 지속될 수 있으며, 배우지 않은 scripted 전략에는 취약할 수 있다.\n5. **모든 학습 seed의 test가 필요하다.** 다음 연구 결론에는 seed 3개 모두의 test와 훈련 변동을 포함하는 분석을 추가하는 것이 좋다. 이번 test는 이미 확인했으므로 다음 하이퍼파라미터 수정 뒤의 최종 검증에는 새 hold-out bank가 필요하다.\n6. **방문 통계의 분해가 도움이 된다.** 사망 NOOP와 생존 상태를 나눈 재방문률, 공격·이동·정지 빈도와 사망 원인, 상황별 replay를 추가하면 행동 양상을 더 구체적으로 설명할 수 있다.")
    add("## 부록 A. seed별 최종 validation 점수와 모델 크기")
    rows=[]
    for case in cases:
        by_seed={r["seed"]:r for r in case["group"]["candidates"]}
        rows.append([case_label(case),*[f"{100*by_seed[seed]['validation']['score']:.2f}" for seed in (7,19,31)],case["selected"]["seed"]])
    add(table(["조건 / 방법","seed 7","seed 19","seed 31","선정 seed"],rows))
    add("## 부록 B. 정확한 난수와 재현 방법")
    add("학습 에피소드 i=0…999에서 환경 seed=derived_seed(master,10,i), 정책 기준 seed=derived_seed(master,11,i)다. 모델 역할 j의 난수는 derived_seed(policy_seed,1,j)로 분리한다. 평가 master=20261031이고 validation 환경/정책 domain은 40/41, test는 50/51이며 pair 인덱스 p를 사용한다. 같은 pair는 두 진영·상대·팀 묶음에서 대응된다.")
    add('''```python
import numpy as np
def derived_seed(master, domain, index=0):
    return int(np.random.SeedSequence([master, domain, index]).generate_state(1)[0])
```''')
    add('''```powershell
$rlPython = 'C:\\Users\\dohyu\\anaconda3\\envs\\rlTeam\\python.exe'
# 동일한 학습을 재현할 때만 실행하세요. 보고서 생성에는 필요하지 않습니다.
& $rlPython -B -X utf8 -u -m smoke_test_mc_td.run --preset agreed
# 저장된 모델 관전: 실행 후 model.json.gz 선택
& $rlPython -B -m smoke_test_mc_td.viewer
```''')
    add("모델·설정·버전·소스 SHA256은 원본 manifest.json에 있고 실제 실행 소스는 source_snapshot/에 보존되어 있다. 최초 자료 수집 당시 현재 소스도 일치했으며, 정리 이후의 재수집은 보존본을 기준으로 검증하고 별도 CLI 프로세스에서 해당 보존 소스로 추론한다. 현재 코드의 이동·변경은 경고 정보로 기록하며 학습 당시 소스를 덮어쓰지 않는다. 원본 학습 로그·모델 파일은 수정하지 않았다. 이전 실행은 archive/legacy_20261031/runs/에 보관한다.")
    add("## 부록 C. 파일과 분석 재생성")
    add("- `figures/`: 그림 1–8의 300dpi PNG와 SVG\n- `data/diagnostics_long.csv`: 모든 중간 진단 원자료\n- `data/untrained_test.csv`: 보고서 작성 시 추가 추론한 기준선\n- `data/paired_bootstrap.csv`: 기준선 대응 차이와 환경 pair 불확실성\n- `data/paper_data.json`: 원본 설정·학습·평가·자원 기록을 모은 분석 자료\n- `data/simulation_traces.json`: 모델별 최종 추론 상태열\n- `videos/`: GIF·AVI·포스터·재생 메타데이터\n- `prepare_paper_data.py`: rlTeam에서 원본 수집과 추론\n- `build_paper_report.py`: matplotlib/Pillow가 있는 Python에서 그래프·문서 생성\n- `render_paper_videos.py`: Pillow로 상태열을 GIF/MJPEG AVI로 렌더링")
    add("분석 코드와 원자료는 함께 제공하지만 Python 환경이 두 개다. rlTeam은 학습 환경·추론을 재현하고, 현재 기본 Anaconda Python은 matplotlib/Pillow로 그래프·영상을 만든다. 출력 영상은 플랫폼별 글꼴 차이가 있을 수 있으나 상태열과 모델 선택은 고정되어 있다. 모델 가중치의 중간 checkpoint를 저장하지 않았다는 한계는 재생성으로 해결되지 않는다.")
    add('''학습을 다시 하지 않고 분석·영상을 재생성하려면 프로젝트 루트에서 실행한다.

```powershell
$rlPython = 'C:\\Users\\dohyu\\anaconda3\\envs\\rlTeam\\python.exe'
$plotPython = 'C:\\Users\\dohyu\\anaconda3\\python.exe'
$run = 'smoke_test_mc_td/runs/20261031T112320430780Z'
$report = 'smoke_test_mc_td/paper_report'
& $rlPython -B -X utf8 "$report/prepare_paper_data.py" $run "$report/data"
& $plotPython -B -X utf8 "$report/build_paper_report.py" $report --final-root (Resolve-Path $report)
& $plotPython -B -X utf8 "$report/render_paper_videos.py" $report
```

그림의 라벨은 한국어이고 Windows Malgun Gothic 글꼴을 사용한다. PNG는 300dpi,
SVG는 편집 가능한 텍스트를 유지한다. 파일 해시와 제작 라이브러리 버전은
`artifacts_manifest.json`에 저장한다.''')
    add(f"원본 실행 경로: `{data['run']}`\n\n통계 재표집 seed=20261031, 횟수=10,000. 모든 그래프의 SD와 CI는 본문에 정의한 범위로만 해석한다.")
    (root/"REPORT.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--final-root",type=Path,required=True)
    args=parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("보고서는 smoke_test_mc_td 안에만 생성할 수 있습니다.")
    make_report(args.root.resolve(),args.final_root)
