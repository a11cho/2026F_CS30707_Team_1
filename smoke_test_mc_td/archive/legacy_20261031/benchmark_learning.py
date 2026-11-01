"""같은 숫자 설정에서 한 명/여섯 명 학습 시간을 독립 프로세스로 비교합니다."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from .config import RunConfig
from .profile_runs import memory
from .runner import train
from territory_rl import ExperimentConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",action="store_true")
    parser.add_argument("--learners",type=int,choices=(1,6),default=6)
    parser.add_argument("--algorithm",choices=("mc","sarsa"),default="mc")
    parser.add_argument("--observation",choices=("local","aware"),default="local")
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if args.child:
        cfg=RunConfig(learner_count=args.learners)
        before=memory()
        wall,cpu=time.perf_counter(),time.process_time()
        models,rows=train(cfg,ExperimentConfig(args.observation,"individual","gross",0),args.algorithm,7)
        cpu,wall=time.process_time()-cpu,time.perf_counter()-wall
        print(json.dumps(dict(learners=args.learners,algorithm=args.algorithm,observation=args.observation,
                              seed=7,episodes=cfg.episodes,ticks=cfg.ticks,height=cfg.height,width=cfg.width,
                              wall_seconds=wall,cpu_seconds=cpu,peak_working_set_bytes=memory()["peak_working_set_bytes"],
                              updates=sum(m.updates for m in models),q_states=sum(len(m.q) for m in models),
                              learner_samples=sum(row["learner_samples"] for row in rows))))
        return
    rows=[]
    for algorithm in ("mc","sarsa"):
        for observation in ("local","aware"):
            for repeat in range(args.repeats):
                # 실행 순서의 영향을 줄이도록 반복마다 1명/6명 측정 순서를 바꿉니다.
                for learners in ((1,6) if repeat%2==0 else (6,1)):
                    result=subprocess.run([sys.executable,"-B","-X","utf8","-m","smoke_test_mc_td.benchmark_learning",
                        "--child","--learners",str(learners),"--algorithm",algorithm,"--observation",observation],
                        check=True,capture_output=True,text=True,encoding="utf-8")
                    row=json.loads(result.stdout)
                    row["repeat"]=repeat
                    rows.append(row)
    summary=[]
    for algorithm in ("mc","sarsa"):
        for observation in ("local","aware"):
            medians={n:statistics.median(r["wall_seconds"] for r in rows if r["learners"]==n and
                      r["algorithm"]==algorithm and r["observation"]==observation) for n in (1,6)}
            summary.append(dict(algorithm=algorithm,observation=observation,one_seconds=medians[1],
                                six_seconds=medians[6],ratio=medians[6]/medians[1]))
    print(json.dumps(dict(note="학습만 측정하며 평가·저장·프로세스 시작·import는 제외합니다.",
                         rows=rows,summary=summary),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

