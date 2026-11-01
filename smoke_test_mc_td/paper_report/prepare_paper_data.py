"""확정 실행의 원본 로그를 모으고, 추가 추론만 수행해 보고서용 자료를 만듭니다."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in list(row.items()):
            if value and key not in ("phase","condition","algorithm","opponent","split"):
                try:
                    row[key]=float(value)
                except ValueError:
                    pass
    return rows


def verify_source_snapshot(run,repo):
    """실행 당시 보존본을 검증합니다. 현재 코드의 이동·변경은 비교 결과로만 남깁니다."""
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    snapshot=(run/"source_snapshot").resolve()
    current_changes=[]
    for name,digest in manifest["source_sha256"].items():
        normalized=name.replace("\\","/")
        saved=(snapshot/normalized).resolve()
        current=(repo/normalized).resolve()
        if not saved.is_relative_to(snapshot) or not current.is_relative_to(repo.resolve()):
            raise ValueError(f"허용 범위를 벗어난 소스 경로: {name}")
        if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"실행 당시 소스 보존본이 없거나 해시가 다릅니다: {name}")
        if not current.is_file():
            current_changes.append(dict(path=name,status="moved_or_missing"))
        elif hashlib.sha256(current.read_bytes()).hexdigest()!=digest:
            current_changes.append(dict(path=name,status="changed"))
    return dict(snapshot_verified=True,current_source_changes=current_changes)


def find_previous_run(module_root,run_id="20261031T093149737642Z"):
    """아카이브된 실행을 먼저 찾고 정리 전의 runs 경로도 지원합니다."""
    candidates=sorted(module_root.glob(f"archive/*/runs/{run_id}"))
    if len(candidates)>1:
        raise ValueError(f"같은 이전 실행의 아카이브가 여러 개입니다: {run_id}")
    if candidates:
        return candidates[0]
    previous=module_root/"runs"/run_id
    if previous.is_dir():
        return previous
    raise FileNotFoundError(f"이전 실행을 runs 또는 archive/*/runs에서 찾지 못했습니다: {run_id}")


def prepare(run,output):
    repo=run.parents[2]
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    results=json.loads((run/"results.json").read_text(encoding="utf-8"))
    verification=verify_source_snapshot(run,repo)
    # 추론도 현재 코드가 아니라 검증한 실행 당시 소스를 별도 CLI 프로세스에서 사용합니다.
    snapshot=(run/"source_snapshot").resolve()
    sys.path.insert(0,str(snapshot))
    import numpy as np
    from territory_rl import ExperimentConfig
    from smoke_test_mc_td.config import RunConfig
    from smoke_test_mc_td.learning import TabularControl
    from smoke_test_mc_td.runner import evaluate,derived_seed
    from smoke_test_mc_td.viewer import ModelSimulation
    import smoke_test_mc_td.learning as learning_module
    if not Path(learning_module.__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError("재수집은 현재 학습 모듈이 미리 import되지 않은 별도 Python CLI 프로세스에서 실행하세요.")
    assert manifest["status"]=="complete" and len(results["groups"])==8
    output.mkdir(parents=True,exist_ok=True)
    print(f"실행 소스 보존본 검증 완료 · 현재 코드 차이 {len(verification['current_source_changes'])}개 · 보존 소스로 추론",flush=True)
    cases=[]
    for group in results["groups"]:
        condition,algorithm=group["condition"],group["algorithm"]
        case=dict(group=group,training={},diagnostics={})
        for seed in manifest["config"]["seeds"]:
            directory=run/condition/algorithm/f"seed_{seed}"
            case["training"][str(seed)]=read_csv(directory/"training.csv")
            case["diagnostics"][str(seed)]=read_csv(directory/"diagnostics.csv")
        case["selected"]=next(s for s in results["selected"] if s["condition"]==condition and s["algorithm"]==algorithm)
        case["test"]=read_csv(run/condition/algorithm/"selected_test.csv")
        cases.append(case)
    cfg=RunConfig(**manifest["config"])
    # 선정은 이미 끝났습니다. 같은 test의 Q=0 기준선을 참고용으로만 추가합니다.
    baseline=evaluate(cfg,ExperimentConfig("local","individual","gross",0),
                      [TabularControl("sarsa",cfg.alpha,cfg.gamma) for _ in range(6)],"test")
    with (output/"untrained_test.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(baseline[0]))
        writer.writeheader()
        writer.writerows(baseline)
    traces=[]
    for case in cases:
        selected=case["selected"]
        path=run/selected["checkpoint"]
        print("체크포인트 분석:",selected["condition"],selected["algorithm"],flush=True)
        simulation=ModelSimulation(path,seed=7)
        role_stats=[]
        for role,policy in enumerate(simulation.policies):
            nonzero=sum(int(np.count_nonzero(v)) for v in policy.q.values())
            role_stats.append(dict(role=role,states=len(policy.q),q_scalars=9*len(policy.q),nonzero=nonzero,
                                   updates=policy.training_updates,
                                   max_abs_q=max(float(np.abs(v).max()) for v in policy.q.values())))
        case["model_statistics"]=role_stats
        case["checkpoint_bytes"]=path.stat().st_size
        case["checkpoint_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        # 4조건 SARSA 영상과 최상위 개인/local MC 비교 영상에만 같은 seed를 사용합니다.
        if selected["algorithm"]=="sarsa" or selected["condition"]=="local_individual_gross_b0":
            frames=[]
            def capture():
                snapshot=simulation.env.snapshot()
                frames.append(dict(tick=snapshot.tick,owner=snapshot.owner.tolist(),areas=list(snapshot.areas),
                                   agents=[dict(id=a.agent_id,position=a.position,alive=a.alive,last_action=int(a.last_action),
                                                last_attack_tick=a.last_attack_tick,respawn_at=a.respawn_at) for a in snapshot.agents],
                                   actions={aid:int(action) for aid,action in simulation.last_actions.items()},
                                   returns=dict(simulation.returns),counts=dict(simulation.counts)))
            capture()
            while not simulation.env.terminated:
                simulation.step()
                capture()
            traces.append(dict(condition=selected["condition"],algorithm=selected["algorithm"],
                               training_seed=selected["seed"],environment_seed=simulation.seed,policy_seed=simulation.policy_seed,
                               mode=simulation.mode,checkpoint=str(path),checkpoint_sha256=case["checkpoint_sha256"],frames=frames))
        del simulation
        gc.collect()
    previous=find_previous_run(repo/"smoke_test_mc_td")
    previous_data=dict(manifest=json.loads((previous/"manifest.json").read_text(encoding="utf-8")),
                       results=json.loads((previous/"results.json").read_text(encoding="utf-8")))
    previous_data["manifest_mtime_utc"]=(previous/"manifest.json").stat().st_mtime
    data=dict(run=str(run),manifest=manifest,cases=cases,baseline_test=baseline,
              resources=read_csv(run/"resources.csv"),previous=previous_data,source_verification=verification,
              generation=dict(training_performed=False,bootstrap_seed=20261031,bootstrap_samples=10000,
                              diagnostic_bank="validation master 20261031/domain40; first two pairs per opponent",
                              simulation_environment_seed=7,simulation_policy_seed=derived_seed(7,11)))
    (output/"paper_data.json").write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (output/"simulation_traces.json").write_text(json.dumps(traces,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print("보고서 자료 수집 완료:",output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--verify-only",action="store_true",help="추론 없이 보존 소스와 아카이브 경로만 검증합니다.")
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("자료는 smoke_test_mc_td 내부에만 저장할 수 있습니다.")
    if args.verify_only:
        run=args.run.resolve()
        repo=run.parents[2]
        result=verify_source_snapshot(run,repo)
        result["previous_run"]=str(find_previous_run(repo/"smoke_test_mc_td"))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        prepare(args.run.resolve(),args.output.resolve())
