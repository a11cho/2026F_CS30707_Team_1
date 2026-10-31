"""학습 로그·체크포인트에서 토의용 한국어 Markdown 보고서를 생성합니다.

원본 로그를 수정하지 않습니다. 별도 자원 측정 파일은 profile_runs.py로 생성합니다.
"""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics as stats
import sys
from zoneinfo import ZoneInfo


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return [{k: (v if k in ("split", "opponent") else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def mean(rows, key):
    return stats.mean(row[key] for row in rows)


def pct(value):
    return str((Decimal(str(value))*100).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))


def table(headers, rows):
    # Python 버전 문자열 등에 들어 있는 |가 Markdown 열을 나누지 않게 합니다.
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                      ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows])


def generate(run_dir, profile_path, output):
    repo = run_dir.parents[2]
    sys.path.insert(0, str(repo))
    import numpy as np
    from territory_rl import ExperimentConfig
    from smoke_test_mc_td.config import RunConfig
    from smoke_test_mc_td.learning import load_models, TabularControl
    from smoke_test_mc_td.runner import derived_seed, evaluate, summarize
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))["profiles"]
    config = RunConfig(**manifest["config"])
    assert manifest["status"] == "complete" and len(data["groups"]) == len(data["selected"]) == 36
    assert all(p["exact_q_reproduction"] for p in profiles)
    mismatches = [name for name, digest in manifest["source_sha256"].items()
                  if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    # 학습 전 기준선은 후보 선정에 쓰지 않고 완료 후 같은 test에서 참고용으로 평가합니다.
    baseline_rows = evaluate(config, ExperimentConfig("local", "individual", "gross", 0),
                             [TabularControl("sarsa", config.alpha, config.gamma)], "test")
    baseline = summarize(baseline_rows)
    records = []
    all_rows = []
    for group in data["groups"]:
        cid, algorithm = group["condition"], group["algorithm"]
        selected = next(s for s in data["selected"] if s["condition"] == cid and s["algorithm"] == algorithm)
        training = [read_csv(run_dir / cid / algorithm / f"seed_{seed}" / "training.csv") for seed in config.seeds]
        first = [row for rows in training for row in rows[:20]]
        last = [row for rows in training for row in rows[-20:]]
        growth = stats.mean((rows[-1]["q_states"]-rows[59]["q_states"]) / rows[-1]["q_states"] for rows in training)
        known = sum(r["known_samples"] for r in last) / sum(r["learner_samples"] for r in last)
        selected_test = read_csv(run_dir / cid / algorithm / "selected_test.csv")
        all_rows.extend(selected_test)
        for candidate in group["candidates"]:
            directory = run_dir / cid / algorithm / f"seed_{candidate['seed']}"
            all_rows.extend(read_csv(directory / "training.csv"))
            all_rows.extend(read_csv(directory / "validation.csv"))
            all_rows.extend(read_csv(directory / "untrained_validation.csv"))
        models, _ = load_models(run_dir / selected["checkpoint"])
        model = models[0]
        preference = np.zeros(3)
        alive_keys, tie_keys, allzero = 0, 0, 0
        for key, values in model.q.items():
            if not key[0]:
                continue
            legal = np.flatnonzero(key[-9:])
            best = legal[np.flatnonzero(values[legal] == values[legal].max())]
            for action in best:
                preference[0 if action == 0 else 1 if action <= 4 else 2] += 1 / len(best)
            alive_keys += 1
            tie_keys += int(len(best) > 1)
            allzero += int(np.all(values[legal] == 0))
        preference /= max(1, alive_keys)
        records.append(dict(group=group, selected=selected, training=training, test=selected_test,
                            first_return=mean(first,"mean_return"), last_return=mean(last,"mean_return"),
                            first_score=mean(first,"score"), last_score=mean(last,"score"),
                            q_states=stats.mean(rows[-1]["q_states"] for rows in training),
                            updates=stats.mean(rows[-1]["updates"] for rows in training),
                            late_growth=growth, late_known=known, preference=preference,
                            zero_fraction=allzero/max(1,alive_keys), tie_fraction=tie_keys/max(1,alive_keys),
                            max_abs_q=max(float(np.abs(v).max()) for v in model.q.values())))
    assert len(all_rows) == 51840
    assert all(r["ticks"] == 32 and r["learner_rejected"] == 0 for r in all_rows)
    assert all(np.isfinite(r["mean_return"]) for r in all_rows)
    start = datetime.strptime(run_dir.name,"%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    end = datetime.fromtimestamp((run_dir/"manifest.json").stat().st_mtime, timezone.utc)
    elapsed = (end-start).total_seconds()
    kst = ZoneInfo("Asia/Seoul")
    lines = []
    def add(value=""):
        lines.append(value)
    add("# MC 제어 / SARSA(0) 전체 그리드 스모크테스트 토의 보고서")
    add(f"\n대상 실행: `{run_dir.name}` · 실행일: {start.astimezone(kst):%Y-%m-%d} · 기준: 저장된 180개 학습 후보와 36개 선택 모델.\n")
    add("## 1. 먼저 읽을 결론")
    add("\n1. **연결·재현 검증은 성공했습니다.** 180개 체크포인트, 36개 조건/알고리즘 평가, 51,840개 경기 기록을 확인했습니다. 모든 경기는 32틱이고 학습자의 금지 행동 제출은 0건입니다. 별도 대표 후보 4개 재학습 결과도 기존 Q-table과 정확히 일치했습니다.")
    add("2. **충분한 수렴이나 안정적인 성능 개선은 확인하지 못했습니다.** 마지막 20에피소드에도 새 key가 계속 추가되고, 중간 greedy 평가·Q 변화량·TD error가 없어 수렴 여부를 엄밀히 판정할 수 없습니다.")
    add("3. **aware의 정보 증가가 표 기반 학습에서는 상태 희소성을 크게 높였습니다.** 팀원 정보가 실제로 key에 들어가지만, 재방문이 적어 이번 예산에서 그 정보를 학습에 활용했다고 보기 어렵습니다.")
    add("4. **학습 상대와 평가 상대의 차이가 큽니다.** random 상대에서의 승리는 높지만 강한 scripted 상대에서는 성적이 낮습니다. 고정 expand 아군 두 명의 기여도 있어 팀 승률을 학습자 개인의 실력으로 해석할 수 없습니다.")
    train_rows=[row for r in records for rows in r["training"] for row in rows]
    assert len(train_rows)==14400 and all(row["score"]==1 for row in train_rows)
    add("이번 **학습 경기 14,400개는 전부 팀 승리**였습니다. 탐험률이 1에 가까운 초기에도 이미 이기므로 학습 승률에는 향상을 구별할 여지가 없습니다. terminal 조건의 최종 보상은 수집된 모든 학습 경기에서 +1로 같았습니다. 보상 평탄화를 수렴으로 오인하지 않는 것이 핵심입니다.")
    add("5. **최고 seed·최고 조건을 고른 결과는 알고리즘 전체 성능 추정이 아닙니다.** 아래의 seed 평균은 validation에 대한 기술통계이며, test는 선택된 후보에만 있습니다. 통계적 유의성이나 알고리즘의 일반적 우열을 주장하지 않습니다.\n")
    add("## 2. 전체 설정과 실험 범위\n")
    params = [("맵 height / width", "6 / 6; 총 36타일, 초기 팀별 18타일"),
              ("에이전트 / 학습자", "3 대 3, 총 6명 / 1명"),
              ("학습자 ID", "red_0 또는 blue_0; 에피소드마다 진영 교대, 같은 Q-table 유지"),
              ("아군 두 명 / 학습 상대", "고정 expand 정책 / 고정 random 정책 세 명"),
              ("초기 배치 / 규칙 버전", "frontier / territory_v0; 초기 양쪽 진영 대칭 배치"),
              ("종료 지평 max_ticks", "32; 실제 termination, 마지막 bootstrap=0"),
              ("부활 지연 respawn_delay", "2; 사망 뒤 후속 2틱 대기, 후보 부족이면 더 지연"),
              ("관측 반경 / local_grid", "2 / float32 (6,5,5)"),
              ("나머지 관측 shape", "self_features=(6,), teammate_features=(2,14), teammate_feature_mask=(2,14), action_mask=(9,)"),
              ("행동", "NOOP 1 + 이동 4 + 공격 4 = 9; mask 안에서만 선택"),
              ("충돌·사망", "같은 목적지 이동 전원 실패, swap 실패, 긴 순환 허용; 공격 후 사망자는 이동 취소"),
              ("공격 cooldown", "공격 직후 다음 틱 공격 금지; 생존·경계·cooldown을 action mask에 반영"),
              ("모델", "독립 tabular Q; 신경망·파라미터 공유·GPU 사용 없음; compact_v1"),
              ("MC", "on-policy first-visit control, 고정 alpha; 매 에피소드 종료 후 갱신"),
              ("TD", "on-policy 1-step SARSA(0); V를 학습하는 TD(0) prediction과 구별"),
              ("alpha / gamma / 초기 Q", "0.1 / 0.99 / 0"),
              ("탐험 epsilon", "에피소드 0..79에서 1.0 → 0.1 선형 감소; 평가 시 0, Q 동률은 무작위"),
              ("학습량", "후보당 80에피소드 × 32틱 = 2,560 환경 틱·학습자 전이"),
              ("학습 seed", ", ".join(map(str,config.seeds))),
              ("그리드", "보상 9 × 관측 2 × 알고리즘 2 × seed 5 = 180후보"),
              ("validation", "후보당 상대 4종 × 10 seed × 2진영 = 80경기; 학습 전 기준선도 동일"),
              ("test", "선택 모델당 상대 4종 × 30 seed × 2진영 = 240경기"),
              ("총 경기 / 환경 틱", "학습 14,400 + validation 14,400 + 기준선 validation 14,400 + test 8,640 = 51,840 / 1,658,880틱"),
              ("학습 구간 환경 틱 / 전체 6명 전이", "460,800 / 2,764,800; 실제 갱신 횟수와 구분"),
              ("원본 결과 상태", manifest["status"])]
    add(table(["항목","실제 값"],params))
    add("\n보상 축은 아래와 같습니다. 모든 신호는 기존 환경의 reward.py를 그대로 사용합니다.\n")
    add(table(["보고서 약어","기본 보상","종료 계수 b"], [
        ("I", "본인 capture credit; 공동 점령은 공격자 수로 나눔", "0, 1"),
        ("N", "팀 면적 순증가 / 3", "0, 1"), ("G", "팀 총획득 타일 / 3", "0, 1"),
        ("A", "현재 팀 면적 / 3", "0, 1"), ("T", "중간 보상 0", "1")]))
    add("\n종료 보상은 b × 승/무/패(+1/0/-1)입니다. I에는 team_signal이 영향을 주지 않아 중복을 제거했고 T,b=0은 항상 0이므로 제외했습니다. N과 I는 목적도 달라 단순 보상 공유 효과만 비교하는 조건은 아닙니다. G 대 I가 같은 gross 사건의 귀속 비교에 더 가깝습니다.\n")
    def label(cid):
        obs,reward,signal,bonus = cid.split("_")
        return f"{'L' if obs=='local' else 'W'}-{'I' if reward=='individual' else {'net':'N','gross':'G','area':'A','terminal':'T'}[signal]}-{bonus}"
    add("표 약어: L=local, W=aware, b0/b1=종료 보너스 계수 0/1. 실제 디렉터리 이름과 대응표는 부록에 있습니다. 아래 점수·비율은 별도 표시가 없으면 %이며, 점수는 승리 1·무승부 0.5의 평균입니다.\n")
    add("## 3. 모델 선정 방식과 전체 대표 모델\n")
    add("각 조건·알고리즘의 seed 5개 중 validation 점수가 가장 높은 것을 선택했습니다. 동점이면 면적 비율, 그것도 같으면 seed 설정 순서(7,19,31,43,59)를 사용합니다. 알고리즘 전체 대표도 같은 validation 기준으로 선택했습니다. test 결과를 보고 다시 선정하지 않았습니다.\n")
    rep_rows=[]
    for algorithm, selected in data["representatives"].items():
        rep_rows.append([algorithm.upper(),label(selected["condition"]),selected["seed"],pct(selected["validation"]["score"]),
                         pct(selected["test"]["score"]),pct(selected["test"]["win"]),pct(selected["test"]["area_fraction"]),pct(selected["test"]["known_state_fraction"])])
    rep_rows.append(["학습 전 Q=0","공통 기준선","—","36.88",pct(baseline["score"]),pct(baseline["win"]),pct(baseline["area_fraction"]),"0.00"])
    add(table(["모델","조건","seed","Val 점수","Test 점수","Test 승률","Test 면적","Test 기방문"],rep_rows))
    add("\nQ=0 test 기준선은 보고서 작성 시 같은 test seed·상대·진영으로 추가 평가한 값이며 기존 results.json에는 없습니다. Q가 모두 0이라 허용 행동 중 무작위로 선택하고 아군 두 명은 expand입니다. local/aware 정보는 이 정책의 선택에 영향을 주지 않습니다. 기준선은 모델 선정이나 그리드 수정에 사용하지 않았습니다.\n")
    add("## 4. 모든 그리드의 성능과 seed 편차\n")
    add("Val 평균±표본표준편차는 독립 학습 seed 5개의 값입니다. Δ기준선은 이 평균에서 같은 조건의 학습 전 validation 평균을 뺀 퍼센트포인트입니다. Test 열은 최고 seed 한 개의 결과이며 seed 평균이 아닙니다.\n")
    rows=[]
    for r in records:
        g,s=r["group"],r["selected"]
        base=stats.mean(c["untrained_validation"]["score"] for c in g["candidates"])
        rows.append([label(s["condition"]),s["algorithm"],f"{pct(g['validation_score_mean'])} ± {pct(g['validation_score_std'])}",
                     f"{100*(g['validation_score_mean']-base):+.2f}",s["seed"],pct(s["validation"]["score"]),pct(s["test"]["score"]),
                     pct(s["test"]["win"]),pct(s["test"]["area_fraction"]),pct(s["test"]["known_state_fraction"])])
    add(table(["조건","방법","5-seed Val 평균±SD","Δ기준선 pp","선택 seed","선택 Val","Test 점수","승률","면적","기방문"],rows))
    add("\n평균 성능이 낮은 조건에서도 최고 seed만 보면 좋아 보일 수 있습니다. 반복된 validation 선정과 80경기의 작은 표본 때문에 선택 편향이 남으며, 표의 미세한 차이를 통계적으로 유의한 차이라고 해석하지 않습니다.\n")
    add("## 5. 수렴 정도: 관측 가능한 학습 진행과 관측 불가능한 부분\n")
    add("**이번 로그만으로 어느 모델도 수렴했다고 판정하지 않습니다.** Q 변화량, TD error, 고정 검증 집합의 학습 중간 평가, true Q 오차를 기록하지 않았습니다. 또한 고정 alpha=0.1·최종 epsilon=0.1·부분 관측·상태 요약이므로 강의의 tabular 수렴 정리 조건을 그대로 적용할 수 없습니다.\n")
    add("아래는 seed 5개의 학습 추이 요약입니다. 초기/후기=1–20/61–80에피소드 평균. 보상은 **같은 행 안에서만** 비교하세요. 학습 중 epsilon도 감소하고 경기 seed도 달라지므로 보상 증가를 순수 정책 개선으로 단정할 수 없습니다. 후기 신규 비율은 마지막 20에피소드에 추가된 key 수 / 최종 key 수입니다. 후기 기방문은 학습 중 샘플의 이미 저장된 key 비율이며 MC/SARSA의 저장 시점 차이도 있습니다.\n")
    rows=[]
    for r in records:
        s=r["selected"]
        rows.append([label(s["condition"]),s["algorithm"],f"{r['first_return']:.3f} → {r['last_return']:.3f}",
                     f"{pct(r['first_score'])} → {pct(r['last_score'])}",f"{r['q_states']:.0f}",f"{r['updates']:.0f}",
                     pct(r["late_growth"]),pct(r["late_known"]),"수렴 미확인"])
    add(table(["조건","방법","학습 보상 초기→후기","학습 점수 초기→후기","최종 key 평균","갱신 평균","후기 신규 %","후기 기방문 %","판정"],rows))
    add("\n관측·알고리즘별 아래 집계는 보상 조건 9개와 seed 5개를 단순 평균한 기술통계입니다. 보상 조건을 독립 반복 실험처럼 취급한 통계 검정은 하지 않았습니다.\n")
    aggregate=[]
    for obs in ("local","aware"):
        for a in ("mc","sarsa"):
            subset=[r for r in records if r["selected"]["condition"].startswith(obs+"_") and r["selected"]["algorithm"]==a]
            candidates=[c for r in subset for c in r["group"]["candidates"]]
            aggregate.append([obs,a,f"{stats.mean(r['q_states'] for r in subset):.1f}",
                              pct(stats.mean(c["validation"]["known_state_fraction"] for c in candidates)),
                              pct(stats.mean(r["late_growth"] for r in subset)),
                              pct(stats.mean(c["validation"]["score"] for c in candidates))])
    add(table(["관측","방법","최종 key 평균","Val 기방문 평균 %","후기 신규 평균 %","Val 점수 평균 %"],aggregate))
    add("\n**관측된 학습 양상:** local 개인 보상 b0의 평균 학습 보상은 MC 0.503→1.000, SARSA 0.525→1.295로 늘지만, 학습 승률은 처음부터 100%입니다. aware 같은 조건은 MC 0.510→0.660, SARSA 0.525→0.605로 증가 폭이 작습니다. 팀 area 보상은 약 300점의 큰 양의 신호라 Q의 크기도 크지만, 점수 크기가 성능 향상은 아닙니다. terminal 보상은 항상 +1이어서 일정한 return만으로 정책 수렴을 판단할 수 없습니다.\n")
    add("\n해석: aware는 팀원 좌표·행동 등을 key에 추가하므로 2,560개 학습 전이 대부분이 서로 다른 key로 갈 수 있습니다. 이는 정보 공유가 본질적으로 나쁘다는 증거가 아니라, 현재 표현·표 기반 모델·예산의 조합이 정보를 재사용하기 어렵다는 증거입니다. 남은 시간·절대 좌표 구간도 key를 나눕니다. 사망 대기는 일반 생존 상황보다 반복되기 쉬워 전체 기방문 비율을 높일 수 있습니다.\n")
    add("## 6. 모델의 양상: 상대별 성적과 상태·행동 선호\n")
    add("### 6.1 모든 선택 모델의 상대별 test 점수\n")
    rows=[]
    for r in records:
        s=r["selected"]
        rows.append([label(s["condition"]),s["algorithm"],*[pct(mean([row for row in r["test"] if row["opponent"]==op],"score"))
                     for op in config.eval_opponents],f"{s['test']['deaths']:.2f}",pct(s["test"]["dead_fraction"]),f"{s['test']['captures']:.2f}"])
    rows.append(["Q=0 기준","—",*[pct(mean([row for row in baseline_rows if row["opponent"]==op],"score"))
                 for op in config.eval_opponents],f"{baseline['deaths']:.2f}",pct(baseline["dead_fraction"]),f"{baseline['captures']:.2f}"])
    add(table(["조건","방법","random","expand","aggressive","defend","본인 사망/경기","대기 %","팀 점령/경기"],rows))
    add("\n각 상대 열은 60경기입니다. 사망과 대기는 학습자 기준이고 captures는 **팀 전체 점령 사건 수**입니다. 같은 타일 재점령도 포함하므로 최종 면적 증가와 다릅니다. random 상대의 높은 성적은 expand 아군 두 명의 도움을 포함하며 개인 정책의 공격·협력 능력을 입증하지 않습니다.\n")
    add("### 6.2 저장된 Q-table의 행동 선호 진단\n")
    add("실제 경기의 행동 빈도 로그는 없으므로, 저장된 **생존 key를 동일 가중치로 한 번씩** 조회한 greedy 행동 선호를 계산했습니다. 최대값 동률은 균등 분배하고 불법 행동은 제외합니다. 아래 수치는 실제 rollout의 행동 비율이 아니며 미방문 key도 포함하지 않습니다. Q=0 허용값 비율과 동률 비율이 높으면 greedy 평가도 상당 부분 무작위입니다. Q 절댓값은 보상 스케일이 달라 행 사이 우열 비교에 쓰지 않습니다.\n")
    rows=[]
    for r in records:
        s=r["selected"]
        rows.append([label(s["condition"]),s["algorithm"],*[pct(v) for v in r["preference"]],pct(r["zero_fraction"]),pct(r["tie_fraction"]),f"{r['max_abs_q']:.4f}"])
    add(table(["조건","방법","정지 선호 %","이동 선호 %","공격 선호 %","허용 Q 전부0 %","최대값 동률 %","max abs Q"],rows))
    add("\n전체 대표인 W-I-b0에서 생존 key의 허용 Q가 전부 0인 비율은 MC 약 90.69%, SARSA 약 98.16%입니다. 따라서 저장된 key에서도 대부분 행동을 구별하지 못합니다. MC에서 종료 보너스를 더하면 Q=0 key는 줄지만, 모두 이기는 경험에 양의 return을 퍼뜨리는 효과도 있으므로 가치가 0이 아니게 됐다는 사실만으로 좋은 행동을 구분했다고 볼 수 없습니다.\n")
    add("\n전술 replay나 행동 시퀀스가 없어 특정 모델이 의도적으로 방어·협력·자살 이동을 했다고 단정할 수 없습니다. 이 표와 사망·점령 지표는 다음 관찰 시나리오를 고르는 진단 자료입니다.\n")
    add("## 7. 시간·메모리·CPU\n")
    add("### 7.1 실제 PC와 원본 전체 실행\n")
    add(table(["항목","확인값 / 출처"],[
        ("CPU", "Intel Core Ultra 7 258V; Win32_Processor 조회"),
        ("물리 / 논리 코어", "8 / 8; 조회값"),
        ("OS 보고 MaxClockSpeed", "2200 MHz; 현재 실효 클럭이나 최대 터보 실측값 아님"),
        ("OS 보고 물리 RAM", f"33,834,045,440 bytes = {33834045440/2**30:.2f} GiB; Win32_ComputerSystem 조회"),
        ("운영체제", manifest["platform"]),
        ("Python", manifest["python"]),
        ("NumPy", manifest["numpy"]),
        ("Gymnasium / PettingZoo", f"{importlib.metadata.version('gymnasium')} / {importlib.metadata.version('pettingzoo')}; 보고서 시점 조회"),
        ("실행 시작", f"{start.astimezone(kst):%Y-%m-%d %H:%M:%S.%f %Z}; run ID 시각"),
        ("완료 파일 시각", f"{end.astimezone(kst):%Y-%m-%d %H:%M:%S.%f %Z}; manifest 최종 수정 시각"),
        ("전체 경과 시간 추정", f"{elapsed:.2f}초 = {elapsed/60:.2f}분; 학습·평가·기준선·직렬화·대기 포함"),
        ("원본 CPU 시간 / peak RAM", "기록하지 않음; 소급 실측 불가"),
        ("병렬화", "명시적 worker 없음; 후보·경기를 순차 실행, GPU 사용 없음")]))
    add("\n파일 시각 기반 경과 시간은 프로파일러로 계측한 순수 학습 시간이 아닙니다. 실행 도중 다른 작업·스케줄링·전원 상태의 영향이 포함될 수 있으며, 원본 전체 CPU 사용률·피크 메모리·온도·실효 클럭은 알 수 없습니다.\n")
    add("### 7.2 보고서 작성 시 대표 후보 4개 별도 재학습 실측\n")
    add("각 후보를 독립 Python 프로세스에서 같은 seed·설정으로 80에피소드 재학습했습니다. import와 기존 체크포인트 로드 후부터 train() 반환까지 perf_counter와 process_time으로 측정합니다. validation/test·체크포인트 저장은 제외하고 관측 생성·행동 선택·학습 지표 집계는 포함합니다. 기존 기준 모델 한 개도 프로세스에 올라가 있으므로 메모리는 학습자만의 크기가 아닙니다.\n")
    rows=[]
    for p in profiles:
        exp=p["experiment"]
        cid=f"{exp['observation_mode']}_{exp['reward_mode']}_{exp['team_signal']}_b{exp['terminal_bonus']:g}"
        rows.append([label(cid),p["algorithm"],p["seed"],f"{p['wall_seconds']:.3f}",f"{p['cpu_seconds']:.3f}",
                     f"{p['ticks']/p['wall_seconds']:.0f}",f"{p['one_core_cpu_percent']:.1f}",
                     f"{p['one_core_cpu_percent']/8:.1f}",f"{p['before']['working_set_bytes']/2**20:.2f}",
                     f"{p['after']['peak_working_set_bytes']/2**20:.2f}",f"{p['after']['private_commit_bytes']/2**20:.2f}","일치"])
    add(table(["조건","방법","seed","벽시계 초","CPU 초","학습 틱/초","1코어 기준 CPU %","8코어 대비 %","시작 WS MiB","프로세스 peak WS MiB","종료 private MiB","Q 재현"],rows))
    add("\nWS는 Windows working set(물리 메모리에 상주하는 프로세스 페이지), peak WS는 **프로세스 시작 이후** 최대값입니다. private commit은 물리 RAM 사용량과 같지 않습니다. 메모리는 GetProcessMemoryInfo로 조회했습니다. CPU%는 CPU초/경과초이며, 8코어 대비 값은 이를 8로 나눈 환산값으로 시스템 전체 CPU 실측 그래프가 아닙니다. 각 경우 한 번만 측정했으므로 성능 벤치마크의 안정된 평균으로 보지 마세요. 네 후보의 최종 Q값·key·갱신 횟수는 기존 결과와 정확히 일치했습니다.\n")
    files=[p for p in run_dir.rglob("*") if p.is_file()]
    checkpoints=list(run_dir.glob("*/*/seed_*/model.json.gz"))
    sizes=[p.stat().st_size for p in checkpoints]
    add("### 7.3 모델 저장량과 확장 비용\n")
    add(table(["항목","값"],[
        ("원본 실행 파일 총량",f"{sum(p.stat().st_size for p in files)/2**20:.2f} MiB / {len(files)}파일"),
        ("압축 체크포인트",f"{len(sizes)}개; 최소 {min(sizes)/1024:.2f}, 평균 {stats.mean(sizes)/1024:.2f}, 최대 {max(sizes)/1024:.2f} KiB"),
        ("재학습 후보 Q 배열 원소만",f"{min(p['q_array_bytes'] for p in profiles)/1024:.2f}–{max(p['q_array_bytes'] for p in profiles)/1024:.2f} KiB"),
        ("Python 객체 비용", "튜플·정수·dict·배열 객체 오버헤드는 Q 배열 원소 크기에 포함하지 않음"),
        ("시간 병목 해석", "전체 경기는 학습 14,400개보다 평가·기준선 37,440개가 더 많음; 순수 train 시간만으로 전체 실행 시간을 예측하면 과소평가")]))
    add("\n대표 후보가 약 2초에 재학습됐다고 원본 180회를 같은 속도로 실행했다고 추정하지 않습니다. 당시 자원 상태가 다르고 평가·저장량도 다릅니다. RAM은 이번 작은 문제에서는 가볍지만 미방문 key가 계속 증가하는 aware 표현은 더 긴 학습에서 메모리 증가도 측정해야 합니다.\n")
    add("## 8. 재현 방법과 정확한 seed 규칙\n")
    add("학습 master seed 순서: **7, 19, 31, 43, 59**. 조건·알고리즘이 달라도 같은 master 목록과 설정을 사용합니다. 실제 에피소드 seed는 master 자체가 아니라 아래 규칙으로 파생합니다.\n")
    add('''```python
import numpy as np
def derived_seed(master, domain, index=0):
    return int(np.random.SeedSequence([master, domain, index]).generate_state(1)[0])

# 학습 에피소드 i = 0..79
env_seed = derived_seed(master, 10, i)
policy_seed = derived_seed(master, 11, i)
# 학습 역할 j의 행동 RNG / 전체 ID 인덱스 k의 봇 RNG
learner_rng_seed = derived_seed(policy_seed, 1, j)
bot_rng_seed = derived_seed(policy_seed, 2, k)
# validation pair p = 0..9 / test pair p = 0..29
val_env = derived_seed(20261031, 20, p)
val_policy = derived_seed(20261031, 21, p)
test_env = derived_seed(20261031, 30, p)
test_policy = derived_seed(20261031, 31, p)
```''')
    add("\n각 평가 pair는 같은 환경 seed로 team=0,1을 순서대로 실행합니다. 초기 배치는 대칭이지만 봇 RNG는 전체 ID 인덱스로 나뉘므로 모든 행동열까지 완전히 좌우 거울상이라는 뜻은 아닙니다. 상대 종류 사이에도 같은 seed bank를 재사용합니다.\n")
    add(table(["master","학습 환경 seed i=0,1,2","학습 policy seed i=0,1,2"],
              [[seed,", ".join(str(derived_seed(seed,10,i)) for i in range(3)),", ".join(str(derived_seed(seed,11,i)) for i in range(3))] for seed in config.seeds]))
    for title,domain,count in (("validation 환경",20,10),("validation 정책",21,10),("test 환경",30,30),("test 정책",31,30)):
        add(f"\n**{title} seed 전체(인덱스 순서):**\n\n`"+", ".join(str(derived_seed(20261031,domain,i)) for i in range(count))+"`\n")
    add("\n프로젝트 루트 PowerShell에서 동일 전체 실행:\n")
    add('''```powershell
$rlPython = 'C:\\Users\\dohyu\\anaconda3\\envs\\rlTeam\\python.exe'
& $rlPython -B -X utf8 -u -m smoke_test_mc_td.run --preset full
```''')
    add("\n새 실행 폴더가 생성됩니다. Gymnasium/PettingZoo의 설치 버전은 원래 pyproject 명시 버전과 다를 수 있으므로 위 실측 버전을 재현하세요. 원본 manifest에는 Python·NumPy·OS와 소스 SHA256이 저장되어 있습니다. 보고서 작성 시 원본에 기록된 모든 소스 해시가 현재 파일과 일치함을 확인했습니다. 추가한 분석 스크립트는 원래 학습 코드나 결과를 수정하지 않습니다.\n")
    add("## 9. 토의할 질문과 다음 실험 우선순위\n")
    add("1. **수렴을 먼저 볼 수 있는 작은 문제로 만들 것인가?** 학습자 하나·고정 전술 배치·짧은 지평·더 작은 관측 key로 시작하면 MC/SARSA의 반복 방문과 가치 전파를 확인하기 쉽습니다. 현재 aware에서 예산만 늘리기 전에 표현의 재사용 가능성을 확인해야 합니다.")
    add("2. **정보 공유 비교에서 무엇을 고정할 것인가?** local/aware가 같은 길이여도 유효 key 수는 크게 다릅니다. 지금 결과를 정보 공유 자체의 이득/손해로 해석하지 말고, 팀원 정보의 더 거친 요약 또는 함수 근사를 별도 조건으로 검토할 수 있습니다.")
    add("3. **성공을 팀 승리와 개인 기여 중 무엇으로 판단할 것인가?** expand 아군 두 명이 random 상대를 압도할 수 있습니다. 학습자 자신의 점령 credit·행동·사망 원인과 아군 제거/교체 대조군이 필요합니다.")
    add("4. **학습 상대 분포를 바꿀 것인가?** random만으로 학습한 뒤 scripted 세 종류까지 일반화하기 어렵습니다. 먼저 train=random/test=random과 상대 일반화를 분리해서 보고, 이후 고정된 상대 혼합 분포를 사전 정의하는 것이 좋습니다.")
    add("5. **수렴을 측정할 로그를 넣을 것인가?** 일정 간격의 고정 greedy 평가, 같은 probe key 집합의 Q 변화, TD error/MC target 변화, key 방문 횟수, 생존/사망별 기방문률, 실제 행동 빈도, 프로세스 CPU/RAM과 phase별 타이머를 추가하면 논의가 훨씬 명확해집니다.")
    add("6. **최고 seed 선정과 알고리즘 비교를 분리할 것인가?** 배포용 대표는 validation으로 고르되 연구용 평균 test는 seed 5개 모두에서 보고해야 합니다. 현재 test를 본 후 변경하는 다음 실험에는 새로운 최종 hold-out bank를 마련해야 합니다.")
    add("7. **종료 보상·gamma·학습률을 어떻게 볼 것인가?** terminal SARSA의 값은 한 스텝씩 전파되고 짧은 학습·희소 재방문에서는 초반 행동까지 퍼지기 어렵습니다. 고정 alpha와 epsilon 설정 때문에 장기 학습만으로 엄밀 수렴을 보장하지 않습니다. 설정 변경은 이번 결과와 분리된 실험으로 기록하세요.\n")
    add("## 부록 A. 각 조건의 seed별 validation 점수\n")
    rows=[]
    for r in records:
        s=r["selected"]
        by_seed={c["seed"]:c for c in r["group"]["candidates"]}
        rows.append([label(s["condition"]),s["algorithm"],*[pct(by_seed[seed]["validation"]["score"]) for seed in config.seeds],s["seed"]])
    add(table(["조건","방법",*[str(seed) for seed in config.seeds],"선택 seed"],rows))
    add("\n## 부록 B. 약어와 원본 파일 위치\n")
    add(table(["약어","실제 조건 디렉터리"],[[label(cid),f"`{cid}`"] for cid in dict.fromkeys(s["condition"] for s in data["selected"])]))
    rel="runs/"+run_dir.name
    add(f"\n- [원본 manifest]({rel}/manifest.json): 설정·실행 버전·소스 해시·완료 상태\n- [전체 결과 JSON]({rel}/results.json): 180후보·36선택 모델과 대표 모델\n- [선택 결과 CSV]({rel}/selected.csv): 조건별 최종 점수\n- 각 `{rel}/<조건>/<알고리즘>/seed_<seed>/training.csv`, `validation.csv`, `untrained_validation.csv`, `model.json.gz`\n- 각 `{rel}/<조건>/<알고리즘>/selected_test.csv`: 상대·진영·pair별 test\n- [별도 자원 실측 원자료](resource_profile.json) 및 [측정 스크립트](profile_runs.py)\n- [이 보고서 생성 스크립트](analyze_results.py)\n")
    add("\n분석 재생성: 프로젝트 루트에서 아래 명령을 실행합니다. 원본 그리드를 다시 학습하지 않습니다.\n")
    add(f'''```powershell
& $rlPython -B -X utf8 smoke_test_mc_td/analyze_results.py smoke_test_mc_td/{rel} --profile smoke_test_mc_td/resource_profile.json --output smoke_test_mc_td/SMOKE_TEST_REPORT_20261031.md
# 자원 측정은 실행 때마다 달라집니다. 다시 측정하려면:
& $rlPython -B -X utf8 smoke_test_mc_td/profile_runs.py smoke_test_mc_td/{rel} | Set-Content -Encoding utf8 smoke_test_mc_td/resource_profile.json
```''')
    output.write_text("\n\n".join(value.strip() for value in lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(output),bytes=output.stat().st_size,grid_rows=len(records),source_hash_match=True,
                          logged_episodes=len(all_rows),baseline_test=baseline,resource_reproduced=True),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("보고서는 smoke_test_mc_td 안에만 저장할 수 있습니다.")
    generate(args.run_dir.resolve(),args.profile.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
