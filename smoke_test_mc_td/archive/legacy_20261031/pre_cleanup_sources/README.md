# MC 제어와 SARSA(0) 스모크테스트

강의 5의 on-policy 제어를 구현했습니다. MC는 에피소드 종료 후 first-visit
return으로, SARSA(0)는 매 틱 한 스텝 TD 타깃으로 행동가치 Q를 갱신합니다.
두 방법 모두 행동 마스크를 적용한 epsilon-greedy 정책과 Q-table을 사용합니다.

현재 기본은 **3대3 양 팀 여섯 명 전원 학습**입니다. 기존 `20261031T093149737642Z`
결과와 토의 보고서는 변경 전의 한 명 학습 실험이며, 여섯 명 학습 결과가 아닙니다.
기존 로그와 체크포인트는 그대로 보존합니다.

## 실행 방법

프로젝트 루트 `2026F_CS30707_Team_1`에서 PowerShell로 실행하세요.

```powershell
$rlPython = 'C:\Users\dohyu\anaconda3\envs\rlTeam\python.exe'
# 계산·재현성·사망/부활·API 검사
& $rlPython -B -m unittest smoke_test_mc_td.test_smoke -v
# 짧은 연결 검사
& $rlPython -B -m smoke_test_mc_td.run --preset quick
# 전체 보상·관측 그리드
& $rlPython -B -m smoke_test_mc_td.run --preset full
# 양 팀 여섯 명을 독립적으로 학습
& $rlPython -B -m smoke_test_mc_td.run --preset full --learners 6
# SARSA만 학습하고 에피소드 수 증가
& $rlPython -B -m smoke_test_mc_td.run --preset full --algorithm sarsa --episodes 500
```

rlTeam의 Python을 직접 지정하므로 conda activate는 필요 없습니다. `-B`는
환경 코드에 `__pycache__`가 생기는 것을 막습니다. 결과는 이 폴더의 `runs/`
아래에만 저장합니다. NumPy·Gymnasium·PettingZoo가 필요하며 unittest는 기본
모듈입니다. matplotlib·Jupyter는 명령행 실행에 필요하지 않습니다.
기존 환경의 `tests/`가 사용하는 pytest 검사는 이 unittest 실행과 별개입니다.

| 설정 | quick | full |
| --- | --- | --- |
| 알고리즘 | MC, SARSA | MC, SARSA |
| 관측 | local, aware | local, aware |
| 보상 | 개인 점령, 종료 보너스 0 | 아래의 보상 조합 9개 |
| 학습 seed | 7, 19, 31, 43, 59 | 동일 |
| seed별 학습 | 4에피소드 × 12틱 | 80에피소드 × 32틱 |
| 학습 실행 수 | 20개 | 180개 |
| validation | 양 팀 각각 random, seed 1개 × 양쪽 진영: 총 4경기 | 두 팀 × 상대 4종 × seed 10개 × 양쪽 진영: 총 160경기 |
| test | 양 팀 각각 random, seed 1개 × 양쪽 진영: 총 4경기 | 두 팀 × 상대 4종 × seed 30개 × 양쪽 진영: 총 480경기 |

quick은 실행 흐름을 확인하는 용도이며 성능 비교용이 아닙니다. full도 작은
스모크테스트 예산이며 80에피소드로 충분히 학습된다고 보장하지 않습니다.

## 코드 구조와 읽을 순서

| 파일 | 핵심 클래스·함수 | 역할 |
| --- | --- | --- |
| learning.py | TabularControl | Q-table, 행동 선택, MC/SARSA 업데이트 |
| encoding.py | encode(), bucket() | 관측을 이산 튜플 key로 요약 |
| runner.py | episode() | 여섯 행동 수집, 동시 전이, 경험 수집·업데이트 |
| runner.py | train(), evaluate() | 여러 학습 에피소드와 평가 경기 실행 |
| config.py | RunConfig | 환경·학습·평가 설정, 그리드, epsilon 일정 |
| run.py | run_grid(), main() | 조건·알고리즘·seed 순회, 저장, 모델 선정, CLI |
| test_smoke.py | LearningTests | 수식·재현성·연결·쉬운 점령 학습 검증 |

알고리즘을 이해하려면 `TabularControl`부터 읽으세요. Q-table은
`q[관측 요약 튜플] = 길이 9의 행동가치 배열` 구조입니다. `act()`는 마스크 안에서
탐험하거나 최대 Q 행동을 선택하며 동률은 무작위로 해소합니다. 처음 보는 key의
Q는 모두 0입니다. `update()`는 공통 식 `Q += alpha * (target - Q)`를 적용합니다.

`finish_episode()`는 경험열의 return을 뒤에서부터 계산하고, 같은 `(key, action)`이
반복되면 시간상 첫 방문만 갱신합니다. 표본평균 1/N 대신 고정 alpha를 사용하는
first-visit MC 제어입니다. 에피소드 수집 중 Q와 epsilon은 고정합니다.

`sarsa_step()`은 `r + gamma * Q(next_key, next_action)`을 타깃으로 갱신합니다.
`episode()`에서 다음 행동을 업데이트 전에 한 번 선택하고 실제 다음 틱에도 그대로
사용합니다. 마지막 틱은 다음 Q값 없이 마지막 보상만 사용합니다.

`episode()`는 모든 에이전트 행동을 모은 뒤 `TerritoryParallelEnv.step(actions)`를
한 번 호출합니다. 각 학습자의 경험열은 분리합니다. 디버그 이벤트는 지표에만 쓰며
정책 입력에는 넣지 않습니다. GUI 없이 실행합니다.

## 관측 표현과 한계

`compact_v1`은 원래 관측을 줄인 이산 표현으로 다음 정보를 담습니다.

- 본인 생존·공격 가능 여부, 부활 대기·남은 시간·좌표의 구간
- 상하좌우 소유권·경계 및 아군/적 점유 여부
- 시야에서 가장 가까운 적 영역·적 에이전트의 대략적인 방향과 거리
- 두 팀원의 위치 구간·생존·공격 가능·대기·직전 행동 및 정보 유효 마스크
- 현재 선택 가능한 행동을 나타내는 action mask

local의 추가 팀원 정보는 환경에서 가려지고 aware에서는 key에 실제 반영됩니다.
두 모드의 key 길이는 같습니다. 정보를 압축하므로 다른 상황이 같은 key가 될 수
있고, 부분 관측이므로 완전한 Markov state가 아닙니다. 평가의
`known_state_fraction`이 낮으면 대부분 처음 보는 상황에서 행동했다는 뜻입니다.
평가 중 새 key를 테이블에 넣거나 Q를 갱신하지 않습니다.

## 학습자와 난수 설정

기본은 **3대3 여섯 명 모두 학습**입니다. MC 실행에서는 여섯 명 모두 MC,
SARSA 실행에서는 모두 SARSA를 사용합니다. 독립 Q-table 6개로 파라미터를 공유하지
않습니다. 모델 role 0–2와 role 3–5가 각각 한 팀이며 에피소드마다 팀 묶음의
red/blue 진영을 교대합니다. 각자 받은 보상·경험으로 갱신하고 사망·부활도 연결합니다.
학습에서는 고정 정책 봇을 사용하지 않습니다. `train_opponent`는 이 모드에서
사용하지 않으며 고정 상대는 평가에만 등장합니다.

`--learners 1` 또는 `--learners 3`은 기존 한쪽 팀 학습 모드를 재현하는 옵션입니다.
그때만 나머지 아군은 expand, 상대는 train_opponent 정책을 사용합니다.
양 팀이 함께 바뀌므로 학습 중 상대는 고정되어 있지 않습니다.

학습 seed는 **7, 19, 31, 43, 59**입니다. SeedSequence로 환경·정책·봇의 난수를
분리합니다. validation/test는 서로 다른 고정 난수 영역이고 모든 후보가 동일한
평가 seed 집합을 사용합니다. 양 팀 6명 모드는 평가 master=20261031,
validation domain=40, test domain=50을 사용하며 정책 domain은 각각 +1입니다.
기존 한 팀 모드는 20/30을 유지합니다. 각 seed를 양쪽 진영에서 평가하며 full에서는
random·expand·aggressive·defend 상대에 같은 경기 수를 배정합니다.

기본 alpha=0.1, gamma=0.99, 초기 Q=0이며 epsilon은 1.0에서 0.1까지 선형 감소합니다.
이 설정은 GLIE/Robbins-Monro 수렴 실험이 아니며 최적 정책 수렴을 주장하지 않습니다.

## 보상·정보 공유 그리드

| 보상 | 기본 신호 | 종료 보너스 계수 |
| --- | --- | --- |
| individual | 본인의 점령 기여도 | 0, 1 |
| team / net | 팀 면적 순증가량 / 3 | 0, 1 |
| team / gross | 팀 총획득 타일 수 / 3 | 0, 1 |
| team / area | 현재 팀 면적 / 3 | 0, 1 |
| team / terminal | 중간 보상 0 | 1 |

종료 보너스는 승/무/패 +1/0/-1에 계수를 곱한 값입니다. 9개 보상과 local/aware
2개로 18조건입니다. individual에서 무시되는 team_signal 중복과 항상 보상 0인
terminal 설정은 제외합니다. alpha·gamma는 기본 그리드에 포함하지 않고 고정합니다.

## 사망과 종료

사망은 종료가 아닙니다. NOOP만 가능한 대기 전이와 대기 중 팀 보상도 보존하며,
부활 이후까지 MC return과 SARSA의 가치 연결을 유지합니다. 고정 지평 T는 실제
종료이므로 bootstrap은 0입니다. 외부 truncation은 완결 MC 에피소드로 오인하지
않도록 오류로 알립니다.

## 모델 선택과 평가

1. 각 조건·알고리즘에서 seed 5개로 각각 학습합니다.
2. 최종 모델을 저장하고 다시 읽어 validation 경기를 실행합니다.
3. 학습 전 Q=0 기준선도 같은 validation 경기에서 평가합니다.
4. validation 경기 점수(승 1, 무 0.5, 패 0)의 평균으로 최고 seed를 고릅니다.
5. 동점이면 최종 면적 비율, 그것도 같으면 설정에서 먼저 나온 seed를 고릅니다.
6. 선택 모델만 별도 test seed로 평가합니다. 양 팀 학습 모델은 각 팀 묶음(3명)을 고정 상대와 따로 평가합니다. 두 팀의 평가 점수를 같은 가중치로 합쳐 모델을 선정하고, 묶음별 점수도 별도로 저장합니다.
7. 알고리즘별 전체 조건의 대표 모델도 validation 지표만으로 고릅니다.

누적 보상은 보상함수마다 의미·크기가 달라 모델 선정에 사용하지 않습니다.
평가에서는 epsilon=0이며 최대 Q 행동의 동률만 seeded 무작위로 선택합니다.
seed 5개의 validation 점수 평균·표본 표준편차를 저장합니다. 최고 seed의 test
성적은 선택된 모델의 성능이며 알고리즘의 5-seed 평균 성능은 아닙니다.
연구 결론에는 모든 학습 seed의 최종 평가와 불확실성 분석이 추가로 필요합니다.
test를 보고 설정을 수정한 뒤 같은 test를 미사용 평가라고 해석하지 마세요.

MC와 SARSA의 환경 틱 예산은 같아도 실제 갱신 횟수는 다르며 모두 기록합니다.

## 저장 결과

```text
runs/<실행시각>/
├─ manifest.json                     # 설정·실행 환경·소스 해시·완료 상태
├─ results.json                      # 후보·선택 모델·알고리즘별 대표 모델
├─ selected.csv                      # 조건별 선택 seed와 평가 요약
└─ <관측_보상_종료보너스>/<알고리즘>/
   ├─ selected_test.csv              # 선택 모델의 test 경기별 결과
   └─ seed_<학습 seed>/
      ├─ model.json.gz               # Q-table과 설정
      ├─ training.csv               # 학습 에피소드별 결과
      ├─ validation.csv             # validation 경기별 결과
      ├─ untrained_validation.csv   # 학습 전 기준선 평가
      └─ summary.json               # 해당 후보 요약
```

CSV의 cohort=0은 모델 role 0–2, cohort=1은 role 3–5입니다.
학습 로그의 mean_return은 첫 팀 묶음 기준이며 양 팀 net 보상이 상쇄되는 것을
피합니다. red/blue_mean_return, red/blue_score, red/blue_area와 각 role의 return·updates·q_states를
별도로 기록합니다. 6명 모드의 deaths/dead_samples/learner_samples는 여섯 명 전체입니다.

`selected.csv`부터 확인하고 `results.json`의 groups에서 seed 간 편차와 기준선을
함께 보세요. CSV에는 상대·진영도 기록됩니다. `mean_return`은 첫 팀 묶음 학습자들의 비할인
에피소드 보상을 평균한 값, `area_fraction`은 최종 팀 면적 비율,
`dead_fraction`은 사망 대기 전이 비율, `learner_rejected`는 금지 행동 제출 수입니다.

manifest의 status는 정상 종료 시 complete, 중단 시 running입니다. 자동 재개는
지원하지 않으므로 새 폴더에 다시 실행하세요. 체크포인트는 gzip JSON입니다.

## Python에서 조건 변경 또는 저장 모델 재평가

프로젝트 루트에서 실행하거나 노트북의 경로 설정 셀 이후 사용하세요.

```python
import sys
sys.dont_write_bytecode = True
from dataclasses import replace
from smoke_test_mc_td.config import RunConfig
from smoke_test_mc_td.run import run_grid

cfg = replace(
    RunConfig(), episodes=500, learner_count=6,
    observation_modes=("local", "aware"),
    reward_specs=(("individual", "gross", 0.0), ("team", "gross", 0.0)),
)
output = run_grid(cfg)
```

저장된 모델만 다시 평가하려면 다음 함수를 사용하세요. 같은 test의 재실행은
동일 결과 재현이며 새로운 독립 평가를 추가하는 것은 아닙니다.

```python
from pathlib import Path
from territory_rl import ExperimentConfig
from smoke_test_mc_td.learning import load_models
from smoke_test_mc_td.runner import evaluate, summarize

checkpoint = Path("smoke_test_mc_td/runs/<실행시각>/<조건>/<알고리즘>/seed_7/model.json.gz")
models, metadata = load_models(checkpoint)
cfg = RunConfig(**metadata["config"])
experiment = ExperimentConfig(**metadata["experiment"])
rows = evaluate(cfg, experiment, models, split="test")
print(summarize(rows))
```

두 노트북은 같은 Python 구현을 부르는 실행 창구이며 알고리즘을 중복 구현하지
않습니다. rlTeam에 Jupyter 커널이 없으면 명령행으로 먼저 사용하세요. 커널 등록은
자동으로 하지 않습니다. 앞으로 이 모듈의 주석과 설명은 한국어로 작성합니다.

## 한 명과 여섯 명 학습 속도 비교

`benchmark_learning.py`는 같은 6×6 맵·80에피소드·32틱·seed 7에서 MC/SARSA와
local/aware를 각각 한 명/여섯 명 모드로 3회씩 독립 프로세스에서 측정합니다.
학습 함수 시간만 재며 평가·저장·import는 제외합니다. 원자료는
`six_learner_benchmark.json`에 저장합니다. 환경 전이는 한 번씩 그대로 실행하고
Q 업데이트와 관측 인코딩이 늘어나므로 속도가 반드시 6배 느려지지는 않습니다.
반면 양 팀을 고정 상대로 각각 평가하므로 기본 평가 경기 수는 기존의 두 배입니다.

기존 토의 보고서와 analyze_results.py는 이전 한 명 실험을 대상으로 합니다.
소스 해시가 바뀐 뒤 이전 보고서 재생성은 검사를 통과하지 않을 수 있으며,
여섯 명 실험 분석에는 cohort와 총 학습자 수를 반영해야 합니다.

실측 비교(동일 PC, 각 3회 중앙값, 후보당 80에피소드 × 32틱):

| 방법 | 관측 | 한 명 학습 초 | 여섯 명 학습 초 | 시간 배율 |
| --- | --- | --- | --- | --- |
| mc | local | 1.834 | 2.411 | 1.31 |
| mc | aware | 1.904 | 2.610 | 1.37 |
| sarsa | local | 1.814 | 2.402 | 1.32 |
| sarsa | aware | 1.951 | 2.713 | 1.39 |

프로세스 peak working set은 한 명 모드 약 41–43MiB, 여섯 명 모드 약 49–54MiB였습니다.
이 수치는 해당 작은 설정의 관측값이며 긴 학습·작은 맵·다른 CPU의 실행 시간을 보장하지 않습니다.
