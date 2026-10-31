# MC 제어 / SARSA(0) 스모크테스트

현재 기준은 **3대3 양 팀 6명 동시 학습, 개인/팀 종료 보상 × local/aware의 네 조건**입니다.
최신 완료 실행은 `runs/20261031T112320430780Z/`입니다.

- [최신 결과 보고서·그래프·영상](paper_report/REPORT.md)
- [확정 실행 설정과 진단 항목](AGREED_RUN.md)
- [저장 모델 관전 방법](VIEWER.md)
- [이전 자료 아카이브와 전체 정리 내역](archive/legacy_20261031/CLEANUP_LOG.md)

## 바로 사용하기

프로젝트 루트 `2026F_CS30707_Team_1`의 PowerShell에서 실행합니다.

```powershell
$rlPython = 'C:\Users\dohyu\anaconda3\envs\rlTeam\python.exe'

# 저장 모델 관전: 학습하지 않고 파일 선택 창을 엽니다.
& $rlPython -B -m smoke_test_mc_td.viewer

# 계산·양 팀 학습·API·재현성 검사
& $rlPython -B -m unittest smoke_test_mc_td.test_smoke smoke_test_mc_td.test_maintenance -v
& $rlPython -B -m smoke_test_mc_td.test_viewer

# 확정 설정으로 새 학습을 시작할 때만 실행합니다. 기존 학습의 재개 명령이 아닙니다.
& $rlPython -B -X utf8 -u -m smoke_test_mc_td.run --preset agreed
```

`-B`는 환경 코드에 Python 캐시 파일이 생기는 것을 막습니다. 학습·분석 결과는
이 폴더 안에만 저장하며 기존 모델을 자동으로 덮어쓰지 않습니다.
관전에는 NumPy·Gymnasium·PettingZoo·Tk가 필요하고, 그래프/영상 생성에는 별도의
matplotlib/Pillow가 있는 기본 Anaconda Python을 사용합니다. 자세한 재생성 명령은 보고서에 있습니다.

## 최신 설정

| 항목 | 값 |
| --- | --- |
| 맵·관측 | height=4, width=4, radius=1, compact_v1 |
| 에피소드·틱 | 후보마다 1,000에피소드 × 최대 64틱 |
| 학습자 | 양 팀 6명, 독립 Q-table 6개, 파라미터 공유 없음 |
| 부활 지연 | 2틱, 사망 대기도 경험열에 포함 |
| 알고리즘 | MC first-visit control / 1-step SARSA(0) |
| alpha·gamma·초기 Q | 0.1 / 0.99 / 0 |
| 탐험 | epsilon=0.8→0.05 선형 감소, mask 안에서만 선택 |
| 학습 seed | 7, 19, 31 |
| 보상 | 개인 점령 credit + 종료 계수 0 / 팀 terminal 승패 + 종료 계수 1 |
| 관측 비교 | local / aware |
| 전체 후보 | 네 조건 × 알고리즘 2개 × seed 3개 = 24회 |
| 중간 진단 | 100에피소드 간격, 고정 validation 일부 32경기 |
| 최종 validation | 후보마다 160경기, 학습 전 Q=0 기준선도 평가 |
| 선정 | validation 경기 점수→최종 면적 비율→seed 순서 |
| 최종 test | 선택 모델마다 480경기, 학습된 두 팀을 각각 고정 상대에 평가 |
| 평가 상대 | random / expand / aggressive / defend |

학습에는 고정 random 상대나 expand 아군이 없습니다. `train_opponent` 필드는
6명 모드에서는 사용하지 않으며 고정 정책은 평가에서만 사용합니다.
모델 role 0–2와 3–5가 팀을 이루고 에피소드마다 진영을 교대합니다.
평가는 epsilon=0이며 최대 Q 동률만 seeded 무작위로 선택합니다.

개인/팀 비교는 보상 귀속뿐 아니라 목적과 시점도 달라집니다. 수렴이나 정보 공유의
일반적인 효과를 단정하지 않습니다. 최고 seed의 test와 모든 seed의 평균 성능도 구분합니다.

## 코드 읽을 순서

| 파일·함수 | 역할 |
| --- | --- |
| `learning.py / TabularControl` | Q-table, 마스크 행동 선택, MC·SARSA 갱신 |
| `encoding.py / encode()` | 허용된 관측을 이산 key로 요약 |
| `runner.py / episode(), train(), evaluate()` | 여섯 행동 동시 제출, 학습·평가·진단 |
| `config.py / agreed_config()` | 이번에 확정한 최신 설정 |
| `run.py / run_grid()` | 조건·seed 순회, 모델 선정, 결과 저장 |
| `runtime_metrics.py / memory()` | 현재 프로세스의 Windows 메모리 측정 |
| `viewer.py / ModelSimulation, SimulationViewer` | 모델을 바꾸지 않는 추론·관전 |
| `test_smoke.py, test_viewer.py, test_maintenance.py` | 학습·관전·정리 후 의존성 검사 |

학습 식은 `Q += alpha * (target - Q)`입니다. MC는 완결 에피소드의 return으로
첫 방문만 갱신하고, SARSA는 실제 다음 행동의 Q로 bootstrap합니다.
사망으로 궤적을 끊지 않고, 실제 종료에서는 bootstrap하지 않습니다.
local/aware 입력은 부분 관측이며 compact_v1도 정보를 줄이므로 Markov state가 아닙니다.

## Python과 노트북 실행

```python
import sys
sys.dont_write_bytecode = True
from dataclasses import replace
from smoke_test_mc_td.config import agreed_config
from smoke_test_mc_td.run import run_grid

# 실행하면 새 학습이 시작됩니다.
cfg = replace(agreed_config(), algorithms=("sarsa",))
# output = run_grid(cfg)
```

두 노트북은 `agreed_config()`를 기준으로 한 짧은 연결 검사를 기본으로 합니다.
`FULL_TRAINING=True`로 바꾸면 해당 알고리즘의 최신 전체 학습(12후보)을 시작합니다.
노트북에는 과거 실행 출력이 남지 않도록 초기화했습니다.
`smoke_test_td.ipynb`의 알고리즘은 상태가치 TD(0)가 아니라 SARSA(0)입니다.

`RunConfig()`와 `--preset full`은 과거 범용 설정(6×6·80에피소드·18조건·5seed)을
호환용으로 유지합니다. **이번 설정을 재현할 때는 agreed_config()/--preset agreed를 사용하세요.**
`--preset quick`도 별도의 과거 연결 검사입니다. 기존 checkpoint 로딩을 위해
RunConfig 기본값을 일괄 변경하지 않았습니다.

## 저장 결과와 재현

최신 원본 모델·로그·manifest·source_snapshot은 `runs/20261031T112320430780Z/`,
최신 보고서·그림·영상·분석 코드는 `paper_report/`에 있습니다.
모델·로그와 실행 당시 source_snapshot은 정리 과정에서 수정하지 않았습니다.

보고서 재수집은 **source_snapshot의 기록 해시**를 기준으로 검증합니다. 현재 코드가
정리되거나 바뀌어도 보존본 검증은 유지하고, 별도 CLI 프로세스에서 보존 소스로 추론합니다.
현재 경로의 차이는 경고 정보로 기록합니다.
이전 실행의 위치는 `archive/*/runs/<실행 ID>/`에서 찾습니다.

학습은 자동 재개를 지원하지 않습니다. 종료된 manifest status는 complete이며,
중단된 실행은 running으로 남습니다. test를 보고 변경한 다음 실험에는 새 최종
평가 bank를 사용해야 합니다.
