# SL-RL 팀 점령 실험

1vs1 자기대전으로 배운 정책 세 개를 한 팀에 배치했을 때, 3vs3에서 직접 배운 팀에 얼마나 근접하는지 측정합니다. exp1에서는 인원별 마스킹 인코더를 사용합니다. exp2에서는 마스킹 없는 인원별 AE와 점령 지도 원본 입력을 비교합니다. 3vs3 평가에 들어갈 때 가중치는 모두 고정합니다.

- [구현 설명과 결정 기록](IMPLEMENTATION.md)
- [실험 순서와 판독 기준](EXPERIMENT_PLAN.md)
- [exp1 설정](experiment/exp1/config.toml)
- [exp2 세 터미널 일괄 실행·재개·결과 확인](experiment/exp2/README.md)
- [exp3 AE 0·9칸 마스킹 × 8·16·32차원 그리드 실행](experiment/exp3/README.md)

`rlTeam`을 활성화한 저장소 루트에서 `python -B SL-RL/experiment/exp2/launch.py`를 실행하면 exp2를 시작합니다. 아래 개별 명령은 exp1 기준입니다.

## 실행 준비

저장소 루트에서 `rlTeam` 환경을 활성화한 상태로 실행합니다. 아래 명령은 VS Code의 CMD와 PowerShell에서 동일하게 작동합니다. Python 3.11 이상, NumPy, CPU PyTorch와 원본 `territory_rl`을 사용합니다.

```text
python -B SL-RL/verify.py
```

위 명령은 별도의 작은 임시 실험에서 규칙 검사, 양쪽 인코더 학습, PPO, 재개, 고정 상대 평가, 맞대결을 실행합니다. 성공하면 임시 실험을 회수합니다.

## 1. 인원별 인코더 학습

```text
python -B SL-RL/run.py pretrain --experiment exp1 --size 1
python -B SL-RL/run.py pretrain --experiment exp1 --size 3
```

각각 `experiment/exp1/pretrain/1v1/fit1`, `3v3/fit1`에 설정·학습곡선·검증 손실이 가장 작았던 `encoder.pt`를 저장합니다. 다시 실행하면 `fit2`가 생깁니다. 각 인원 조건 내에서는 같은 인코더를 세 PPO 시드에서 재사용합니다.

## 2. 학습

아래 순서로 새 실험에서 실행하면 `run1`부터 `run6`까지 생깁니다. 각 명령은 400회 업데이트 × 8경기로 3,200경기를 학습합니다. 실행 번호는 폴더 할당 순서이며 병렬 실행 시에는 콘솔 경로와 `settings.json`을 확인하세요.

```text
python -B SL-RL/run.py train --size 1 --seed 7 --fit fit1
python -B SL-RL/run.py train --size 3 --seed 7 --fit fit1
python -B SL-RL/run.py train --size 1 --seed 19 --fit fit1
python -B SL-RL/run.py train --size 3 --seed 19 --fit fit1
python -B SL-RL/run.py train --size 1 --seed 31 --fit fit1
python -B SL-RL/run.py train --size 3 --seed 31 --fit fit1
```

기본 실험은 `exp1`입니다. `--fit`을 생략하면 인원·게임·SSL 설정이 일치하는 가장 최근 번호의 fit을 사용합니다. run 안의 체크포인트에는 인코더 자체도 들어갑니다. 길게 돌리기 전 `--updates 1`을 붙이면 8경기만 실행합니다. 이는 별도의 새 run이므로 이후에는 실제 출력된 run 번호를 쓰세요.

### 중단 후 재개

```text
python -B SL-RL/run.py resume --run run1
python -B SL-RL/run.py resume --run run1 --updates 500
```

재개는 해당 run의 `latest.pt`에 저장된 두 팀·옵티마이저·난수 상태와 설정을 사용합니다. `--updates`는 최종 목표입니다. 400에서 500으로 연장하면 800경기가 추가됩니다. 기본 5업데이트 및 실행 마지막에 저장하므로 갑작스러운 중단 시 마지막 저장 이후의 경기를 다시 수행합니다. 첫 체크포인트 이전에 중단되면 새 run으로 시작하세요. 같은 run의 재개는 한 터미널에서 실행하세요. SSL의 `fit` 재실행은 새로운 사전학습입니다.

### 최종 및 최고 체크포인트만 보관하기

`train` 또는 `resume`에 `--keep-last-and-best`를 붙이면 평소처럼 학습 중간에 저장하고, 목표 업데이트까지 정상 완료한 후 저장된 시점들을 3vs3으로 검증합니다. 옵션은 실행마다 선택합니다. 생략하면 기존 방식으로 모든 번호 파일을 보관합니다.

```text
python -B SL-RL/run.py train --size 1 --seed 7 --fit fit1 --updates 400 --keep-last-and-best
python -B SL-RL/run.py resume --run run1 --updates 500 --keep-last-and-best
```

이미 학습한 exp1은 다음 명령으로 재학습 없이 정리할 수 있습니다. 각 run의 `latest.pt`가 지정한 업데이트인지 확인한 후 진행합니다.

```text
python -B SL-RL/run.py prune-checkpoints --experiment exp1 --run run1 run2 run3 run4 run5 run6 --at-update 400
python -B SL-RL/run.py evaluate --run run1 --checkpoint best.pt
```

두 학습 팀 모두 random·expand·aggressive·defend를 상대합니다. 최종 평가와 분리한 고정 시드로 양 진영을 교대하며, 전체 경기에서 **(승 + 0.5 × 무) / 경기 수**가 가장 높은 시점을 고릅니다. 동점이면 더 늦은 업데이트를 선택합니다. noop은 동작 확인용이라 선정에서 제외합니다. 기본값은 학습 팀·상대별 4경기, 즉 체크포인트당 32경기입니다. 80개를 비교하면 run당 2,560경기입니다. `--validation-games 2`로 줄이거나 더 큰 짝수로 늘릴 수 있습니다. 소수 경기의 선정 점수에는 표본 변동이 있으므로 최종 보고에는 별도 `evaluate` 결과를 사용하세요.

검증 결과는 `selection_업데이트_번호.json`에 후보별 승·무·패와 함께 기록합니다. 최고 체크포인트와 기록을 저장한 뒤 해당 run 바로 아래의 번호 체크포인트를 삭제합니다. 남는 `latest.pt`는 이어학습·기본 평가용이고 `best.pt`는 최고 시점 평가용입니다. 같은 시점이 두 역할을 맡을 수도 있습니다. 삭제된 중간 시점은 다시 평가할 수 없습니다. 학습 CSV, 설정, 기존 평가 결과와 인코더는 보존합니다. 이후 확장 학습에서 다시 정리하면 기존 최고와 현재 남아 있는 후보를 비교합니다.

## 3. 기존 봇 전체 평가

```text
python -B SL-RL/run.py evaluate --run run1 --replay
python -B SL-RL/run.py evaluate --run run2 --replay
python -B SL-RL/run.py evaluate --run run3
python -B SL-RL/run.py evaluate --run run4
python -B SL-RL/run.py evaluate --run run5
python -B SL-RL/run.py evaluate --run run6
```

모두 **3vs3**로 평가합니다. 1vs1 run도 그대로 지정합니다. 한 run의 빨강 학습 정책과 파랑 학습 정책을 각각 다섯 상대와 40경기씩 평가하므로 기본 합계는 400경기입니다. 각 40경기의 앞 20경기는 해당 정책이 빨강, 뒤 20경기는 파랑입니다. 같은 평가 시드를 양 진영에서 쌍으로 사용합니다. `--replay`는 상대·학습 정책·실제 진영별 첫 경기의 틱별 상태와 행동을 JSON으로 저장합니다.

```text
python -B SL-RL/run.py evaluate --run run1 --opponents expand aggressive --games 10 --checkpoint update_0100.pt
```

## 4. 복제 팀과 직접 학습 팀의 맞대결

```text
python -B SL-RL/run.py duel --run run1 --versus run2 --replay
python -B SL-RL/run.py duel --run run3 --versus run4
python -B SL-RL/run.py duel --run run5 --versus run6
```

두 run의 빨강·파랑 학습 정책을 모두 교차하면 4개 조합입니다. 기본 40경기씩 총 160경기를 치릅니다. 승률·점령 격차·평균 보상은 첫 번째 `--run` 관점입니다.

## 결과 위치와 읽는 순서

| 경로 (exp1 기준) | 확인할 내용 |
|---|---|
| `pretrain/1v1/fit1/metrics.csv`, `pretrain/3v3/fit1/metrics.csv` | 검증 손실, 가려진 전체 칸과 전선 칸의 복원 정확도, 초기 지도 및 이웃 다수결 성적 |
| `run1/settings.json`, `training.csv` | 설정·시드·인코더 출처, 팀별 보상·승률·최종 격차·PPO 손실·엔트로피·KL·시간 |
| `run1/update_0005.pt`, `run1/latest.pt` | 특정 학습 시점 및 재개용 최신 체크포인트 |
| `run1/evaluation/eval1/summary.csv` | 각 학습 팀의 상대별 승·무·패와 승률 |
| `run1/evaluation/eval1/games.csv` | 개별 경기 시드, 실제 진영, 에이전트당 보상, 점령 격차 |
| `comparison/eval1/` | 맞대결 설정과 위 두 종류의 CSV. 재평가는 다음 eval 번호에 저장 |

다음 비교 실험은 예를 들어 `experiment/exp4` 폴더에 해당 실험의 `config.toml`을 준비하고 각 명령에 `--experiment exp4`를 붙이세요. exp1의 run 번호와 독립적으로 시작합니다. 실험 결과와 모델 가중치는 기본 Git 추적에서 제외합니다.

## 실험 분석을 다시 실행하는 명령

저장소 루트의 rlTeam 터미널에서 실행합니다. 기본값은 exp1 설정의 목표 업데이트와 경기 수입니다. 각 단계는 실험 폴더에서 run의 팀 인원·시드를 읽습니다. 같은 시드의 1인 학습 run과 3인 학습 run이 맞대결 대상입니다.

```text
python -B SL-RL/analysis.py plots --experiment exp1
python -B SL-RL/analysis.py evaluate --experiment exp1
python -B SL-RL/analysis.py report --experiment exp1
python -B SL-RL/analysis.py replays --experiment exp1
```

| 단계 | 입력과 결과 |
|---|---|
| `plots` | 학습 CSV와 각 run이 실제 사용한 인코더의 기록에서 팀·시드별 Actor/Critic loss, entropy, KL, SSL 복원 그래프, 처음·마지막 25업데이트 요약표를 만듭니다. |
| `evaluate` | 고정 상대 전체와 같은 학습 시드의 1vs1 복제 팀 대 3vs3 직접 학습 팀을 평가합니다. 같은 업데이트·상대·시드·경기 수·진영의 행이 모두 있는 기존 결과를 재사용합니다. 누락된 평가만 실행하며, 전부 다시 하려면 `--refresh`를 붙입니다. |
| `report` | 학습 그래프, 고정 상대별 승률 그림, 맞대결 승·무·패, 학습 팀·실제 진영별 CSV, 상대 경로로 원자료를 연결한 한국어 REPORT.md를 생성합니다. 저장된 평가만 읽습니다. |
| `replays` | 각 학습 인원 × 고정 상대마다 한 경기와 맞대결 한 경기만 골라 애니메이션 GIF로 저장합니다. 기존 경기 시드로 선택된 경기만 재실행하고 원래 지표와 일치하는지 검사합니다. 화면에는 실제 진영, 학습 정책, 점령 면적, 틱, 각 에이전트 행동이 표시됩니다. |

예를 들어 `analysis.py replays --experiment exp1 --runs run2 run5 --fps 4`는 시드 19의 두 run만 사용합니다. 임의 시점은 모든 단계에 `--update 200`처럼 같은 값을 전달하세요. 평가 경기 수를 바꾸면 `evaluate`, `report`, `replays`에 같은 `--games`를 사용하세요. 평가와 리플레이 재현에는 해당 체크포인트가 필요합니다. 정리한 run에서도 최종·최고로 보존한 업데이트 번호를 지정할 수 있습니다.

기본 학습 이미지 위치는 `experiment/exp1/result/analysis_u0400/`입니다. 평가 보고서는 그 아래 `eval_20261120_40_random-expand-aggressive-defend-noop/`, GIF와 선정 근거 `index.csv`는 보고서 폴더의 `replays/`에 생성됩니다. `--runs`를 쓰면 선택한 run 이름의 하위 폴더로 구분합니다. 같은 조건으로 그림이나 GIF를 다시 만들면 해당 파생 파일을 갱신합니다. 기존 수동 분석 폴더와 원본 학습·평가 기록은 유지합니다.

리플레이의 ‘좋은 경기’는 **승리 → 무승부 → 패배** 순이며, 같은 결과에서는 최종 점령 격차, 에이전트당 누적 보상 순으로 고릅니다. 모두 같으면 먼저 기록된 후보를 택합니다. 고정 상대에서는 각 학습 조건의 관점, 맞대결에서는 1vs1 복제 팀 관점입니다. 기본 exp1의 GIF는 11개입니다. 이 영상은 가장 잘한 사례이고, 일반적인 실력은 전체 경기 승률로 판단합니다.

명령·평가 탐색은 `analysis.py`, 그래프·표·보고서는 `analysis_plots.py`, 대표 경기 선정·재현·영상 저장은 `analysis_replays.py`에서 관리합니다.
