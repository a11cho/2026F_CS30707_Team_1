# Model A · 점유율 보상과 평가 방향 정렬

팀마다 Actor·Critic 한 쌍을 세 명이 공유합니다. 두 팀의 모델과 optimizer는 분리하고,
빨강 정책은 학습 내내 빨강, 파랑 정책은 파랑으로 자기대전합니다.

## 이번 보상

- A-T는 매 틱 team_weight × 아군 면적 / 전체 면적을 각 팀원에게 지급합니다.
- A-S는 같은 팀 보상에 alpha × 본인의 실제 점령 참여를 더합니다.
- alpha 설정 이름은 reward.capture_weight이며 기본값은 1입니다.
- 공동 점령 참여자는 각각 alpha를 받습니다. 종료 승패 보너스와 사망 감점은 0입니다.
- 종료 틱에도 최종 전이 후 점유율 보상은 지급합니다.

area_share는 지속적으로 영역을 보유하는 것을 목표로 합니다. 조기 종료 시 받을 틱 수가
달라지며, 최종 승패와 누적 점유율은 서로 다른 평가 지표입니다.

## 새 실험의 학습

저장소 루트에서 rlTeam을 활성화합니다. Python 3.11 이상과 원본 환경 의존성이 필요합니다.
추가 의존성은 modelA/requirements.txt에 기록되어 있습니다.

```powershell
conda activate rlTeam
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-T --seed 19
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=0.1
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=1
```

기본 예산은 3,200경기(400업데이트 × 8경기), 맵 3×10, 최대 128틱입니다.
본 학습은 사용자가 직접 실행합니다. 짧게 확인하려면 --set train.updates=1과
--set game.max_ticks=4를 추가할 수 있습니다.

    experiment/area_selfplay_alpha_grid/
      configs/a_t.toml
      configs/a_s.toml
      configs/evaluation.toml
      runs/A-T_seed19_<실행ID>/
      runs/A-S_alpha0.1_seed19_<실행ID>/
      runs/A-S_alpha1_seed31_<실행ID>/
      result/

조건·alpha·시드마다 별도 실행 폴더가 생성됩니다. alpha 후보 목록은 자동 실행하지 않으며
원하는 값을 명령마다 지정하세요. A-S에서 alpha=0이면 보상은 A-T와 같습니다.
A-T의 capture_weight는 사용하지 않습니다.

## 설정과 이어 학습

실험의 configs/*.toml을 편집하거나 --set 구역.항목=값을 반복해 덮어씁니다.
실효 설정은 실행 폴더의 config.json에 저장됩니다.

| 구역 | 주요 설정 |
| --- | --- |
| game | height, width, max_ticks, respawn_delay, initial_placement |
| reward | mode(A-T/A-S), signal(area_share/gap_delta), team_weight, capture_weight(alpha) |
| model | hidden_size, previous_actions |
| train | seed, updates, episodes_per_update, checkpoint_every |
| ppo | learning_rate, gamma, gae_lambda, clip_ratio, entropy_coef, value_coef, max_grad_norm, epochs, minibatch_size |
| evaluation | seed, opponents(상대→짝수 경기 수), deterministic, mirror_to_training_side |
| runtime | device(cpu/cuda), threads |

체크포인트는 양 팀 모델·optimizer·난수 상태·진행도·지표를 원자적으로 저장합니다.
기본 5업데이트마다 및 마지막에 저장하고, 중단 시 마지막 저장부터 이어갑니다.

```powershell
python -B -m modelA train --resume <실행폴더>
python -B -m modelA train --resume <실행폴더> --set train.updates=800
```

updates는 누적 목표입니다. --output <새폴더>를 주면 체크포인트에서 별도 실행으로
분기합니다. 이어 학습에서는 보상과 alpha를 바꿀 수 없습니다. 새 alpha 실험은
새 모델로 시작하세요. 장치 변경은 이후 난수 재현 결과에 영향을 줄 수 있습니다.

## 평가 시 좌우반전

```powershell
python -B -m modelA evaluate --checkpoint <실행폴더> --opponent expand=80 --opponent random=80 --opponent defend=80
```

경기 수는 정책 하나당 양 진영 합계입니다. expand=80이면 각 정책을 파랑 40경기 →
빨강 40경기로 평가하므로 두 정책을 합쳐 160경기입니다. 같은 pair는 같은 환경 seed를 씁니다.
각 학습팀·상대·진영별 결과를 evaluations/eval_<시각>/에 새로 저장합니다.

기본 mirror_to_training_side=true입니다. 정책의 원래 학습 진영과 현재 진영이 다르면
맵·모든 좌표·행동 마스크·직전 행동·행동 관련 관측을 좌우반전합니다. 선택 행동은
다시 실제 좌표계로 변환해 환경에 제출합니다. 학습 진영에서 평가할 때는 변환하지 않습니다.
이 정렬은 반대 진영에서 학습 당시와 같은 방향 관계를 제공하려는 처리입니다.
기존의 반전 없는 일반화 평가와 구별해서 비교하세요.

```powershell
python -B -m modelA evaluate --checkpoint <실행폴더> --set evaluation.mirror_to_training_side=false
```

eval_games.csv의 mirrored, eval_summary.json의 settings에 반전 설정을 기록합니다.
기본 행동은 마스크 후 argmax이고 deterministic=false로 샘플링할 수 있습니다.

## 그래프와 영상

```powershell
python -B -m modelA.plot_training <실행폴더>
python -B -m modelA.plot_training <실행폴더> --show
python -B -m modelA.plot_evaluation <평가폴더1> <평가폴더2> --output modelA/experiment/area_selfplay_alpha_grid/result/comparison
python -B -m modelA.render_replay <평가폴더>
```

학습 그래프는 실행의 training_curves.png에 저장합니다. 평가 비교는 조건별 요약 표(evaluation_summary.png), 상대별 상세 표(evaluation_상대.png),
브라우저용 comparison.html을 생성합니다. 맨 오른쪽 실제 승률을 0% 빨강~100% 초록으로 표시하며,
무승부를 절반 반영하는 경기 점수와 구분합니다. comparison.csv에는 실행별 원본 평가 경로를,
summary.csv에는 같은 조건의 시드·양 팀 합산 결과를 기록합니다.

평가 JSONL은 정책 × 상대 × 진영마다 첫 경기 하나만 저장합니다. 영상은 별도 명령에서
정책·상대별 파랑 경기 → 빨강 경기를 이어 붙입니다. 화면의 색은 실제 진영이며,
원래 학습 정책·현재 진영·방향 정렬 여부를 표시합니다. 결과는 실험 result/ 아래에 저장합니다.
원본 환경과 학습 보상을 재검증하며, 실제로 실행된 세계 좌표계 행동을 재생합니다.

## 이전 실험 코드의 보관 위치

이번 변경 직전의 코드와 실험 설정은 archive/pre_area_share_20261118/modelA/에
보존했습니다. 보관 범위·SHA256은 해당 아카이브의 archive_manifest.json을 보세요.
이전 실험의 학습·평가·그림·영상은 experiment/smoke_test_001/ 및
experiment/selfplay_3200ep_3seeds/의 기존 위치에 있습니다. 각 실행의 README.md에
당시 코드를 다시 참조하는 경로를 적었습니다.

기존의 reward.signal이 없는 체크포인트와 설정은 gap_delta로 읽어 원래 변화량
보상을 유지합니다. 예전 JSONL도 원래 보상으로 검증합니다. 기존 체크포인트를 새 코드로
재평가하면 기본 방향 정렬이 적용됩니다. 옛 평가를 재현하려면 false를 명시하거나
아카이브 실행기를 사용하세요. 보유 면적 실험을 하려면 새 area_share 설정으로 학습합니다.

아카이브의 코드를 사용하는 새 프로세스 실행 예시는 다음과 같습니다.

```powershell
python -B modelA/archive/pre_area_share_20261118/run_archived.py --help
```

가장 오래된 고정 상대 실험의 별도 아카이브는 archive/legacy_fixed_opponent_20261118/입니다.

## 개발 검증

```powershell
python -B -m unittest modelA.test_workflow -v
```

점유율·alpha·기존 변화량 보상, 양 팀 갱신·재개, 관측·행동·직전 행동의 반전,
실제 평가 행동 대응, 원본 리플레이 호환성과 PNG·GIF 생성을 작은 경기로 검사합니다.
