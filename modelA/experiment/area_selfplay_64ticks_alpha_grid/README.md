# 64틱 자기대전·개인 점령 계수 비교

최대 틱 수를 128에서 64로 줄이고 A-S의 α=0.5·1·2를 비교합니다.
이전 실행 A-S_alpha1_seed7_20261117T220020783922Z의 실제 config.json을 기준으로 설정을 복사했습니다.
모델 구조·PPO·보상식·학습 경기 수는 유지하며 기존 실험과 결과는 수정하지 않습니다.
설정 준비와 동작 확인만 수행했으며 본 학습은 실행하지 않았습니다.

| 항목 | 값 |
|---|---|
| 맵 | 높이 3 × 너비 10 |
| 최대 틱 | 64 |
| 보상 | 매 틱 아군 점유율 + α × 개인 점령 참여 |
| 개인 계수 α | 0.5, 1, 2 |
| 종료 보너스·사망 감점 | 없음 |
| 학습 시드 | 7, 19, 31 |
| 실행 수 | 9 |
| 실행당 경기 수 | 3200 (400업데이트 × 8경기) |
| 학습 방식 | 양 팀 동시 자기대전, 팀별 Actor·Critic 공유, 학습 진영 고정 |
| 부활 지연 | 0 |
| 배치·학습률·할인율 등 | configs/a_s.toml에 이전 실험의 실제 설정 보존 |
| 평가 | expand·random·defend 각각 정책당 40경기, 양 진영 각 20경기 |
| 평가 난수·행동 | 20261120, 결정적 행동, 반대 진영 좌우반전 |

## 학습

저장소 루트의 rlTeam 환경에서 아래 명령을 실행합니다. 반복문 없이 순서대로 실행합니다.
세 터미널을 쓰면 α별 세 명령을 각각 한 터미널에 배정할 수 있습니다.

```powershell
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 7 --set reward.capture_weight=0.5
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=0.5
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 31 --set reward.capture_weight=0.5
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 7 --set reward.capture_weight=1
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=1
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 31 --set reward.capture_weight=1
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 7 --set reward.capture_weight=2
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=2
python -B -m modelA train --experiment area_selfplay_64ticks_alpha_grid --condition A-S --seed 31 --set reward.capture_weight=2
```

실행 결과는 이 실험의 runs/A-S_alpha값_seed번호_실행ID/에 자동 저장됩니다.
설정에서 평가 기본값도 40경기로 지정했습니다. 새 실험 이름과 --condition A-S를 반드시 지정하세요.

## 평가와 결과 출력

```powershell
python -B -m modelA.plot_training <실행폴더>
python -B -m modelA evaluate --checkpoint <실행폴더> --config modelA/experiment/area_selfplay_64ticks_alpha_grid/configs/evaluation.toml
python -B -m modelA.plot_evaluation <평가폴더1> <평가폴더2> --output modelA/experiment/area_selfplay_64ticks_alpha_grid/result/evaluation_comparison
python -B -m modelA.render_replay <평가폴더>
```

평가 또한 학습 체크포인트의 64틱 조건을 사용합니다.
리플레이 기본 출력은 이 실험의 result/ 아래에 저장됩니다.

## 해석 시 주의

- 이번 최대 길이는 64틱입니다. 리플레이에서 중반 이후 무너지는 양상이 관찰되어, 짧은 경기 조건을 비교하려는 실험입니다.
- 3200경기의 최대 환경 전이는 409600틱에서 204800틱으로 감소합니다(기존 대비 50%). 업데이트당 최대 전이도 1024에서 512틱으로 줄어듭니다.
- 초반 전선 형성과 점령 행동에 집중하는 짧은 경기 실험으로 해석합니다. 이동 사망과 재점령 문제는 틱 수 변경만으로 해결된다고 가정하지 않습니다.
- 경기 종료 시점이 달라지므로 기존 128틱 승률과는 별도 과제의 성적으로 비교해야 합니다. 이번 64틱 조건 안에서 α를 우선 비교합니다.
- 누적 보상은 경기 길이의 영향을 받습니다. 승률·평균 점유율·최종 점유율·틱당 사망률을 함께 확인합니다.
- 대표 리플레이 분석용 첫 경기 기록은 기존 평가 코드와 동일하게 저장합니다.
