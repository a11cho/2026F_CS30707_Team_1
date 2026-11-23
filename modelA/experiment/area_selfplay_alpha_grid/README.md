# 점유율 보상 · 자기대전 · 개인 보상 alpha 탐색

팀 보상은 매 틱 전이 후 아군 면적 / 전체 면적입니다. team_weight=1이며,
A-S는 capture_weight(alpha) × 개인 점령 참여를 더합니다. 종료 보너스는 0입니다.

기본 예산은 3,200경기(400업데이트 × 8경기), 맵 3×10, 최대 128틱입니다.
양 팀이 팀별 Actor·Critic을 공유하며 고정 진영으로 자기대전합니다.
실행마다 조건·alpha·시드·시각이 다른 runs/ 하위 폴더를 생성합니다.

```powershell
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-T --seed 19
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=0.1
python -B -m modelA train --experiment area_selfplay_alpha_grid --condition A-S --seed 19 --set reward.capture_weight=1
```

alpha 후보는 사용자가 정해 명령마다 지정합니다. 자동으로 후보 목록을 실행하지 않습니다.
alpha=0인 A-S는 보상상 A-T와 같습니다. A-T에서는 capture_weight를 사용하지 않습니다.
본 학습은 사용자 실행이며, 코드 검증은 작은 경기로만 수행했습니다.

평가 기본값은 반대 진영의 관측·행동을 학습 진영 기준으로 좌우반전합니다.
그래프와 리플레이 명령은 modelA/README.md를 참고하세요.
