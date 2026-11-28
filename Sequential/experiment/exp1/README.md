# Sequential exp1

저장소 루트의 rlTeam 터미널에서 실행합니다.

```bat
python -B Sequential/experiment/exp1/run.py launch
python -B Sequential/experiment/exp1/run.py plot
python -B Sequential/experiment/exp1/run.py evaluate
```

첫 명령은 Windows 콘솔 세 개에서 시드 7·19·31의 3대3 자기대전을 시작합니다. 각 시드는 3×10 맵, 최대 64틱, 업데이트당 8경기씩 400회 학습합니다. 양 팀에 별도 정책을 두고, 팀 안에서는 세 개체가 정책을 공유합니다. 보상은 매 틱 자기 팀 점유율과 개인 점령 기여의 합입니다. 세부 설정과 exp2 비교 대상 경로는 config.toml에 있습니다.

- `runs/seed7`, `runs/seed19`, `runs/seed31`에 설정, 팀별 학습 CSV, 검증 CSV, latest.pt와 best.pt를 저장합니다. 20 업데이트마다 네 봇(random, expand, aggressive, defend)을 상대로 학습 팀별·봇별 6경기(양 진영 합계)를 치릅니다. 검증 점수는 (승 + 0.5 × 무) / 경기 수이고 동점이면 최근 모델을 택합니다.
- 중단 후 launch를 다시 실행하면 마지막 저장 시점부터 재개합니다. 완료한 시드는 건너뜁니다. 한 시드만 재개하려면 `python -B Sequential/experiment/exp1/run.py train --seed 19`를 사용하세요. 설정을 변경한 실험은 별도 실험 폴더에 저장하세요.
- `result/training`의 training.png에는 시드·학습 팀별 PPO 손실, 엔트로피, KL, 평균 에이전트 보상과 최종 영토 격차가 있습니다. validation.png는 고정 상대 검증 추이입니다.
- evaluate는 기본적으로 양쪽 모두 latest.pt를 사용합니다. 양쪽의 검증 최고 모델을 비교하려면 `evaluate --checkpoint best.pt`로 실행합니다. 평가 때는 탐욕 행동을 사용합니다.
- 각 학습 팀을 봇 5종 및 같은 시드 exp2 raw 1대1 복제·3대3 직접 학습 정책과 평가합니다. 독립 정책마다 red·blue 학습본을 각각 상대합니다. 정책 조합당 40경기, 앞 20경기는 빨강, 뒤 20경기는 파랑으로 뛰며 시작 시드를 짝짓습니다. 총 3 × 2 × (5 + 2 × 2) × 40 = 2,160경기입니다.
- 평가할 때마다 `result/evaluation1`, `evaluation2`, …에 설정, 개별 경기(games.csv), 시드·학습 팀·상대별 성적(summary.csv)을 저장합니다. model과 opponent 각각에 대해 공동공격 목표/전체 공격 목표, 공동점령/전체 점령, 사망·이동 실패를 SL-RL과 같은 함수로 집계합니다. 비율은 분자를 합하고 분모를 합해 계산하며 분모 0은 빈칸입니다. 동일한 경기에서 두 팀을 측정하므로 경기 도중 실제로 만난 협력 기회의 수는 서로 다를 수 있습니다. 별도의 8×8 고정 전술 장면 검사는 이 3×10 평가에 포함하지 않습니다.
