# 16차원 기준 결과의 출처

아래 두 조건은 기존 실험에서 학습·평가를 완료한 결과를 **실제 파일로 복사**했습니다. 표의 링크는 exp3 안에 있는 복사본을 엽니다. 복사본마다 `ORIGIN.json`에 원본 실험, run별 인원·시드, 업데이트, UTC 복사 시각, 원본 파일별 SHA-256을 기록합니다. 새 분석에서 run 번호는 출처와 함께 읽습니다.

| 조건 | 원본 | 복사본의 1대1 run (시드 7·19·31 순) | 복사본의 3대3 run (동일 시드 순) | 사후 평가·리플레이 |
|---|---|---|---|---|
| 0칸 · 16차원 | [exp2](../exp2/README.md) → [복사 기록](copied/exp2/ORIGIN.json) | run1, run2, run3 | run6, run5, run4 | [보고서](copied/exp2/result/analysis_u0400/ae/eval_20261120_40_random-expand-aggressive-defend-noop/REPORT.md) · [GIF 목록](copied/exp2/result/analysis_u0400/ae/eval_20261120_40_random-expand-aggressive-defend-noop/replays/index.csv) |
| 9칸 · 16차원 | [exp1](../exp1/config.toml) → [복사 기록](copied/exp1/ORIGIN.json) | run1, run2, run3 | run4, run5, run6 | [보고서](copied/exp1/result/analysis_u0400/eval_20261120_40_random-expand-aggressive-defend-noop/REPORT.md) · [GIF 목록](copied/exp1/result/analysis_u0400/eval_20261120_40_random-expand-aggressive-defend-noop/replays/index.csv) |

각 복사본에는 인원별 인코더·학습곡선, PPO 설정·학습 CSV·최종 체크포인트, 업데이트 400의 고정 상대 평가 6세트와 맞대결 3세트, 기존 보고서·그림·대표 GIF가 들어 있습니다. exp2에서는 무마스킹 AE 6개 run의 결과를 골랐습니다. exp1의 `mask_fraction=0.30`은 30칸 중 정확히 9칸입니다.

exp2의 `best.pt`와 `validation.csv`는 학습 인원과 같은 환경에서 온라인 검증해 저장한 값입니다. exp1은 5업데이트마다 번호 모델을 기록했고, 이번에 재사용한 원본에는 같은 방식의 온라인 검증 이력이 없습니다. exp1 복사본은 `latest.pt`로 400번째 모델을 복원하며, 중간 번호 모델은 원본 exp1에 있습니다. 여섯 조건의 공통 모델 선택 기준은 400번째 업데이트입니다.
