# 확정된 양 팀 동시 학습 실행안

이 실행은 `--preset agreed`로 시작합니다. 기존 quick/full 설정과 이전 결과는 보존합니다.

```powershell
$rlPython = 'C:\Users\dohyu\anaconda3\envs\rlTeam\python.exe'
& $rlPython -B -X utf8 -u -m smoke_test_mc_td.run --preset agreed
```

## 고정 설정

- height=4, width=4, 최대 64틱, 후보별 1,000에피소드, 관측 반경 1
- 3대3 양 팀 여섯 명 전원 학습, 독립 Q-table 6개, 모델 묶음별 진영 교대
- MC first-visit control / SARSA(0), alpha=0.1, gamma=0.99, 초기 Q=0
- epsilon=0.8에서 0.05로 선형 감소, 부활 지연 2틱
- seed=7, 19, 31; 각 seed에서 다시 초기화해 독립 학습
- 개인 점령 보상(종료 계수 0) / 팀 terminal 승패 보상(종료 계수 1)
- local / aware와 조합하여 4조건 × 2알고리즘 × 3seed = 24회 학습
- frontier 초기 배치, 기존 환경 규칙과 compact_v1 관측 인코딩 유지
- 학습 중 random·scripted 봇 없음. 고정 정책은 평가에서만 사용

## 평가와 선정

학습된 두 팀 묶음을 각각 random/expand/aggressive/defend 상대에 평가합니다.
validation은 팀 묶음 2 × 상대 4 × seed 10 × 진영 2 = 후보당 160경기입니다.
같은 validation의 학습 전 Q=0 기준선도 저장합니다.
seed 3개 중 validation 평균 경기 점수(승 1, 무 0.5, 패 0)가 높은 모델을 선정하며,
동점이면 최종 면적 비율, 완전 동률이면 seed 목록 순서를 사용합니다.
두 팀 묶음의 성적은 같은 가중치로 합치고 묶음별 성적도 기록합니다.
선택 모델의 test는 팀 묶음 2 × 상대 4 × seed 30 × 진영 2 = 480경기입니다.
평가는 epsilon=0이며 Q 동률만 seeded 무작위입니다. Q와 방문 통계를 갱신하지 않습니다.
서로 다른 보상함수의 누적 보상으로 모델을 선정하지 않습니다.

평가 master=20261031, validation 환경 domain=40/정책=41,
test 환경 domain=50/정책=51입니다. 실제 seed는 기존 SeedSequence 규칙으로 파생합니다.
학습 중 상대도 학습하므로 학습 대전 승률 자체를 발전·수렴의 기준으로 삼지 않습니다.

## 중간 진단과 자원 측정

- 학습 전 및 100에피소드마다 validation의 고정된 작은 부분집합 평가
  (팀 묶음 2 × 상대 4 × seed 2 × 진영 2 = 32경기)
- 학습 중간 test 평가나 test를 이용한 모델 선택·하이퍼파라미터 수정 없음
- state-action 방문 수, 2회/5회 이상 방문한 쌍 수, 평균 방문 횟수 기록
- 업데이트 타깃 오차, Q 업데이트 절댓값과 구간 최대값 기록
  (MC 타깃 오차를 TD error로 해석하지 않음)
- 첫 학습 에피소드에서 모델별 최대 128개 key를 고정한 probe Q 변화 기록
  (전체 상태 공간의 수렴을 입증하는 값은 아니며 사망·부분 관측도 포함 가능)
- 에피소드별 순수 학습 CPU/경과 시간과 단계별 CPU/경과 시간 기록
- Windows working set·프로세스 시작 이후 peak working set·private commit 기록
  (private commit과 실제 상주 RAM은 구분)

진단용 평가는 학습용 환경/RNG와 분리되어 있고 다음 에피소드의 난수에 영향을 주지 않습니다.
Counter와 오차 집계는 행동 선택·Q 타깃에 사용하지 않으며 추가 RNG도 소비하지 않습니다.
고정 alpha·부분 관측·현재 예산에서 엄밀한 수렴은 보장하지 않습니다.

## 저장물

새 `runs/<시각>/` 아래에만 저장합니다.

- manifest.json: 정확한 설정·시작/종료 시각·전체 CPU/경과 시간·버전·소스 SHA256
- source_snapshot/: 이번 실행에 사용한 학습/환경 소스의 복사본
- resources.csv: 학습(진단 평가 포함)·validation·기준선·test 단계별 시간과 메모리
- progress.json: 최근 완료 후보 수와 중간 진단 상태
- 후보별 diagnostics.csv: 중간 평가·방문·Q 변화·시간·메모리
- 후보별 probe_keys.json.gz: Q 변화를 비교한 고정 key 목록
- 후보별 training.csv: 양 팀/모델별 보상·업데이트 수와 에피소드 순수 학습 시간
- 후보별 model.json.gz: 최종 모델 6개; 관전 프로그램으로 로드 가능
- 기존 validation.csv / untrained_validation.csv / selected_test.csv / results.json / selected.csv

학습 phase 시간에는 중간 진단이 포함됩니다. 순수 에피소드 학습 시간은 training.csv를,
중간 평가 비용은 diagnostics.csv를 보아 구분합니다. peak 메모리는 해당 구간만의
최대값이 아니라 프로세스 수명 전체의 최대값입니다.
전체 실행이 정상 종료되면 manifest status가 complete가 됩니다.
