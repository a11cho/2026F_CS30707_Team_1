# 저장 모델 시뮬레이션 관전

`viewer.py`는 학습과 분리된 실행 프로그램입니다. 기존 환경·인코더·모델 로더를
읽어서 사용하며 `train()`, Q 업데이트, 체크포인트 저장을 호출하지 않습니다.
관전 중 Q-table 사전과 배열은 읽기 전용이며, 원본 모델 파일을 수정하지 않습니다.
이 기능은 전체 그리드 학습을 시작하거나 이어서 학습하지 않습니다.

## 1. 창을 열어 모델 선택

프로젝트 루트 `2026F_CS30707_Team_1`의 PowerShell에서 실행합니다.

```powershell
$rlPython = 'C:\Users\dohyu\anaconda3\envs\rlTeam\python.exe'
& $rlPython -B -m smoke_test_mc_td.viewer
```

창에서 **체크포인트 열기…**를 누르고 학습 결과의 `model.json.gz`를 선택하세요.
처음에는 일시정지 상태입니다. 체크포인트를 바꾸면 대결 모드는 자동으로 선택됩니다.

- 모델 6개인 새 체크포인트: 양 팀 모두 저장된 모델로 대결
- 모델 1/3개인 기존 체크포인트: 학습 모델과 고정 정책으로 대결
- 예전 한 명 학습 파일은 여섯 명 학습 모델이 아니므로 양 팀 모델 모드를 지원하지 않습니다.

현재 전체 실행 `20261031T093149737642Z`는 한 명 학습 결과입니다. 여섯 명 모드는
앞으로 양 팀 학습으로 저장하는 체크포인트에서 사용할 수 있습니다.

## 2. 파일 경로를 지정해 바로 열기

기존 실제 체크포인트를 여는 예시입니다. 한 명 모델은 선택 진영의 0번 에이전트를
담당하고 같은 팀 나머지 두 명은 기존 방식대로 expand 정책을 사용합니다.

```powershell
$checkpoint = 'smoke_test_mc_td/runs/20261031T093149737642Z/aware_individual_gross_b0/sarsa/seed_7/model.json.gz'
& $rlPython -B -m smoke_test_mc_td.viewer $checkpoint --opponent random --seed 7
```

여섯 명 체크포인트에서는 경로만 지정해도 양 팀 대결이 됩니다. 또는 고정 상대와
대결시킬 학습 팀 묶음을 고를 수 있습니다.

```powershell
# $checkpoint에 여섯 모델이 든 체크포인트 경로를 넣은 뒤 실행하세요.
# 양 팀: role 0~2와 role 3~5의 대결
& $rlPython -B -m smoke_test_mc_td.viewer $checkpoint --mode both
# 두 번째 묶음 role 3~5를 파랑 진영에 배치해 expand 상대와 대결
& $rlPython -B -m smoke_test_mc_td.viewer $checkpoint --mode fixed --cohort 1 --team blue --opponent expand
```

## 3. 화면 조작

| 기능 | 조작 |
| --- | --- |
| 재생 / 일시정지 | 버튼 또는 Space |
| 한 틱 진행 | 버튼 또는 Enter; 자동 재생 중이었다면 일시정지 |
| 같은 상황 재생 | 같은 seed로 초기화 버튼 또는 R |
| 속도 | 틱 간격 슬라이더: 20~2000ms |
| 진영·대결 방식 변경 | 설정을 바꾸고 설정 적용 · 처음부터 |
| 다른 상황 | 환경 seed를 변경한 후 적용 |
| 정책 동률 처리 난수 변경 | 정책 seed 변경; 빈칸이면 환경 seed에서 자동 파생 |
| 종료 | Esc 또는 창 닫기 |

입력칸·콤보박스·텍스트에 포커스가 있을 때는 Space/Enter/R 관전 단축키가 작동하지
않으므로 버튼을 사용하세요. 종료 틱에 도달하면 자동 일시정지합니다.

왼쪽에는 전체 맵이 표시됩니다. 오른쪽에는 현재 틱, 두 팀 면적, 환경·정책 seed,
에이전트별 모델 role, 좌표·생존·공격 가능 상태, 직전 행동, 누적 보상과 사건이
표시됩니다. 에이전트 목록과 사건 목록은 스크롤할 수 있습니다.

모델의 직전 선택에 표시되는 **기방문/미방문**은 해당 관측 key가 저장된 Q-table에
있는지, **최댓값 N개**는 허용 행동 중 같은 최대 Q값의 행동이 몇 개인지 뜻합니다.
미방문 key와 Q 동률은 허용 행동 중 seeded 무작위로 고릅니다. epsilon은 0입니다.

## 4. 관전자 정보와 모델 입력의 분리

화면의 전체 맵과 모든 에이전트 정보는 관전자용입니다. 모델의 행동에는 계속
기존 local/aware 관측과 action mask만 전달됩니다. 화면에 보이는 전역 정보를
정책에 전달하지 않습니다. 맵 크기·틱·반경·보상은 체크포인트의 설정을 복원합니다.
관전에서 학습 당시와 다른 크기나 반경으로 임의 변경하지 않습니다.

같은 체크포인트·대결 모드·모델 묶음·진영·고정 상대·환경 seed·정책 seed를 쓰면
행동 동률과 무작위 부활까지 동일하게 재현됩니다. 화면 갱신과 속도는 상태 전이
난수를 소비하지 않습니다. GUI에서 모델을 바꾸거나 설정을 적용하면 처음부터
시작하며, 같은 seed로 초기화는 현재 적용된 두 seed를 그대로 사용합니다.

## 5. 창 없이 추론 연결만 검사

```powershell
& $rlPython -B -m smoke_test_mc_td.viewer $checkpoint --headless --opponent random
```

체크포인트의 마지막 틱까지 실행하고 최종 면적·보상·사건 수를 JSON으로 출력합니다.
파일은 저장하지 않습니다. 자동 테스트나 GUI 없는 환경에서 사용할 수 있습니다.

관전 기능만 검사하려면 다음 명령을 사용합니다. 테스트가 임시로 만드는 작은 모델은
`smoke_test_mc_td` 안에 생성 후 삭제합니다. 실제 학습 결과는 바꾸지 않습니다.

```powershell
# 창을 만들지 않는 기본 검사
& $rlPython -B -m smoke_test_mc_td.test_viewer
# 숨긴 Tk 창에서 버튼·타이머·초기화·모드 변경·정리 검사까지 수행
& $rlPython -B -m smoke_test_mc_td.test_viewer --gui
```

검사 대상은 추론 중 모델·체크포인트 불변성, 기존 평가와 동일한 전이, seed 재현,
렌더링 독립성, 1/3/6 모델 호환, UI 제어입니다. 기존 학습 테스트는 별도로
`python -B -m unittest smoke_test_mc_td.test_smoke -v`로 실행합니다.
