# 아카이브 정리 기록

정리일: 2026-11-01 (Asia/Seoul). 대상은 `smoke_test_mc_td` 내부입니다.
이 아카이브 이름의 20261031은 이전 실험의 실행일을 뜻합니다.

## 이동한 이전 자료

| 원래 경로 | 현재 위치(이 아카이브 기준) | 의미 |
| --- | --- | --- |
| SMOKE_TEST_REPORT_20261031.md | SMOKE_TEST_REPORT_20261031.md | 이전 한 명 학습 보고서 |
| resource_profile.json | resource_profile.json | 이전 대표 후보 자원 측정 |
| six_learner_benchmark.json | six_learner_benchmark.json | 1명/6명 속도 비교 원자료 |
| analyze_results.py | analyze_results.py | 이전 보고서 생성 코드 |
| benchmark_learning.py | benchmark_learning.py | 이전 속도 측정 코드 |
| profile_runs.py | profile_runs.py | 이전 후보 재학습 자원 측정 코드 |
| runs/20261031T085626900021Z/ | runs/20261031T085626900021Z/ | 초기 quick 연결 검사 |
| runs/20261031T093149737642Z/ | runs/20261031T093149737642Z/ | 이전 한 명 학습 180후보 결과 |

8개 최상위 항목, 내부 파일 총 1,052개를 이동했습니다. 내용을 수정하거나 삭제하지
않았습니다. 파일별 원래 경로·새 경로·바이트 크기·SHA256은 cleanup_manifest.json의
moved_files에 있습니다. 이전 보고서의 상대 경로를 유지하도록 runs 하위 구조도 보존했습니다.

여기의 Python 파일과 명령 예시는 역사 보관용입니다. 옛 실행 경로·버전에 대한
가정이 있으므로 현재 학습 진입점으로 사용하지 마세요. 현재 학습은 모듈 루트의
`run.py --preset agreed`, 관전은 `viewer.py`를 사용합니다.

## 현재 코드 의존성 수정

- 새 runtime_metrics.py: 기존 메모리 측정 함수 본문을 옮긴 현재 공용 모듈입니다.
- run.py와 runner.py: profile_runs 대신 runtime_metrics.memory를 가져옵니다.
- 학습 알고리즘·보상·관측·최신 하이퍼파라미터는 변경하지 않았습니다.
- paper_report/prepare_paper_data.py: 기록된 source_snapshot의 해시를 기준으로 검사합니다.
  현재 소스의 이동·변경은 경고 정보로 남기며 재수집을 막지 않습니다.
- 같은 재수집 스크립트의 추론은 별도 CLI 프로세스에서 검증한 source_snapshot 코드를
  사용합니다. 현재 코드가 나중에 바뀌더라도 학습 당시 추론 구현을 사용하도록 했습니다.
- 이전 실행은 archive/*/runs/<실행 ID>/에서 먼저 찾고, 정리 전 runs 경로도 지원합니다.
- test_maintenance.py: 위 경로·무결성·메모리 계측을 검사하는 테스트를 추가했습니다.

## 문서와 노트북 수정

- README.md: 최신 4조건·3 seed·4×4·64틱·1,000에피소드·양 팀 6명 구성으로 재작성했습니다.
  최신 보고서, 확정 설정, 관전, 아카이브를 연결합니다.
- VIEWER.md: 최신 여섯 명 체크포인트와 실제 대표 모델 경로로 예시를 바꿨습니다.
- smoke_test_mc.ipynb / smoke_test_td.ipynb: agreed_config()를 사용합니다.
  기본 FULL_TRAINING=False는 최신 네 조건·세 seed의 작은 연결 검사이고,
  True일 때만 해당 알고리즘의 전체 12후보 학습을 시작합니다. 옛 출력은 초기화했습니다.
- paper_report/build_paper_report.py / REPORT.md: 보존 소스 검증과 새 아카이브 위치를 설명합니다.
- paper_report/artifacts_manifest.json: 바뀐 보고서와 생성 코드의 해시를 갱신했습니다.
  기존 데이터·그림·GIF·AVI는 다시 만들거나 바꾸지 않았습니다.

호환성을 위해 RunConfig 기본값과 --preset full의 과거 범용 설정은 유지했습니다.
최신 실험을 재현할 때는 agreed_config()/--preset agreed를 사용하세요.
이미 열려 있는 노트북 커널은 재시작한 뒤 갱신된 셀을 사용하세요.

## 보존·백업

- 최신 runs/20261031T112320430780Z/ 전체: 원래 위치와 바이트 그대로 보존
- 최신 모델 24개·학습/평가 로그·source_snapshot: 수정하지 않음
- paper_report의 그래프·영상·분석 데이터: 원래 위치와 바이트 그대로 보존
- territory_rl 환경 코드: 읽기만 했으며 수정하지 않음
- 수정 전 파일 10개: pre_cleanup_sources/ 아래에 원래 상대 경로로 백업
- 새 runtime_metrics.py와 test_maintenance.py는 기존 파일을 덮어쓴 것이 아닙니다.

## 검증과 실행 내역

정리 전후 이동 파일 1,052개, 수정 전 백업 10개와 보호 파일 274개의 SHA256을 검사합니다.
학습/정리 검사 19개, 관전 추론 검사 5개를 실행하며 GUI 창은 띄우지 않습니다.
메모리 계측을 실제 저장 흐름에서도 확인하기 위한 작은 1후보 통합 검사는
cleanup_validation/에 저장합니다. 전체 1,000에피소드 학습은 다시 실행하지 않습니다.
보고서 경로·보존 소스·아카이브 검색은 --verify-only로 검사합니다.
실제 성공 여부와 완료 시각은 cleanup_manifest.json의 status/post_checks를 확인하세요.

## 되돌리거나 다시 검토할 때

cleanup_manifest.json의 moves/backups/installs에 모든 경로와 해시가 있습니다.
pre_cleanup_sources에는 코드·안내문·노트북·보고서·산출물 해시표의 수정 전 사본이 있습니다.
최신 실행 manifest와 source_snapshot은 과거 실험을 설명하는 기록이므로 현재 코드에
맞추려고 덮어쓰지 마세요. 아카이브 자체를 다른 위치로 옮기면 archive 검색 규칙과
이전 보고서의 상대 링크도 함께 검토해야 합니다.

## 최종 검증 결과

학습·정리 검사 19개, 관전 추론 검사 5개와 작은 1후보 통합 검사가 모두 통과했습니다.
보존 소스 검사와 아카이브 탐색, 보존 소스로부터의 추론 모듈 import가 통과했습니다.
이동 1,052개·백업 10개·보호 274개 및 보고서 산출물 32개의 해시가 일치했습니다.
전체 학습 재실행은 하지 않았습니다. 최신 모델·그림·영상과 환경 코드는 바꾸지 않았습니다.
