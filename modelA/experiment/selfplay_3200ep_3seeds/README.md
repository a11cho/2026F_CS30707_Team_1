# 변경 전 실행 코드 안내

이 폴더는 면적 차이 변화량 보상(gap_delta)으로 수행한 이전 실험입니다.
학습 체크포인트·평가·그래프·리플레이는 원래 위치에 보존했습니다.

당시 실행 코드와 실험 설정은 [코드 아카이브](../../archive/pre_area_share_20261118/README.md)의
modelA/ 아래에 있습니다. 아카이브 경로는 modelA/archive/pre_area_share_20261118/입니다.
변경 전 코드는 평가 시 반대 진영에 좌우반전을 적용하지 않았습니다.

현재 modelA 기본 코드는 점유율 보상(area_share)과 학습 진영 기준 방향 정렬 평가를
지원합니다. 이 폴더의 과거 지표는 현재 기본 보상으로 학습한 결과가 아닙니다.
기존 signal 없는 체크포인트를 현재 코드로 읽으면 gap_delta를 유지합니다.
현재 코드로 재평가하면 기본 반전이 적용되므로 옛 평가를 재현할 때는
--set evaluation.mirror_to_training_side=false를 명시하세요.

아카이브 실행기를 사용하는 명령은 저장소 루트에서 다음처럼 실행합니다.

    python -B modelA/archive/pre_area_share_20261118/run_archived.py --help
