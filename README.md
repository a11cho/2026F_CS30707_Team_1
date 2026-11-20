# Territory Control 환경 실행

3v3 인접 타일 공격 환경이다. 학습기 없이 환경·키보드·scripted 에이전트를 실행한다.

## 설치 및 실행

PowerShell에서 `project` 디렉터리를 기준으로 실행한다. 기존 `.venv`가 있으면 설치 단계는 생략할 수 있다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\play.ps1 -ManualAll -Strategy feedback -RewardMask 0,1,0
```

전략은 `independent`(개별), `sequential`(순차), `feedback`(되먹임) 중 선택한다. `-ManualAll`을 생략하면 선택한 한 명만 키보드, 나머지는 scripted 컨트롤러다.

```powershell
.\.venv\Scripts\python.exe -m territory_rl --strategy sequential --manual-all
.\.venv\Scripts\python.exe -m territory_rl --mode headless --strategy feedback --reward individual --capture-weight 2 --log runs/feedback.jsonl
.\.venv\Scripts\python.exe -m territory_rl --mode headless --reward team --team-signal terminal --win-reward 1 --loss-reward 0
.\.venv\Scripts\python.exe -m territory_rl --mode replay --log runs/feedback.jsonl
.\.venv\Scripts\python.exe -m pytest -q
```

`headless`와 pytest는 환경 기능을 실행하며 모델 학습을 하지 않는다. 로그에는 설정, 회차별 원요청/정규화 행동, 최종 행동, 점령 기여, 보상 항, 종료 이유가 포함된다. replay는 회차 수집과 최종 전이를 모두 검증한다.

## 기본 설정과 보상

- 맵 8×8, 최대 128틱, `respawn_delay=0`. 종료가 아니고 빈 아군 타일이 있으면 사망 틱 후반에 부활하고 다음 틱부터 행동한다.
- 부활 딜레이 기능은 설정으로 남아 있다. `--height`, `--width`, `--ticks`, `--respawn-delay`로 기본값을 바꾼다.
- 종료는 최대 틱 도달 또는 한 팀의 생존자 0명·영역 0칸이다. 종료 틱에는 부활하지 않는다.
- 개인 보상은 실제 점령에 참여한 공격자마다 `capture_weight`를 지급한다. 단독 점령과 공동 점령에서 참여자 한 명이 받는 값은 같다. 점령에 참여하지 않은 에이전트의 개인 점령 보상은 0이다. 사망 보상 차감·생존 보너스·이동 페널티는 추가하지 않는다.
- 팀 `area`는 매 틱 현재 팀 면적 × `area_weight`, `gap`은 매 틱 (아군 면적 − 상대 팀 면적) × `gap_weight`를 각 팀원에게 지급한다. 두 항의 크기는 가중치로 조절한다. `terminal`은 실제 종료 때만 승/패/무승부 설정값을 지급한다.
- `--reward-mask AREA GAP TERMINAL`로 세 시그널을 선택한다. 여러 값을 1로 켜면 합산하고, 0인 항은 제외한다. 개인 점령 항은 이 마스크와 독립적이다.

미정 계수의 **실행용 초기값**은 개별 결정, 팀 area, 점령/면적/격차 가중치 1, 승/패/무승부 1/0/0이다. 연구적으로 확정된 계수나 성능이 검증된 설정을 의미하지 않는다. CLI의 `--help`와 `ExperimentConfig`에서 값을 변경할 수 있다.

## 보상 마스크 선택

마스크 순서는 **면적, 상대 격차, 종료**다. 격차는 이전 틱과의 차이가 아닌 현재 상대 팀과의 면적 차이다. 기본은 면적만 활성화한다.

| 선택 | 마스크 |
| --- | --- |
| 면적만 | `1 0 0` |
| 격차만 | `0 1 0` |
| 종료만 | `0 0 1` |
| 격차 + 종료 | `0 1 1` |
| 팀 시그널 전부 끄기 | `0 0 0` |

```powershell
.\.venv\Scripts\python.exe -m territory_rl --reward team --reward-mask 0 1 0 --gap-weight 1
.\play.ps1 -ManualAll -RewardMask 0,1,1
```

```python
experiment = ExperimentConfig(reward_signal_mask=(0, 1, 0), gap_weight=1.0)
env = TerritoryEnvironment(experiment=experiment)
env.reset(seed=7)
# 다음 에피소드에서 다른 마스크를 선택할 수도 있다.
env.reset(seed=8, options={"reward_signal_mask": [0, 0, 1]})
```

명시 마스크가 `--team-signal`과 `--include-terminal-reward`를 덮어쓴다. 마스크를 생략하면 두 옵션으로 팀 시그널을 선택한다. 개인 모드의 기본은 점령 항만 지급하지만 명시 마스크를 켜면 해당 팀 시그널을 추가한다. `(0,0,0)`으로 개인 점령 항까지 꺼지지는 않는다.

설정과 마스크는 reset 로그에 보존한다. `reward_components`의 `capture`, `area`, `gap`, `terminal`을 합하면 실제 보상이다. `capture_credit`은 API에서 해당 틱의 개인 점령 참여 횟수를 나타낸다. replay는 현재 보상 공식으로 다시 계산한 결과를 로그와 비교한다. 키보드 화면에는 현재 마스크를 표시한다.

격차는 고정 맵에서 `2 × 아군 면적 − 전체 면적`이므로 면적 보상과 밀접하다. 상대 우세를 부호로 표시하고 매 틱 피드백을 제공하려는 선택이며, 학습 개선 여부는 별도로 검증할 가설이다.

## 키보드 조작

- WASD: 이동, 방향키 또는 IJKL: 공격, Space: 정지 예약.
- 1~6 또는 Tab: 조작 대상 선택. 화면 번호 1~3은 API의 팀 내 번호 0~2에 대응한다.
- Enter: 현재 결정 단계 제출. 개별은 한 틱, 순차는 한 명, 되먹임은 한 회차씩 진행한다.
- 순차의 전체 수동 모드는 다음 제출자를 자동 선택한다. 여섯 명의 제출이 끝나야 한 틱 진행한다.
- 되먹임은 첫 Enter로 제안만 수집한다. 화면의 `first_pass_proposal`을 보고 다시 키를 예약한 뒤 두 번째 Enter로 최종 행동을 실행한다. 두 번째 입력이 없으면 정지다.
- P: 단계 자동 진행/일시정지, R: 같은 seed로 재시작, Esc: 종료.

화면은 전체 물리 상태의 디버그 뷰다. 관측 API의 아군/적군 상대 부호와 별개로 실제 red/blue 색을 표시한다. 전략·회차·다음 제출자·선택한 에이전트가 볼 수 있는 행동도 표시한다.

## 관측 및 행동 API

관측은 `tilemap`, `team_actions`, `auxiliary` 세 항목을 반환한다.

| 항목 | 형식과 의미 |
| --- | --- |
| `tilemap` | 기본 `(15, 15)` int8 격자. 본인의 `position`을 `(7, 7)`에 대응시켜 전체 맵을 그린다. |
| `team_actions` | `(3,)` int8. 자신 포함 아군 0..2의 공개된 이번 틱 행동이다. |
| `auxiliary` | 소속·생존·공격 가능·부활 대기, 본인 번호, 면적, 시간, 결정 전략·회차·행동 출처, 행동 공개 마스크·one-hot, 행동 마스크를 담는 사전이다. |

기본 타일 값은 맵 밖 0, 빈 아군/적군 영역 +1/-1, 아군 0..2는 +2/+3/+4, 적군 0..2는 -2/-3/-4다. 타일 값은 `config.py`의 `TileEncoding`에 모았으며 실행별로도 변경할 수 있다. 모든 값은 서로 다른 int8 정수다. 관측 생성과 scripted 봇은 이 설정을 함께 사용한다.

```python
from territory_rl import Action, ExperimentConfig, TerritoryEnvironment, TileEncoding

encoding = TileEncoding(
    outside=0,
    friendly=1,
    enemy=-1,
    friendly_agents=(5, 6, 7),
    enemy_agents=(-5, -6, -7),
)
experiment = ExperimentConfig(decision_strategy="feedback", tile_encoding=encoding)
env = TerritoryEnvironment(experiment=experiment)
env.reset(seed=7)
obs = env.observe("blue", 2)
assert obs["tilemap"][7, 7] == encoding.friendly_agents[2]
mask = obs["auxiliary"]["action_mask"]

env.begin_decision()
env.agent_session("blue", 2).submit_action(Action.NOOP)
assert env.finish_decision_round() is None
second = env.agent_session("blue", 2)
proposals = second.observe()["team_actions"]
second.submit_action(Action.NOOP)
result = env.finish_decision_round()
```

`team_actions`는 미공개 0, 정지 1, 이동 2..5, 공격 6..9다. `independent`는 본인 제출만, `sequential`은 앞서 확정한 아군 행동을 공개한다. `feedback` 2차는 동결된 아군 3명의 1차 제안을 공개한다. 공개 여부와 원-핫 표현은 `auxiliary`에 보존한다. 에이전트별 보조 배열은 아군 0..2 다음 적군 0..2 순서다.

사망해도 `position`은 사망 직전 좌표를 유지하며 `alive=False`로 점유를 해제한다. 그 좌표를 관측 중심으로 사용하고 사망자는 타일맵에 그리지 않는다. 부활하면 새 좌표가 중심이 된다. 사망 초기 시나리오에도 맵 안의 `position`, `alive=False`, `respawn_at`을 지정한다.

### 공통 관측 회전

`randomize_observation_rotation=False`가 기본이다. 활성화하면 reset마다 환경 공통 각도 하나를 0/90/180/270도 중 추첨하고, 모든 에이전트가 에피소드 동안 동일한 시계 방향 회전을 사용한다. 배치·부활 RNG와 관측 회전 RNG는 분리한다. 테스트·재생에서는 활성 설정과 함께 `reset(options={"observation_rotation": 1})`처럼 90도 단위 값을 지정할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m territory_rl --mode headless --randomize-observation-rotation --log runs/rotated.jsonl
.\play.ps1 -ManualAll -RandomizeObservationRotation
```

타일맵, 공개 행동, `auxiliary.action_mask`, `session.available_actions()`는 모두 관측 방향이다. **세션 제출과 `TerritoryParallelEnv.step()`은 관측 방향 행동을 받아 실제 맵 방향으로 역변환한다.** 예를 들어 시계 방향 90도 관측의 오른쪽 이동은 실제 맵 위쪽 이동이다. 키보드 디버그 맵도 같은 각도로 표시한다.

세션은 실행 코드 1..9를 받으며, 표준 Parallel step은 `Discrete(9)` 인덱스 0..8을 받는다. `Action.from_index`와 `.index`로 변환한다. `TerritoryEnvironment.step()` 및 `action_mask()`는 규칙 실행용 실제 맵 방향 API다. 수동 변환에는 `action_to_world()`와 `action_to_observation()`을 사용한다.

Parallel 일괄 step은 independent용이며 sequential/feedback은 `begin_decision`, `agent_session`, `finish_decision_round`로 수집한다. 자동 컨트롤러는 `EpisodeManager.step()`, 수동 단계 진행은 `EpisodeManager.advance()`를 사용한다. 사용자 정의 타일 값을 쓰는 봇에는 `FrontierController(..., tile_encoding=experiment.tile_encoding)`을 전달한다.

윈도우는 `observation_window_size=15`가 기본이다. 홀수이며 `2 * max(height, width) - 1` 이상이어야 한다. 더 큰 맵은 `--observation-window-size` 또는 ExperimentConfig로 충분한 크기를 지정한다.

로그의 reset에는 회전 횟수와 타일 설정을 기록한다. 회차별 `observation_requested`는 관측 방향, `requested`와 `applied` 및 최종 `actions`는 실제 맵 방향이다. 보상은 현재 공식으로 검증한다.

상세 규칙은 [env.md](env.md), 구현 항목과 검증 범위는 [implementation_plan.md](implementation_plan.md)를 참고한다.
