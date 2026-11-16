# Territory Control 환경 실행

3v3 인접 타일 공격 환경이다. 현재 구현 버전은 `territory_v1`이며 학습기 없이 환경·키보드·scripted 에이전트를 실행한다.

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

`headless`와 pytest는 환경 기능을 실행하며 모델 학습을 하지 않는다. 로그에는 설정, 규칙 버전, 회차별 원요청/정규화 행동, 최종 행동, 점령 기여, 보상 항, 종료 이유가 포함된다. replay는 회차 수집과 최종 전이를 모두 검증한다.

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

```python
from territory_rl import Action, ExperimentConfig, TerritoryEnvironment

env = TerritoryEnvironment(experiment=ExperimentConfig(decision_strategy="feedback"))
env.reset(seed=7)
obs = env.observe("blue", 2)  # 팀 이름과 팀 내 번호 0..2
# obs["owner"]: 아군=1, 적군=-1; 좌표와 방향은 회전/반전하지 않는다.
# 에이전트 배열: 아군_0..2, 적군_0..2; self_index는 요청한 번호다.

env.begin_decision()
session = env.agent_session("blue", 2)
session.submit_action(Action.NOOP)
assert env.finish_decision_round() is None  # 누락=정지, 1차 제안을 동결
second = env.agent_session("blue", 2)       # 회차가 바뀌면 세션을 다시 받는다.
proposals = second.observe()["team_action_codes"]
second.submit_action(Action.NOOP)
result = env.finish_decision_round()        # 2차 최종 행동만 실행
```

`owner`는 ±1이다. `agent_teams`는 아군 1/적군 0의 별도 소속 표시이고 `agent_alive`는 생존 마스크다. `team_action_codes`는 자신 포함 아군 3명 번호순이다. 미관측 0, 정지 1, 이동 2~5, 공격 6~9다. 미관측 one-hot은 전부 0이며 `team_actions_known`으로 확정 NOOP과 구분한다. 되먹임 2차는 최종 제출이 아닌 동결된 1차 제안을 관측한다. 환경이 저장한 전략과 회차에 맞게 공개하므로 관측 호출에 외부 행동 버퍼를 전달하지 않는다.

`Action.index`는 길이 9 마스크의 0 기반 인덱스다. 규칙 엔진과 세션은 실행 코드 1..9를 받고 0을 거부한다. **표준 `TerritoryParallelEnv.step`만 `Discrete(9)` 인덱스 0..8을 받아 변환한다.** `Action.from_index`와 `.index`로 명시적으로 변환한다.

Parallel adapter의 일괄 step은 개별 전략용이다. 순차·되먹임은 adapter의 `begin_decision`, `agent_session`, `finish_decision_round`로 수집한다. 수집을 우회한 step은 거부한다. 코어의 직접 `step(actions)`는 최종 행동 집합을 재현하는 규칙 실행용이며 결정 전략을 자동 실행하지 않는다. 자동 컨트롤러 연결에는 `EpisodeManager.step()`, 수동 단계 진행에는 `EpisodeManager.advance()`를 사용한다.

## 버전 변경

v0의 0=정지, 지역 격자 local/aware 관측 및 `observation_radius`/`observation_mode`/`terminal_bonus` 설정은 새 API로 대체했다. 이전 로그를 새 행동 코드로 조용히 재생하지 않고 버전 오류로 거부한다. 과거 설정은 새 전략·보상 설정으로 명시적으로 이전해야 한다. 상세 규칙은 [env.md](env.md), 구현 항목과 검증 범위는 [implementation_plan.md](implementation_plan.md)를 참고한다.
