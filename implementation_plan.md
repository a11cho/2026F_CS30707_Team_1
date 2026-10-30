# 강화학습 환경 객체 지향 구현 계획

이 문서는 [env.md](env.md)에 정의된 인접 타일 공격형 3v3 게임을 Python으로 구현하기 위한 계획이다. 2026-10-30에 기존 환경 사례를 비교하여 보완했다. 평가 근거와 출처는 [environment_design_review.md](environment_design_review.md)에 기록한다. 이번 단계는 설계이며 구현 코드는 작성하지 않는다. 보상·부활·사망 시점 등 미정 게임 규칙은 아래 권고를 채택한 뒤 `env.md`에 명시해야 한다.

핵심 보완은 **일괄 행동 API, 전이 이벤트, 개인/팀 보상 귀속, local/aware 관측 계약, 학습 연결을 앞당긴 검증 순서**다. `TerritoryEnvironment`를 규칙 엔진으로 두고, 기존 session 방식은 사람·scripted 실행용 facade로 유지한다.

## 1. 설계 원칙

1. **상태의 소유자를 하나로 정한다.** `TerritoryEnvironment`가 에이전트 상태와 현재 틱을 소유하고, `TileMap`이 타일 소유권을 소유한다. 매니저와 시각화는 이 상태를 직접 수정하지 않는다.
2. **규칙 판정과 상태 변경을 분리한다.** 공격·이동 판정 클래스는 현재 상태와 행동을 받아 결과를 계산한다. 환경이 그 결과를 한 번에 반영한다. 에이전트 처리 순서가 결과를 바꾸지 않게 하기 위해서다.
3. **보상·부활·관측을 독립적인 설정으로 둔다.** 보상의 개인/팀 귀속과 terminal/per-tick 지급 목표를 구별한다. 네 실험 조건은 동일 전이 규칙을 사용한다.
4. **외부에는 읽기 전용 결과를 돌려준다.** 관측과 화면용 스냅샷을 통해 내부 상태가 외부에서 수정되지 않게 한다.
5. **게임 규칙과 실행 도구를 분리한다.** 환경은 에피소드 실행 방식, 학습 알고리즘, 화면 표시 방법을 알 필요가 없다.
6. **추상화는 필요한 만큼만 둔다.** 상태 없는 resolver는 함수로 시작해도 된다. 클래스 수나 파일 수를 완료 기준으로 삼지 않으며, 범용 ECS·네트워크·JAX/C++ 최적화는 MVP에서 제외한다.
7. **학습과 분석의 근거를 보존한다.** 보상으로 압축하기 전 소유권 변화·기여자·이동 실패·사망·부활 사건을 남긴다. 로그와 보상 계산은 난수를 소비하거나 상태를 변경하지 않는다.

## 2. 예정된 파일 구조

```text
project/
├─ env.md
├─ implementation_plan.md
├─ environment_design_review.md
├─ configs/              # 규칙, 2×2 실험, 평가 상대/seed 설정
├─ territory_rl/
│  ├─ __init__.py
│  ├─ models.py          # 행동, 좌표, 관측, 결과 등의 값 객체
│  ├─ events.py          # 소유권 변화, 공격/이동 결과, 사망·부활 사건
│  ├─ config.py          # 게임 설정
│  ├─ agent.py           # 에이전트 상태
│  ├─ controller.py      # 에이전트의 행동 선택 인터페이스
│  ├─ map.py             # 타일 소유권
│  ├─ attack.py          # 동시 공격 판정
│  ├─ movement.py        # 동시 이동 판정
│  ├─ reward.py          # 보상 규칙
│  ├─ respawn.py         # 부활 시점 규칙
│  ├─ observation.py     # 에이전트에게 보여줄 정보
│  ├─ environment.py     # 상태 소유, step(actions)로 한 틱 전이
│  ├─ parallel_env.py    # PettingZoo Parallel API adapter
│  ├─ session.py         # AgentSession 및 틱별 행동 수집
│  ├─ manager.py         # 에피소드 실행과 기록
│  ├─ metrics.py         # 보상과 독립적인 평가 지표
│  └─ visualization.py   # 화면 표시
├─ scripts/              # headless 실행, 전술 검증, 이후 학습·평가 진입점
└─ tests/
   └─ ...               # 규칙별 검증 코드
```

이 구조는 책임의 경계이며 모든 파일을 첫날부터 만들 필요는 없다. 예를 들어 이동 충돌은 `movement.py`, 보상 귀속은 `reward.py`에서 바꿀 수 있어야 한다. 학습 알고리즘은 별도 모듈로 추가하되 작은 PPO 연결 검증은 전체 GUI 완성 전에 수행한다. NumPy/Gymnasium/PettingZoo/pytest와 선택한 학습기의 실제 사용 버전은 구현 시작 시 고정한다.

## 3. 클래스와 책임

| 파일 | 클래스·자료형 | 소유하거나 수행하는 일 | 하지 않는 일 |
| --- | --- | --- | --- |
| `models.py` | `Action`, `Position`, `Observation`, `StateSnapshot`, `StepResult` | 9개 행동과 좌표 표현, 외부로 전달할 변경 불가능한 데이터 | 게임 상태 변경 |
| `config.py` | `GameConfig`, `ExperimentConfig` | 전자는 맵·틱·전이 규칙, 후자는 reward/observation mode·상대·seed·학습 예산 검사 | 틱 진행 |
| `agent.py` | `AgentState` | ID, 팀, 위치, 생존 여부, 마지막 공격 시도 틱, 부활 예정 틱, 직전 완료 틱의 적용 행동과 유효 여부 | 다른 에이전트와의 충돌 판정 |
| `controller.py` | `AgentController` | 에이전트의 판단 로직이 구현할 인터페이스; 자신의 환경 접속 객체에서 관측·가능 행동을 조회하고 행동 제출 | 게임 상태 소유, 틱 진행 |
| `map.py` | `TileMap` | 타일 소유권 저장, 초기 절반 분할, 경계·벽 검사, 소유권 변경, 면적 계산 | 에이전트 위치 저장 |
| `attack.py` | `AttackResolver` | 모든 공격을 타일별·팀별로 집계하여 변경할 소유권 계산 | 맵 직접 변경, 보상 지급 |
| `movement.py` | `MovementResolver` | 모든 이동 의도를 함께 받아 성공·실패와 최종 위치 계산 | 에이전트 상태 직접 변경 |
| `events.py` | `TransitionEvents` | 타일 변화와 기여자, 행동 판정 및 사망·부활 결과 | 보상값 결정, 상태 변경 |
| `reward.py` | `RewardPolicy`, `RewardBreakdown` | 전후 상태와 사건으로 개인/팀 보상 및 세부 항 계산; terminal/per-tick은 별도 설정 | 틱 진행, 맵 변경 |
| `respawn.py` | `RespawnPolicy`, `ImmediateRespawn`, `DelayedRespawn` | 사망 틱에서 부활 가능한 틱을 결정 | 부활 위치 선정, 에이전트 상태 직접 변경 |
| `observation.py` | `ObservationBuilder` | 환경의 현재 정보에서 특정 에이전트의 관측 생성 | 내부 상태 변경 |
| `environment.py` | `TerritoryEnvironment` | 맵·에이전트·틱·환경 난수 상태를 소유하고 `step(actions)`를 원자적으로 적용 | 정책 실행, 화면 출력, 입력 대기 |
| `parallel_env.py` | `TerritoryParallelEnv` | spaces, agents lifecycle, reset/step 반환 형식을 변환 | 별도의 충돌·보상 규칙 구현 |
| `session.py` | `AgentSession`, `ActionCollector` | 같은 틱 관측 조회와 행동 수집; ID·episode·tick 검증 | 게임 상태 변경, transition 구현 |
| `manager.py` | `EpisodeManager` | 에이전트별 컨트롤러 호출, 틱 진행, 에피소드 반복·결과 기록 | 규칙 판정, 내부 상태 직접 변경 |
| `metrics.py` | 지표 집계 함수 | 승패·면적·생존·충돌·기여 집계 | 학습 보상 변경 |
| `visualization.py` | `Renderer`, `TkRenderer` | `StateSnapshot`을 받아 타일·에이전트·틱·면적을 표시 | 환경 상태 변경, 틱 진행 |

`AgentController`, `RewardPolicy`, `RespawnPolicy`, `Renderer`는 작은 Protocol 또는 함수 계약으로 시작한다. 미정 대안을 모두 미리 구현하지 않고 채택한 규칙부터 만든다. 시각화는 `rgb_array` 출력을 먼저 지원하고, `TkRenderer`는 필요할 때 추가한다. headless 실행에서 tkinter를 import하거나 창을 생성하지 않도록 한다.

### 에이전트 상태의 소유권

`TerritoryEnvironment`는 `agent_id → AgentState` 컬렉션을 소유한다. 위치·팀·생존·부활 대기 정보는 이곳에만 저장한다. `AgentState`는 자신의 상태 전환에 필요한 작은 메서드만 제공한다. 공격 가능 여부는 별도 필드로 중복 저장하지 않고 **현재 틱, 마지막 공격 시도 틱, 생존 여부**로 계산한다. `TileMap`에는 에이전트 정보를 넣지 않는다.

`AgentState`는 게임 내부의 상태 객체이고, `AgentController`는 관측을 보고 행동을 선택하는 외부 객체다. 둘은 역할이 다르다. 컨트롤러는 `AgentState`를 직접 받지 않고 자신의 `AgentSession`으로만 환경에 접근한다.

타일은 `owner[n,m]`, 에이전트는 6개의 `AgentState`로 시작한다. 점유 배열이 필요하면 위치에서 매 틱 재생성하는 파생 데이터로 둔다. `alive=False`이면 위치는 `None`이며 내부의 `(-1,-1)` 배열 인덱싱으로 표현하지 않는다. frozen dataclass만으로 NumPy 배열까지 불변이 되지 않으므로, 관측·snapshot은 내부 배열과 메모리를 공유하지 않게 복사한다.

직전 행동은 완료된 틱의 적용 행동만 저장하고 현재 제출 버퍼와 분리한다. 대기 중 행동은 NOOP, reset 직후의 과거 행동은 유효하지 않은 것으로 표시한다. 사망·부활 때 마지막 공격 시도 틱을 지우지 않고 절대 틱 기준으로 cooldown이 자연히 끝나게 하는 안을 권고한다. 이를 채택할지 7절에서 명시한다.

## 4. 클래스 간 흐름과 공개 API

```text
학습기 → TerritoryParallelEnv.step(actions) → TerritoryEnvironment.step(actions)
EpisodeManager → AgentController → AgentSession → ActionCollector
       └──── 수집한 actions ─────────────────→ TerritoryEnvironment.step(actions)
                                                 ├─ TileMap + AgentState
                                                 ├─ AttackResolver + MovementResolver
                                                 ├─ TransitionEvents
                                                 └─ RewardPolicy + RespawnPolicy
완료 상태 → ObservationBuilder → 에이전트별 관측
TerritoryEnvironment.snapshot() → Renderer
```

`EpisodeManager`의 session facade가 `agent_session(agent_id)`를 제공한다. 세션은 ID에 묶인 `observe()`, `available_actions()`, `submit_action(action)` 창구이고, `step()`이나 `reset()` 권한을 제공하지 않는다. 제출 버퍼는 `ActionCollector` 하나만 소유한다. 학습기는 session을 거치지 않고 Parallel adapter에 행동 사전을 전달한다. 두 경로는 동일한 규칙 엔진을 사용한다.

환경과 세션의 기능은 다음과 같다.

| 메서드 | 동작 |
| --- | --- |
| 코어 `reset(seed, options)` | 맵·에이전트·틱·환경 RNG를 초기화한다. 선택적 options는 검증된 시나리오 설정만 허용한다. |
| facade `agent_session(agent_id)` | 해당 에이전트에게 묶인 `AgentSession`을 반환한다. |
| `AgentSession.observe()` | 자기 에이전트의 관측을 반환한다. 틱은 진행하지 않는다. |
| `AgentSession.available_actions()` | 자기 에이전트가 제출할 수 있는 행동을 반환한다. 직전 틱에 공격했다면 공격을 제외하고, 부활 대기 중이면 빈 목록을 반환한다. |
| `AgentSession.submit_action(action)` | 자기 에이전트의 현재 틱 행동을 collector에 접수한다. episode/tick token으로 오래된 제출을 거부한다. |
| 코어 `step(actions)` | 미제출 행동을 정지로 채우고 일괄 전이를 적용한다. `StepResult`에 사건·보상 내역·종료를 담는다. |
| adapter `reset(seed=None, options=None)` | `observations, infos` 사전을 반환한다. |
| adapter `step(actions)` | `observations, rewards, terminations, truncations, infos`를 에이전트 ID별로 반환한다. |
| adapter `observation_space(id)`, `action_space(id)` | 고정된 Dict 관측 공간 및 `Discrete(9)`를 반환한다. |
| adapter `state()` | 중앙 학습 확장용 전역 수치 상태를 반환한다. 기본 IPPO actor/critic에는 전달하지 않는다. |
| `snapshot()` | 매니저의 기록과 시각화에 쓸 읽기 전용 상태를 반환한다. |
| adapter `render()`, `close()` | 렌더링과 자원 정리를 제공하며 상태 전이는 일으키지 않는다. |

`EpisodeManager`는 완료 상태의 관측을 고정한 뒤 컨트롤러를 호출하고, collector에서 행동을 받아 `step(actions)`를 한 번 호출한다. 제출 순서와 관측 조회 횟수가 결과를 바꾸지 않아야 한다. 미제출을 NOOP로 채우는 게임 규칙은 유지하되, 학습 adapter는 행동 key 누락을 오류로 알려 학습기 버그를 숨기지 않는 구성을 권고한다.

`possible_agents`는 6개 고정 ID다. **권고:** 부활을 기다리는 동안에도 `agents`에는 남겨 두고, 학습 API에서는 NOOP만 유효하게 한다. 게임 session의 ‘선택할 행동 없음’과 학습기의 dummy NOOP를 구별한다. 일시 사망에 termination을 주지 않으며 team 보상은 대기 중인 팀원에게도 전달한다. 사망 틱까지의 개인 credit도 소실시키지 않는다. T 도달 시 마지막 반환에는 6명의 최종 관측·보상을 포함하고 이후 `agents=[]`로 정리한다. 종료 뒤 전이는 reset 전까지 허용하지 않는다.

학습 adapter의 계약은 [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)를 따른다. 표준 API 채택만으로 특정 PPO 라이브러리에 자동 연결되는 것은 아니며 Dict 관측·mask·팀별 정책 routing을 작은 rollout에서 실제 확인해야 한다.

## 5. 한 틱의 상태 전이

아래 순서는 공격 직후 사망을 확정하는 권고안을 채택했을 때의 계약이다. 확정 전에는 3단계를 임의로 구현하지 않는다. `s_t`는 t회의 전이가 끝난 상태이며 다음 호출은 t+1번째 틱을 해결한다.

1. 틱 시작 생존자·위치·행동·면적을 고정한다. 행동 key·범위 검사를 상태 변경 전에 마친다. 미제출은 NOOP로 채우고, mask로 금지된 행동의 처리 규칙을 적용한다.
2. `AttackResolver`가 시작 상태에서 유효한 공격을 모두 집계한다. 환경이 타일 색을 한 번에 변경하고 공격 시도 및 소유권 변화 이벤트를 만든다. 같은 틱 죽게 되는 공격자의 유효 공격도 포함한다.
3. 공격 후 적 영역에 놓인 에이전트를 사망 처리하고 점유를 해제한다. 그 에이전트의 이동은 취소한다.
4. 생존자의 시작 위치와 이동 의도로 충돌·의존 관계를 만든다. 여러 이동이 한 목적지를 요구하면 모두 실패시킨다. 실패한 점유자의 칸을 원하는 이동에도 실패를 전파한다. 2명 교환 및 긴 순환은 선택한 규칙으로 해결한다.
5. 성공한 이동을 한 번에 적용한다. 도착 타일은 공격 결과가 반영된 소유권으로 검사하여 추가 사망을 처리한다. 죽었다고 다른 이동을 소급해서 성공시키지 않는다.
6. 팀별 면적·증감·개인 점령 credit과 이벤트를 확정한다. 부활은 영역을 바꾸지 않는다.
7. 다음 상태가 종료 상태가 아니라면 부활 예정자를 처리한다. 아군의 빈 타일 후보를 결정적인 순서로 만든 뒤 환경 RNG로 중복 없이 선택한다. 후보 부족 처리 규칙을 적용한다.
8. 보상·종료·틱 카운터를 갱신하고 다음 관측을 만든다. T 도달은 내재적 종료, 외부 강제 중단은 truncation으로 구별한다. 영역 0 자체로 조기 종료하지 않는다.
9. `StepResult`를 반환한다. session 경로에서는 collector를 비우고 새 tick token을 발급한다. 매니저는 결과를 기록한다.

이동 판정은 개별 에이전트를 순서대로 옮기지 않고, 출발지·목적지 관계를 먼저 만든 후 전체 결과를 계산한다. 특히 점유자가 이동에 실패하는 경우 그 칸으로 들어오려던 이동에도 실패가 전파되어야 한다.

경계 실패·동일 목적지 실패를 먼저 결정한 뒤, 비어 있는 칸으로 끝나는 연쇄를 성공시키고 정지/실패 점유자로 끝나는 연쇄를 실패시킨다. 남은 순환은 따로 판정한다. 한 틱에 사망은 한 번만 기록하고, 사망 원인 phase와 타일을 남긴다. 총 면적 보존 및 생존자 위치의 유일성을 매 전이 검증 대상으로 둔다.

## 6. 구현 순서

| 단계 | 작업 | 완료 기준 |
| --- | --- | --- |
| 0. 명세 고정 | 7절의 전이에 필요한 미정 규칙 결정 | 예시 입력·출력으로 사망/부활/충돌 합의, config의 미정값 검출 |
| 1. 값과 상태 | 상태·맵·config·이벤트 작성 | 절반 분할, 위치 유일성, seed 재현, 상태 외부 수정 방지 |
| 2. headless 전이 | 공격·이동·사망·부활·보상 연결 | 실제 6명 게임이 정확히 T틱 실행되고 이벤트와 면적이 일치 |
| 3. 학습 계약 | 관측·mask·Parallel adapter | API/seed 검사, 사망 대기와 최종 관측, local 정보 차단 검증 |
| 4. 정책 기준선 | random·확장·방어·집중 공격, session/manager | 전술 사례에서 협력 효과·교착·부활 악용 여부 측정 |
| 5. 작은 PPO 연결 | shared IPPO, scripted 상대 고정 | 2×2 모두 짧은 rollout/update 가능; masked action log-prob·return 검사 |
| 6. 실험 실행 | 보상/관측/예산 고정 후 여러 seed 학습·평가 | 학습에 없던 상대와 좌우 교환 평가, 보상과 별도 성과 로그 |
| 7. 데모 | RGB replay·Tk UI·사람 입력 수집 | 사람이 입력해도 같은 action trace의 headless 결과와 동일 |

단계별 검증은 `tests/`에 작성한다. 핵심 규칙 사례 외에 10절의 상태 불변식과 실험 계약을 검사한다. API 테스트 통과와 학습 성능 향상은 별개다. 초기 성능은 먼저 scripted 기준선 및 간단한 상대에서 확인한다.

## 7. 구현 전에 결정할 게임 규칙

아래는 [env.md](env.md)에서 아직 확정되지 않은 항목이다. 해당 단계에 들어가기 전 `env.md`에 결정을 기록한다.

| 항목 | 필요한 결정 |
| --- | --- |
| 보상 | 개인/팀 귀속, 공동 점령 credit, terminal/per-tick 목표, 계수·정규화; 주 실험과 추가 통제 구분 |
| 부활 | 같은 틱 후반 또는 `d`틱 뒤; 대기 틱 계산법, 빈 아군 칸이나 아군 영역이 없을 때 처리, 동시 부활 우선순위, cooldown 유지 여부 |
| 공격 후 사망 | 공격으로 자기 타일을 잃은 에이전트가 그 틱에 이동할 수 있는지 |
| 순환 이동 | 두 명의 자리 교환 외에 여러 명이 서로의 칸으로 이동하는 경우 |
| 가능한 행동 | 벽·맵 바깥·점유 칸을 향한 행동을 목록에서 제외할지, 선택은 허용하고 실행만 실패시킬지 |
| 제출 오류 | 제한된 공격 제출, 존재하지 않는 행동, 같은 틱의 중복 제출 처리 |
| 관측 | local 시야 반경, aware 팀원 상태/직전 행동, 사망 중 관측, 남은 틱, 상대 cooldown 공개 범위 |
| 시작 설정 | `n`, `m`, `T`, 초기 배치, 내부 장애물, 위치 중복 허용 여부 |
| 보상 전달 | 팀 보상을 팀원에게 어떻게 전달할지 |

규칙이 미정이어도 값 객체·맵·공격 집계 등 확정된 부분은 구현할 수 있다. 이동은 사망 후 점유와 순환 처리의 영향을 받으므로 연쇄 판정 단위까지만 독립 검증한다. 실행 config는 필요한 규칙이 미정이면 시작을 거부한다. 권고값을 은밀한 default로 채우지 않는다.

## 8. 관측과 행동의 구체적 계약

아래는 local/aware 비교를 위한 권고 형태다. `R`은 채택한 관측 반경, `C`와 feature 수는 명세 확정 후 고정한다. 맵 소유권은 팀 관점의 채널로 변환한다.

| 필드 | 형태/내용 | 실험 간 처리 |
| --- | --- | --- |
| `local_grid` | `float32[C, 2R+1, 2R+1]`; 아군 땅·적 땅·벽/맵 밖·본인·보이는 아군·보이는 적 | 동일 |
| `self_features` | 생존, local grid 유효 여부, 공격 제한, 부활 대기, 정규화된 남은 시간; 자기 위치 공개 여부는 양쪽 동일 | 동일 |
| `teammate_features` | `[2,F]`; 고정 슬롯의 위치·생존·공격 제한·부활 대기·직전 행동 | local에서는 전부 0 |
| `teammate_feature_mask` | 공개·유효 여부; 죽은 팀원의 위치와 초기 직전 행동 등을 구별 | local에서는 전부 0 |
| `action_mask` | `int8[9]`; 정지/이동 4/공격 4의 고정 순서 | 규칙과 공개 정보로 계산 |

행동 인덱스의 예시는 `NOOP=0, UP=1, DOWN=2, LEFT=3, RIGHT=4, ATTACK_UP=5, ATTACK_DOWN=6, ATTACK_LEFT=7, ATTACK_RIGHT=8`이다. 구현 시 한 번 고정하고 정책·렌더러·로그가 같은 enum을 사용한다.

공격 제한 틱에는 공격 mask만 0으로 만든다. 대기 중인 에이전트의 학습용 mask는 NOOP만 1이다. 적 타일·점유 타일 방향 이동을 무조건 mask하지 않는다. 팀원의 현재 제출 행동이나 숨겨진 적 상태를 읽어 mask를 만들지 않는다. 벽 공격을 mask로 금지할지, 실제 공격 시도 후 실패로 세어 cooldown을 부여할지는 7절에서 먼저 정한다.

mask로 금지되어 NOOP로 정규화된 요청과, **유효하지만 실패한 공격 시도**를 구분한다. 후자에는 기존 게임 규칙대로 cooldown을 준다. 제출 action·적용 action·거부 이유를 이벤트에 따로 남긴다.

행동 선택과 PPO update에서 동일한 당시 mask로 log-probability를 계산해야 한다. rollout에 mask를 저장한다. dead agent의 NOOP-only 분포는 log-probability와 entropy가 0이며, 생존 여부를 termination mask로 사용하지 않는다. value target은 부활 이후까지 이어진다. 정책손실/entropy 집계에서 대기 샘플을 제외할지는 명시적으로 고정하고, value/return 연결은 유지한다.

부분 관측 자체는 의도된 설계다. 시간·cooldown을 관측에 넣어도 local 관측이 완전한 Markov state가 되는 것은 아니다. 중앙 `state()`에는 전체 타일·모든 위치/생존·cooldown·부활 대기·틱 등 전이 예측에 필요한 상태를 포함하고, 기본 IPPO 관측과 섞지 않는다.

## 9. 이벤트, 보상, 기록

`StepResult`에는 `tick`, `events`, `area_before`, `area_after`, `rewards_by_agent`, `reward_breakdown`, `terminated`, `truncated`를 둔다. 사건 자료는 보상 모드와 무관하게 동일하게 생성한다.

| 사건 | 최소 필드 | 용도 |
| --- | --- | --- |
| 타일 소유권 변화 | 좌표, 이전/이후 팀, 승리 팀 공격자 목록 | 개인 credit과 팀 gain/loss |
| 공격 판정 | 공격자, 목표, 유효 시도 여부, 팀별 수, 동률/변경 없음 등 결과 | cooldown, 중복·방어 분석 |
| 이동 판정 | ID, 출발/목적지, 성공, 실패 이유 | 충돌·교착·혼잡도 분석 |
| 사망 | ID, 위치, 공격 후/이동 후 구분, 부활 예정 틱 | 생존 및 대기 지표 |
| 부활 | ID, 새 위치, 예정/실제 시각, 지연 이유 | 후보 부족·복귀 시간 |

공동 획득 credit을 균등 분할하는 안에서는 한 타일의 credit 합이 정확히 1이어야 한다. `RewardPolicy(before, after, events)`는 같은 사건에서 선택한 목적의 보상을 산출한다. team 보상은 구성원 3명에게 동일하게 주고, 이미 `/3`한 팀 평균 신호를 다시 나누지 않는다. raw gain/loss와 실제 학습 보상 계수는 별도로 저장한다. 보상식 선택의 해석상 차이는 검토 문서 3절을 따른다.

환경 난수와 policy/opponent 난수를 분리한다. `reset(seed)` 후 같은 행동열은 부활 위치까지 재현되어야 한다. 후보 타일·에이전트 목록을 정렬한 후 추출하며 Python set/dict의 우연한 순회 순서에 난수 소비를 맡기지 않는다. 후보보다 부활 대기자가 많으면 오래 기다린 순서를 우선하고, 같은 시각의 동률은 seeded 추첨으로 정하는 안을 권고한다.

보상 모드·관측 모드만 바꾸고 같은 행동열을 재생했을 때 게임 상태와 RNG 진행이 같아야 한다. 렌더링·지표 계산도 RNG를 소비하지 않는다. 학습 시 환경 여러 개를 배치한다면 각 worker의 seed를 구별한다.

최소 실행 메타데이터는 규칙/관측/보상 버전, 전체 config, master 및 환경/상대/학습 seed, 코드 버전, 라이브러리 버전, 상대 정책 버전, 환경 틱 수, 학습 에이전트 transition 수다. 처음에는 config+seed+행동열로 replay하며 필요할 때만 RNG 상태를 포함한 checkpoint를 추가한다.

사람 입력의 wall-clock deadline은 manager/UI의 책임이다. core step은 입력·네트워크·sleep을 기다리지 않는다. 늦은 입력을 다음 틱에 자동 적용할지 버릴지, 세 사용자의 입력 경로와 관측 범위는 데모 구현 전에 정한다.

## 10. 검증과 완료 기준

| 범주 | 핵심 검증 |
| --- | --- |
| 공격 | 동률 유지, 우세 팀 점령, 아군색 공격, 공동 credit 합, 공격 다음 틱 금지 및 그다음 해제 |
| 이동 | 같은 목적지 전원 실패, swap 실패, 빈 칸으로 끝나는 연쇄 성공, 막힌 연쇄 실패, 4명 순환 |
| 단계 간 상호작용 | 공격 후 죽은 이동자의 취소/점유 해제, 동시 사망 공격자의 공격 유지, 같은 틱 칠한 목적지 이동 |
| 부활 | d=0 및 d>0의 첫 행동 틱, 중복 없는 위치, 후보 부족/0, 영역 0과 전멸 구분, 종료 틱 처리 |
| 불변식 | 매 틱 `A_0+A_1=M`, 생존자는 자기 팀 타일에 위치, 생존 위치 유일, 사망자 위치 없음 |
| 불변성·재현 | 관측 수정이 상태에 영향 없음, reset 반복/두 인스턴스 일치, action 사전·session 제출 순서 불변 |
| 정보 경계 | local의 추가 팀원 feature 0, 숨겨진 상태만 바꿔도 허용된 local 관측/mask 불변, 현재 제출 행동 비노출 |
| 실험 통제 | 동일 action trace의 보상/관측 모드 변경이 전이를 바꾸지 않음, RGB 표시 유무에 따른 전이 동일 |
| API | `parallel_api_test`, `parallel_seed_test`, 반환값 shape/dtype/space, 사망 대기 NOOP, 마지막 6명 보상/관측 |
| 시간·학습 | 정확히 T번 전이, 내재적 termination과 외부 truncation의 bootstrap 차이, 부활 전후 return 연결 |
| 규칙 유용성 | 방어/확장 딜레마, 필요한 집중 공격과 낭비 공격 구별, 이동 혼잡, 고의 사망의 이득 측정 |

공식 API 검사는 [PettingZoo 환경 검사 문서](https://pettingzoo.farama.org/content/environment_tests/)를 기준으로 한다. mask 밖 행동도 space 안에 있을 수 있으므로 테스트용 임의 sampling의 처리와 학습의 mask 사용을 함께 검증한다. 종료 의미는 [Gymnasium 시간 제한 문서](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)를 따른다.

성능은 headless 랜덤 rollout으로 환경 steps/s, 관측 생성 시간, 렌더링 제외 시간을 측정한다. 측정 없이 목표 수치를 보장하지 않는다. 먼저 NumPy 기반 구현과 6명 루프로 진행하고 병목이 확인된 부분만 최적화한다.

첫 구현의 완료 기준은 ‘화면이 움직임’이 아니라 **동일 trace가 재현되고, 규칙 테스트와 API 계약을 만족하며, 네 실험 조건에서 작은 학습 rollout을 생성할 수 있음**이다. 실제 학습 우열과 협력 효과는 그 뒤 실험으로 판단한다.
