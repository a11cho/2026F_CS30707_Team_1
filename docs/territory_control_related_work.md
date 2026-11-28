# Territory Control 선행 연구

2026년 11월 29일 기준으로, 3대3 영역 점령에서 팀원 행동 공개가 협력과 경기 성과에 미치는 영향을 다룬다. 같은 틱의 확정 행동을 공유하는 순차 결정에는 **SeqComm**, 행동 제안과 송신자의 최종 행동을 연결하는 학습에는 **MCS**, 제안이 수신자의 선택에 미치는 영향을 검증하는 데에는 **Lowe 등의 통신 평가**가 직접적인 비교 근거다. [Ding et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d6be51e667e0b263e89a23294b57f8cf-Abstract-Conference.html) · [Zhu et al., RLJ 2026](https://rlj.org/papers/zhu2026learning) · [Lowe et al., AAMAS 2019](https://arxiv.org/abs/1903.05168)

## 팀원 행동을 공개하는 시점과 내용

환경의 `independent`에서는 팀원 행동을 받지 않는다. `sequential`에서는 앞 순번이 확정한 행동을 보고 선택한다. `feedback`에서는 세 팀원이 동시에 1차 제안을 제출한 뒤 동결된 제안을 보고 각자 최종 행동을 정한다. 세 조건 모두 최종 제출을 모아 한 번의 환경 전이를 실행한다. 15×15 자기 중심 관측은 기본 8×8 보드를 모두 담는다. 따라서 같은 공간 관측에서 **이번 틱에 팀원이 무엇을 하려는가**에 관한 정보가 달라진다. [환경의 관측 및 행동 공개 규칙](../env.md)

| 선행 연구 | 방법과 프로젝트의 접점 |
| --- | --- |
| Ding et al., **SeqComm**, NeurIPS 2024 | 모델이 예측한 의도의 가치로 매 스텝 결정 우선순위를 정한다. 앞 순번의 확정 행동은 뒤 순번에게 전달하고, 선택된 행동은 환경에서 동시에 실행한다. 우리 순차 환경은 슬롯 순서로 제출하므로 앞선 확정 행동의 정보 효과를 비교할 수 있다. [논문 §3–4](https://proceedings.neurips.cc/paper_files/paper/2024/file/d6be51e667e0b263e89a23294b57f8cf-Paper-Conference.pdf) |
| Kim, Park & Sung, **Intention Sharing**, ICLR 2021 | 학습한 환경·동료 모델로 미래 궤적을 상상하고, 이를 압축해 의도 메시지로 공유한다. Feedback PPO의 메시지는 같은 틱에서 선택한 9개 행동 중 하나이므로, 성능에 영향을 줄 수 있는 시간 범위와 메시지 내용이 구체적으로 다르다. [논문 및 발표](https://iclr.cc/virtual/2021/poster/3363) |
| Zhu, Dastani & Wang, **MCS**, RLJ 2026 | 메시지로 송신자의 실행 행동을 예측하는 보조 손실을 둔다. 원 연구는 여러 과제의 관측을 공통 메시지 공간에 인코딩한다. Feedback PPO는 이 예측 목적을 1차 이산 행동 제안과 같은 송신자의 2차 최종 행동에 적용한다. [논문 §4.1–4.3](https://rlj.cs.umass.edu/2026/papers/Paper130.pdf) · [프로젝트의 예측 손실](../ppo/README.md#mcs-송신자-행동-예측-손실) |
| Foerster et al., **DIAL**, NeurIPS 2016 | 중앙 학습에서 연속 메시지로 수신자에서 송신자까지 경사를 전달하고 실행에서 메시지를 이산화한다. Feedback PPO는 순전파에서 수집한 이산 제안을 쓰고, straight-through 추정으로 수신자 PPO 손실의 경사를 전달한다. 송신자의 최종 행동 예측 손실과 이 경사 경로를 각각 제어할 수 있다. [DIAL 논문](https://arxiv.org/abs/1605.06676) · [프로젝트의 역전파 조건](feedback_ppo_communication.md) |
| Lowe et al., **On the Pitfalls of Measuring Emergent Communication**, AAMAS 2019 | 메시지가 송신자의 행동을 알려 주는지(positive signaling)와 수신자의 선택을 실제로 바꾸는지(positive listening)를 구분한다. 같은 체크포인트와 보드에서 팀원 제안 공개만 바꾸는 평가가 수신 효과에 해당한다. [논문](https://arxiv.org/abs/1903.05168) |

Feedback PPO에서 수신자 손실의 송신자 역전파를 끈 모델은 같은 3,274건의 엄호 기회에서 제안을 가리면 449건, 공개하면 517건에 성공했다. 이 재실행은 보드, 상대 행동, 1차 제안을 고정한 한 틱의 효과를 잰다. 384경기씩 평가했을 때 같은 모델의 경기 점수는 공개 시 24.61%, 가림 시 22.79%였고, 처음부터 가림 상태로 학습한 모델은 26.95%였다. 전체 경기는 정책에 따라 달라지는 상태 방문과 기회 수까지 반영한다. [Feedback PPO 실험과 한계](feedback_ppo_communication.md)

## 1대1 정책 전이와 교사 모방

SL-RL의 3×10 보드·64틱 실험에서 raw 관측을 사용한 3대3 직접 학습 팀은 1대1 정책 복제 팀에 358승·33무·89패를 기록했다. 9칸을 가리고 복원한 16차원 표현에서는 직접 학습 팀이 188승·50무·242패였다. 복제 팀을 상대 난이도 기준으로 쓸 때는 관측 표현과 대진을 함께 명시해야 하는 이유다. 같은 저장소의 계획은 raw 3대3 독립 팀을 순차 모델과 비교하고, 같은 순차 체크포인트에서 앞선 팀원 행동을 가려 실제 정보 이용도 측정하는 것이다. [SL-RL 전이 결과와 순차 가설](../SL-RL/SL_RL_RESEARCH_SUMMARY.md)

| 선행 연구 | 방법과 프로젝트의 접점 |
| --- | --- |
| Agarwal, Kumar, Sycara & Lewis, **Learning Transferable Cooperative Behavior in Multi-Agent Team**, AAMAS 2020 확장 초록 | 에이전트와 환경 개체의 그래프에서 메시지를 주고받으며 팀 크기가 달라지는 상황을 다룬다. 팀원 수에 맞춰 입력 표현을 설계하는 계열로, raw·압축 표현에 따라 정책 전이가 달랐던 SL-RL 결과와 연결된다. [원문](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p1741.pdf) |
| Liu et al., **MaskMA**, arXiv 2024 개정본 | SMAC에서 일부 유닛을 가리고 보이는 유닛들의 정책을 공동으로 학습한다. SL-RL은 지도 타일을 가려 복원하고 인코더를 고정한 뒤 PPO를 학습한다. 가리는 단위와 사전학습 목표를 명시하면 두 결과의 비교 범위를 설명할 수 있다. [프리프린트](https://arxiv.org/abs/2310.11846) |
| Abboud & Gal, **MATES**, arXiv 2026 | 단일 에이전트 정책을 고정하고, 팀원 정보를 받아 그 정책의 입력으로 바꾸는 어댑터를 팀 경험으로 학습한다. 1대1 정책을 그대로 복제하는 기준에 팀 맥락만 추가해 비교할 때 가까운 구조다. [프리프린트](https://arxiv.org/abs/2609.26010) |
| Takahagi, Motokawa & Sugawara, **Post-Training Policy Replication**, arXiv 2026 | 물체 수집에서 4인 팀 정책을 3~8인 팀에 복제한다. 팀 크기와 무관한 집계 관측을 사용하고, 무학습 복제와 초기 가중치를 재사용한 추가 학습을 평가한다. 우리 실험은 경쟁이 있는 1대1에서 3대3으로 전이해, 팀 크기뿐 아니라 아군 간 조율 요구도 바뀐다. [프리프린트](https://arxiv.org/abs/2611.34035) |
| Ross, Gordon & Bagnell, **DAgger**, AISTATS 2011 · Anthony, Tian & Barber, **Expert Iteration**, NeurIPS 2017 | DAgger는 학생이 방문하는 상태를 교사가 다시 라벨링하여 시범과 실행의 상태 분포가 달라지는 문제를 다룬다. Expert Iteration은 탐색의 선택을 신경망에 모방시키고, 그 정책으로 다시 탐색을 유도한다. SL-RL exp4는 고정 봇이 방문한 상태에서 시범을 수집한다. Qwen 정책의 탐색 모방은 학생이 방문한 상태에서 탐색 교사를 호출한다. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) · [Expert Iteration](https://arxiv.org/abs/1705.08439) · [탐색 교사 실험](centralized_team_policy.md#탐색-행동을-지도-신호로-사용한-대조) |

`expand`와 `aggressive`의 시범을 모방한 SL-RL exp4에서 교사 행동 일치율은 41.0%에서 67.6%로 올랐다. 같은 평가에서 두 상대에 대한 승률은 각각 5.42%에서 15.42%, 2.50%에서 15.83%로 올랐다. 행동 일치율은 시범 상태에서 측정하고, 경기 승률은 학생이 스스로 방문한 상태에서 측정한다. [SL-RL exp4](../SL-RL/SL_RL_RESEARCH_SUMMARY.md) · [DAgger](https://proceedings.mlr.press/v15/ross11a.html)

## 중앙 공동 정책과 언어 모델

| 선행 연구 | 방법과 프로젝트의 접점 |
| --- | --- |
| Yu et al., **MAPPO**, NeurIPS 2022 | 여러 협력 벤치마크에서 PPO를 강한 기준선으로 검증했다. 중앙 가치 함수를 쓰는 분산 행동 정책과, 한 정책이 729개 공동 행동 중 하나를 직접 고르는 프로젝트의 중앙 기준은 실행 시 정보 구조가 다르다. [MAPPO 논문](https://arxiv.org/abs/2103.01955) · [프로젝트 중앙 정책](centralized_team_policy.md) |
| Wen et al., **Multi-Agent Transformer**, NeurIPS 2022 | 여러 에이전트의 공동 행동을 자기회귀적으로 선택한다. 현재 Qwen 공동 정책은 9³개 조합을 한 번에 채점한다. MAT는 한 중앙 정책이 순서대로 슬롯의 행동을 고르는 공동 정책 비교에 가깝고, SeqComm은 별도의 에이전트가 확정 행동을 넘겨받는 순차 비교에 가깝다. [MAT 논문](https://arxiv.org/abs/2205.14953) · [SeqComm 논문](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d6be51e667e0b263e89a23294b57f8cf-Abstract-Conference.html) |
| Carta et al., **GLAM**, ICML 2023 | 텍스트로 표현한 BabyAI 과제에서 FLAN-T5를 정책으로 사용하고 환경 상호작용으로 온라인 PPO를 수행한다. 상태를 문자열로 표현해 사전학습 모델을 게임 정책에 연결하는 설계에 가깝다. 우리 정책은 텍스트 백본의 특징과 공간 CNN을 결합해 공동 행동 헤드를 사용한다. [GLAM 논문](https://proceedings.mlr.press/v202/carta23a.html) |
| Xu et al., **DipLLM**, ICML 2025 | 대화 없는 Diplomacy에서 한 세력의 여러 유닛 명령을 자기회귀적으로 분해하고, 수집한 게임 데이터에서 유닛 가치로 가중한 손실로 LLM을 튜닝한다. 경쟁과 협력이 공존하는 게임에서 공동 명령을 생성하는 사례이며, 보드 직렬화와 공동 행동 표현을 비교할 근거다. [DipLLM 논문](https://proceedings.mlr.press/v267/xu25b.html) |

별도의 슬롯별 Qwen 실험에서 통신 허용 모델은 학습 수집 중 4.09%의 선택에서 의도를 보냈다. 시드 47의 최종 argmax 평가에서는 승급전 16경기와 고정 보드 64개에서 모든 슬롯이 `SILENT`를 선택했다. 같은 가중치에서 수신을 가려도 행동과 결과가 같았다. 이 실행에서 관찰한 경기 차이는 통신 허용이라는 학습 조건에 해당하며, 평가에서 메시지가 기여했는지는 고정 가중치에서의 수신 대조가 보여준다. [Qwen 통신 평가](centralized_team_policy.md#실제-메시지-사용) · [Lowe et al., AAMAS 2019](https://arxiv.org/abs/1903.05168)

## 팀 대전과 협력 행동을 평가하는 벤치마크

| 선행 연구 | 평가 대상과 Territory Control의 접점 |
| --- | --- |
| Resnick et al., **Pommerman**, arXiv 2018 | 2대2 격자 대전의 Team Radio 변형에서는 팀원이 매 단계 8개 단어 중 두 개를 송신할 수 있다. 우리 통신의 기호는 이번 틱의 합법 행동에 대응한다. 제안 회차를 마감하면 모든 팀원이 같은 기호를 받아 최종 행동을 다시 고른다. [논문](https://arxiv.org/abs/1809.07124) · [공식 환경](https://github.com/MultiAgentLearning/playground) |
| Singh, **Territory Paint Wars**, arXiv 2026 | 두 에이전트가 동시에 행동하며 종료 시 타일 수로 승패를 판정한다. 자기대전 점수가 안정적이어도 고정 상대에 대한 성능은 악화될 수 있음을 보고했다. 우리 3대3 실험의 상대별 점수와 여러 고정 봇을 사용한 평가에 연결된다. [프리프린트](https://arxiv.org/abs/2604.04983) |
| Hawili et al., **MARL-BattleGrounds**, arXiv 2026 | 서로 다른 능력을 가진 5대5 팀을 대전시키고, 보지 못한 열세 초기 배치에서 5~10틱 안에 역전하는 전술 시험을 제공한다. Territory Control의 동일 상태 재실행은 한 틱의 상대 행동을 고정하고 제안 공개로 엄호나 이동 판정이 달라지는지 측정한다. [프리프린트](https://arxiv.org/abs/2610.14883) |
| Cardei, Landers & Doryab, **Coordination Matters**, arXiv 2026 | 팀 수익과 함께 간섭, 공동 과업 완료, 참여, 조율 효율을 분석한다. 수익이 비슷한 정책에서도 행동 패턴이 달랐다. 우리 실험은 실제로 엄호할 수 있었던 기회를 분모에 두고 같은 보드에서 수신 입력을 바꿔, 제안이 그 행동에 기여했는지도 확인한다. [프리프린트](https://arxiv.org/abs/2605.06557) |
| Papoudakis et al., **Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks**, NeurIPS 2021 | 여러 협력 과제에서 독립 학습, 중앙 학습 정책 경사, 가치 분해를 공통 평가 절차로 비교한다. 이 프로젝트에서 독립·순차·되먹임 정책을 비교할 때도 맵, 팀 보상, 상대, 환경 틱 예산, 평가 시드를 맞춰 어떤 조건의 차이인지 드러낼 수 있다. [논문](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/a8baa56554f96369ab93e4f3bb068c22-Paper-round1.pdf) |

## 경쟁 상대, 협력 지표, 보상

| 선행 연구 | 프로젝트에서 대응하는 항목 |
| --- | --- |
| Agapiou et al., **Melting Pot 2.0**, arXiv 2022 | Territory의 자원 소유와 점령 환경, 그리고 학습에서 보지 못한 배경 정책을 넣는 평가 방식이 관련된다. 우리 환경에서는 팀 소유 타일에 대한 공격 수를 집계하며 종료 시 영역으로 승패를 정한다. [벤치마크](https://arxiv.org/abs/2211.13746) · [Territory 구현](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/territory.py) |
| Ellis et al., **SMACv2**, NeurIPS 2023 · Jaderberg et al., **Capture the Flag**, Science 2019 | SMACv2는 기존에 경과 시간만으로도 승리하는 사례를 찾아, 미지의 절차 생성 시나리오에서 평가했다. Capture the Flag는 서로 경쟁하는 팀을 여러 지도·정책 조합에서 평가했다. 우리 고정 보드는 엄호나 안전한 진입이 가능한지를 재고, 여러 상대와의 대전은 정책이 그 기회를 만들고 승리까지 이어 가는지를 잰다. [SMACv2](https://proceedings.neurips.cc/paper_files/paper/2023/file/764c18ad230f9e7bf6a77ffc2312c55e-Paper-Datasets_and_Benchmarks.pdf) · [Capture the Flag](https://www.science.org/doi/10.1126/science.aau6249) |
| Foerster et al., **COMA**, AAAI 2018 | 팀 보상에서 개별 기여를 추정하기 위해 나머지 팀원의 행동을 고정하고 한 에이전트의 행동만 주변화하는 중앙 가치 기준을 쓴다. 우리 엔진에서 한 공격을 빼고 재실행하는 대조는 같은 반사실 원리로, 관측된 공동 공격 중 실제로 점령이나 방어 판정을 바꾼 사건을 정의할 수 있다. [COMA 논문](https://arxiv.org/abs/1705.08926) · [SL-RL 협력 지표](../SL-RL/SL_RL_RESEARCH_SUMMARY.md) |
| Ng, Harada & Russell, **Policy Invariance under Reward Transformations**, ICML 1999 | 같은 할인율로 `γΦ(s′) − Φ(s)`를 더하고 종료 상태의 잠재값을 0으로 두면, 보상 항의 할인합은 초기 상태에만 의존한다. SL-RL exp5가 이 방식을 쓴다. Feedback PPO의 매 틱 격차 항은 경기 중 앞서 있는 정도를 누적하므로, 종료 승패 외에 유리한 상태를 얼마나 오래 유지했는지도 보상한다. [정책 불변성 논문](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) · [SL-RL exp5 설정](../SL-RL/experiment/exp5/README.md) · [Feedback PPO 보상](feedback_ppo_communication.md) |
| R. Agarwal et al., **Deep Reinforcement Learning at the Edge of the Statistical Precipice**, NeurIPS 2021 | 학습 반복이 적으면 집계 점수의 추정 오차가 크다는 점을 보였다. 현재 Feedback PPO의 3개 학습 시드에서는 시드별 효과와 협력 기회의 분자·분모가 해석의 단위다. 같은 정책에서 나온 수천 개 틱에는 같은 학습 시드와 경기의 영향이 공유된다. [논문](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) · [Feedback PPO 시드별 결과](feedback_ppo_communication.md) |

보고서의 연구 질문은 **같은 보드를 보는 3대3 팀에서 이번 틱의 확정 행동 또는 제안을 공유하면, 어떤 협력 상황에서 결과가 달라지고 그 변화가 경기 성과로 이어지는가**로 잡을 수 있다. 순차 조건은 앞선 행동의 공개와 가림을, 되먹임 조건은 동결된 제안의 공개와 가림을 비교한다. 두 비교에서 한 틱의 엄호·진입·이동 양보 결과와 전체 경기의 종료 영역 및 승패를 함께 기록하면, 각 정보가 도움이 된 상황과 효과가 관찰된 범위를 설명할 수 있다. [순차 모델 실험 가설](../SL-RL/SL_RL_RESEARCH_SUMMARY.md) · [Feedback PPO의 동일 상태 재실행](feedback_ppo_communication.md)
