# ADR: Agentic RAG Architecture (LangGraph × Chainlit × Azure OpenAI GPT-5)

---

## 1) 배경(Context)

사내 지식과 웹 정보를 결합해 **정확하고 근거가 있는 답변**을 생성해야 합니다. 단순 RAG는 단일 패스로 검색-생성을 수행하여 복잡한 질의와 최신성, 신뢰성 요구를 만족하기 어렵습니다. 이에 따라 **LangGraph 기반의 에이전트형(Agentic) RAG**를 적용하여 계획 수립, 도구 호출(웹검색·DB·Vector), 자기점검/크리틱, HITL(Human-in-the-Loop) 승인 루프를 포함하는 **반복적 프로세스**로 고도화합니다. UI는 **Chainlit**으로 제공하여 멀티턴 대화, 승인·수정, 근거 열람을 지원합니다. LLM은 **Azure OpenAI의 GPT-5** 배포를 사용합니다.

---

## 2) 요구사항(Requirements)

1. 요청 정보에 대해 **인터넷 검색** 수행
2. **Chatting Web UI** 제공(**Chainlit** 사용)
3. **멀티턴 대화** 지원
4. **HITL**(인간 검토/승인/수정) 포함
5. **LangGraph** 프레임워크 사용
6. **LLM: Azure OpenAI GPT-5** 사용(사내 구독/리소스)
7. **모듈/기능 단위 디렉토리 분리**로 유지보수성 확보
8. **자세한 README** 제공(설치/실행/아키텍처/운영 가이드)

### **성공 지표(예시)**

* 정답성/근거성 평가 점수(예: Faithfulness, Answer Relevance) ≥ 기준치
* 웹 최신성 충족률(최근 X일 내 문헌 인용 비율)
* HITL 개입률/승인시간 감소 추이(학습/튜닝 전후)
* 평균 응답 지연(latency) ≤ SLA, 비용/100회 호출 ≤ 예산

---

## 3) 결정(Decision)

* **에이전트 실행 엔진**: LangGraph 상태기계로 Planner, Retriever(Web & Vector), Evidence Grader, Synthesizer, Critic, HITL Gate, Responder 루프를 구성
* **웹검색**: Azure Bing Web Search API(또는 사내 프록시)로 1차 소스 수집, 규범 준수(robots.txt, rate limit), 중복 제거 및 신뢰도 점수화
* **벡터 검색**: Azure AI Search(또는 pgvector/FAISS)로 사내 문서 임베딩 인덱스 운영
* **LLM**: Azure OpenAI GPT-5(툴콜/함수호출 활성화, 시스템/도구 프롬프트 분리), 안전정책 필터
* **HITL**: 신뢰도 임계치 미달·고위험 주제·규제 도메인 시 **Chainlit 승인 패널**로 라우팅(수정/승인/반려)
* **메모리**: LangGraph Checkpointer(예: Redis/SQLite) + 요약 메모리(대화 압축) + 장기 벡터 메모리(사용자 컨텍스트)
* **관측성/평가**: 프롬프트/도구 호출 로그, 근거 추적, 오프라인/온라인 RAG 평가 파이프라인 도입
* **구조화**: 기능/모듈별 디렉토리 분리, IaC/CI 포함 배포 전략(Azure Container Apps 또는 App Service)

---

## 4) 아키텍처 개요(Architecture Overview)

```mermaid
flowchart LR
  U[사용자/오퍼레이터] -->|질의/승인| CL[Chainlit Web UI]
  CL -->|대화 메시지| ORC[LangGraph Orchestrator]
  ORC --> PL[Planner]
  PL --> RT{도구 라우팅}
  RT -->|웹| WS[Web Search Tool]
  RT -->|벡터| VS[Vector Retriever]
  WS --> EG[Evidence Grader]
  VS --> EG
  EG -->|근거/점수| SY[Synthesizer]
  SY --> CR[Critic(Self-check)]
  CR -->|저신뢰/고위험| HG[HITL Gate]
  HG -->|승인/수정| CL
  CR -->|OK| RESP[Final Responder]
  RESP --> CL
  ORC <-->|상태/체크포인트| CP[(Memory Store)]
```

### **주요 데이터 흐름**

1. Planner가 하위 과제/도구 선택 → 2) Web & Vector 동시 검색 → 3) 근거 스코어링/중복 제거 → 4) 초안 생성 → 5) Critic이 사실성/근거성 점검 → 6) 신뢰도 임계치 미달 시 HITL → 7) 승인 후 응답.

---

## 5) 컴포넌트 설계(상세)

### 5.1 Orchestrator & State (LangGraph)

* **상태 스키마**: `{ messages, plan, queries, evidences, citations, confidence, risk_label, human_feedback, halt }`
* **Checkpointer**: Redis 또는 SQLite; 세션 키는 사용자ID+대화ID
* **조기 종료/최대 스텝**: `max_steps`, `early_stop_on_confidence≥τ`

### 5.2 Planner

* 질의 분해/서브쿼리 생성, 도구 라우팅 정책(웹/벡터/둘다), 쿼리 리라이팅

### 5.3 Web Search Tool

* Azure Bing Web Search API(또는 사내 검색 게이트웨이)
* **전처리**: DOM 본문 추출, 언어 판별/번역, 중복 제거, 출처 신뢰도/신선도 점수화
* **캐시**: URL→요약 캐시, 재호출 절감

### 5.4 Vector Retriever

* 인덱스: Azure AI Search(또는 pgvector)
* 전략: Hybrid BM25 + Dense, Passage 수준 인덱싱, 필터링(부서/권한)

### 5.5 Evidence Grader

* 관련성/신뢰도/중복/모순 평가(LlM-비용 절감을 위해 규칙+작은 모델 혼합)
* K 문서로 압축, 인용 메타데이터 표준화

### 5.6 Synthesizer

* 구조화 템플릿(답변/근거/한줄요약/다음액션), 인용각주 생성

### 5.7 Critic(Self-check)

* 사실 검증 프롬프트, 금칙/규제 주제 감지, 신뢰도 산출

### 5.8 HITL Gate

* 임계치 `τ_conf`, 리스크 라벨(예: 법률/의료/재무) 기준 라우팅
* Chainlit UI에서 **Approve / Edit / Reject / Request More Evidence** 액션 제공
* 모든 상호작용 감사 로그 보관

### 5.9 Memory

* **요약 메모리**: 멀티턴 맥락 유지, 토큰 절감
* **장기 메모리**: 사용자 프로필/환경/선호를 벡터화 저장(권한 모델 적용)

### 5.10 Observability & Eval

* 구조화 로그(도구 호출, 선택 근거, 결론), 지표 대시보드
* 오프라인 세트로 RAGasA(Eval) 주기 실행; 온라인 사용자 피드백 수집

---

## 6) 성능·비용 전략

* 동적 K선정, 랭크-프루닝, 요약 캐시, 도구 호출 상한
* 비동기/병렬 검색, 스트리밍 응답, 토큰 예산 관리(프롬프트 압축)

---

## 7) 디렉토리 구조(모듈 분리)

```text
├─ src/
│  ├─ rag_agent/
│  │  ├─ graphs/
│  │  │  └─ agent_graph.py
│  │  ├─ agents/
│  │  │  ├─ planner.py
│  │  │  ├─ synthesizer.py
│  │  │  └─ critic.py
│  │  ├─ retrievers/
│  │  │  ├─ web_search.py
│  │  │  └─ vector_store.py
│  │  ├─ graders/
│  │  │  └─ evidence_grader.py
│  │  ├─ hitl/
│  │  │  └─ review_gate.py
│  │  ├─ memory/
│  │  │  ├─ checkpointer.py
│  │  │  └─ conversation_summarizer.py
│  ├─ ui/
│  │  └─ app_chainlit.py
│  └─ main.py
├─ tests/
├─ .env.example
├─ pyproject.toml
├─ README.md
├─ ADRs.md
```

---

## 8) README.md 구성 지침(상세)

1. 프로젝트 개요(문제 정의, Agentic RAG 소개)
2. 빠른 시작(로컬/도커, 환경변수, 최초 인덱싱)
3. 아키텍처(본 ADR 다이어그램, 데이터 흐름)
4. LangGraph 노드/상태 설명, 시퀀스 예시
5. 웹검색/벡터검색 설정(Azure/Bing/AI Search)
6. Chainlit 사용법(액션 버튼, HITL 승인/수정 플로우)
