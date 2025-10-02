# AI Startup Investment Evaluation Agent

> 본 프로젝트는 헬스케어 스타트업에 대한 투자 가능성을 자동으로 평가하는 AI 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview

이 저장소는 **LangGraph 기반 에이전틱(Agentic) RAG 워크플로우**를 통해 헬스케어 스타트업의 투자 적합성을 평가합니다.

- 투자 브리핑 PDF를 로드하여 핵심 기업 정보 추출
- 실시간 시장 조사 결과를 결합하여 분석 강화
- 멀티 에이전트 협업(기술 분석 → 시장 평가 → 경쟁사 분석 → 투자 자문) 수행
- 최종 결과를 투자 판단 보고서로 정리 후 PDF로 출력

---

## Features

- **Document Loader & Text Splitter**

  - Parent–Child Chunk 구조: 페이지 단위(Parent)와 세부 단위(Child) 분할
  - RecursiveCharacterTextSplitter로 토큰 단위 맥락 보존

- **Embedding & Vector Store**

  - HuggingFace 임베딩 모델 기반의 Dense Embedding 생성
  - FAISS Vector Store 구축

- **Hybrid Retriever**

  - Dense Retrieval (임베딩 기반 유사도 검색)
  - Sparse Retrieval (BM25 키워드 기반 검색)
  - 두 결과를 가중치 기반으로 통합하여 포괄성과 정밀성 동시 확보

- **Reranker**

  - Cross-Encoder 모델로 재정렬 수행
  - Hybrid Retrieval 결과 중 의미적으로 가장 관련성이 높은 문서만 상위에 배치

- **MMR (Maximal Marginal Relevance)**

  - 중복 문서를 줄이고 다양성을 확보
  - λ 가중치를 활용해 관련성과 다양성 균형 조정

- **Citation-aware Response**

  - 문서의 출처·페이지 정보를 메타데이터에 포함
  - 모든 응답이 신뢰 가능한 인용 근거를 포함

- **Agentic Workflow Integration**
  - (QueryParser → TechScribe → MarketEvaluator → CompetitorAnalyzer → InvestmentAdvisor → ReportGenerator)
  - Retrieval·Reranking 전략으로 확보한 맥락을 기반으로 에이전트별 전문 분석 수행

---

## Tech Stack

| 범주              | 세부 사항                                      |
| ----------------- | ---------------------------------------------- |
| 언어 및 런타임    | Python 3                                       |
| 오케스트레이션    | LangGraph (State Machine), LangChain runnables |
| LLM               | OpenAI `gpt-4o-mini`                           |
| 임베딩 모델       | HuggingFace MiniLM                             |
| 벡터 데이터베이스 | FAISS                                          |

---

## Agents

| Agents                                    | 설명                                                                                                             |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| QueryParser(쿼리 파서 에이전트)           | 질의에서 관련 기업명을 감지하고 그래프 상태를 초기화                                                             |
| TechScribe (기술 요약 에이전트)           | 9개 분석 항목별로 출처 인용이 보장된 RAG 기반 심층 분석 수행, 헬스케어 기업 보고서로부터 후보 스타트업 정보 추출 |
| MarketEvaluator (시장성 평가 에이전트)    | 실시간 웹 검색 및 시장 동향 요약, 헬스케어 산업의 최근 트렌드 반영                                               |
| CompetitorAnalyzer (경쟁사 분석 에이전트) | TechScribe 결과를 바탕으로 경쟁 구도 분석 (SWOT, KSF, Porter’s 5 Forces 등)                                      |
| InvestmentAdvisor (투자 판단 에이전트)    | 경쟁사 분석 및 시장 평가 결과를 종합하여 각 스타트업의 투자 여부와 근거 제시                                     |
| ReportGenerator (보고서 작성 에이전트)    | 모든 에이전트의 결과를 취합하여 최종 마크다운 보고서 작성 및 PDF 변환                                            |

---

## Architecture

<img width="480" height="704" alt="Image" src="https://github.com/user-attachments/assets/bf298d34-a8c5-47e9-ab40-46209e400fe8" />

---

## Evaluation

### 평가 원칙

- 각 섹션은 비중에 따라 배점이 달라집니다.
- 섹션은 **체크리스트 문항**으로 구성되며, 각 문항은 **True/False**로만 평가합니다.
- 섹션 점수 = (True 개수 ÷ 문항 수) × 섹션 만점
- 섹션 점수는 **소수 1자리 반올림**으로 표기합니다.
- 총점 = 모든 섹션 점수의 합 → **소수 1자리 반올림**으로 표기합니다.

### 섹션별 배점표

| 섹션 (한글/영문)                 | 만점 | 문항 수 |
| -------------------------------- | ---: | ------: |
| 창업자 (Owner)                   |   30 |       6 |
| 시장성 (Opportunity Size)        |   25 |       6 |
| 제품/기술력 (Product/Technology) |   15 |       6 |
| 경쟁 우위 (Moat)                 |   10 |       5 |
| 실적 (Traction)                  |   10 |       5 |
| 투자 조건 (Deal Terms)           |   10 |       6 |

### 의사결정 기준

- 기본 임계값: **70점** (`DECISION_THRESHOLD_100 = 70`)
- **총점 ≥ 70.0** → _“투자”_
- **총점 < 70.0** → _“보류”_

---

## Directory Structure

```plaintext
.
├── agents/         # 에이전트 모듈
│   ├── CompetitorAnalyzer.py
│   ├── InvestmentAdvisor.py
│   ├── MarketEvaluator.py
│   ├── ReportGenerator.py
│   ├── TechScribe.py
│   ├── QueryParser.py
├── data/           # 투자 브리핑 PDF
├── outputs/        # 결과물 저장
│   └── create_pdf.py
├── prompts/        # 프롬프트 템플릿
│   ├── competitor_analyzer.py
│   ├── investment_advisor.py
│   ├── market_evaluator.py
│   ├── report_generator.py
│   ├── techscribe.py
│   ├── queryparser.py
├── rag/            # RAG 관련 모듈
│   └── advanced_retriever.py
├── graph_state.py
├── main.py
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```

---

## 시작하기

1. `pip install -r requirements.txt`로 의존성 설치
2. [wkhtmltopdf 공식 다운로드 페이지](https://wkhtmltopdf.org/downloads.html)에서 자신의 운영체제(OS)에 맞는 설치 파일 다운로드 및 설치
3. `.env`에 OpenAI 자격 증명(필요 시 HuggingFace / DuckDuckGo 키 포함) 입력
4. `data/` 폴더에 투자 브리핑 PDF를 넣고 `python main.py` 실행
5. 결과 PDF는 `outputs/final_report.pdf`에서 확인 가능

---

## Contributors

- 김정윤 : 보고서 생성 에이전트, 보고서 PDF 변환
- 김채연 : 경쟁사 분석 에이전트, 서비스 구조 도식화
- 신동연 : 시장성 평가 에이전트, 기술 요약 에이전트(State 출력, 임베딩 수정)
- 원주혜 : Langgraph 기반 에이전트 통합, 쿼리 파서 에이전트
- 전혜민 : 투자 판단 에이전트, RAG용 raw data 구득, RAG 전략 고도화
- 정성희 : 기술요약 에이전트
