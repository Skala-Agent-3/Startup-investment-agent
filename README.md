# AI Startup Investment Evaluation Agent

> 본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.


## 👀 Overview

- Objective : 헬스케어 스타트업 후보 중 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석
- Method : AI Agent + Agentic RAG
- Output: 보고서 

## ⭐ Features

- 헬스케어 기업 분석 보고서(PDF 파일) 기반 정보 추출
- 헬스케어 산업 최근 동향, 후보 스타트업 정밀 탐색 정보, 투자 추천 기업 요약 정보, 투자 판단 및 근거를 담은 보고서 작성
- 최종 보고서는 PDF로 다운로드 가능

## 💻 Tech Stack 

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini(OpenAI API), multilingual-e5-base(Huggingface) |
| Retrieval  | FAISS           |

## ⛓️ Agents
| Agent            | 설명                                                                             | 입력              | 출력                                   |
| ---------------- | ------------------------------------------------------------------------------ | --------------- | ------------------------------------ |
| **오케스트레이터 에이전트** | 사용자의 질문을 받아 전체 Workflow를 조율. 기업명 추출, 각 Agent 실행 순서 관리, StateGraph를 통해 상태 전이 관리 | 사용자 입력 (질문)     | 검색할 회사명    |
| **기술 요약 에이전트**   | PDF 기반 기업별 기술·제품·BM 요약                                                         | 기업명/문서 컨텍스트     | 기업별 기술 요약 텍스트                        |
| **시장성 평가 에이전트**  | 산업 리포트·RAG 검색 결과로 시장 규모/트렌드 평가                                                 | 산업명/시장 키워드      | 시장 개요, 트렌드, 리스크                 |
| **경쟁사 분석 에이전트**  | 기업 요약 정보를 받아 비교 분석 (SWOT, KSF, 5 forces)                                       | 각 기업 요약         | 경쟁사 비교 보고서                       |
| **투자 판단 에이전트**   | 시장성 + 경쟁사 분석 결과 종합 → 투자 여부 결정                                                  | 시장성 평가 + 경쟁사 분석 | 투자 여부 (투자/보류/고위험) 및 사유               |
| **보고서 작성 에이전트**  | 모든 결과를 통합 → 최종 PDF 보고서 작성                                                      | 앞선 모든 Agent 결과  | PDF 보고서 (기업 리스트, 시장성, 경쟁사 분석, 최종 결론) |



## 🧩 Architecture
<img width="778" height="1246" alt="Image" src="https://github.com/user-attachments/assets/99c41f80-8bf7-46ab-b678-36cafaf040c0" />

## 🔢 Evaluation
| 섹션           | 만점 | 문항 수 | 
| ------------ | -: | ---: | 
| Owner        | 30 |    6 |
| Market       | 25 |    6 |   
| Product/Tech | 15 |    6 |  
| Moat         | 10 |    5 |  
| Traction     | 10 |    5 | 
| Deal Terms   | 10 |    6 |


## 📂 Directory Structure
- **data/**: 분석할 스타트업 PDF 문서를 저장합니다.  
- **agents/**: 평가 기준(시장성, 기술력 등)에 따라 동작하는 개별 Agent 모듈이 위치합니다.  
- **prompts/**: LLM 실행 시 사용하는 프롬프트 템플릿을 관리합니다.  
- **outputs/**: 평가 실행 후 생성된 결과 파일이 저장됩니다.  
- **main.py**: 프로젝트 실행의 진입점 스크립트입니다.  
- **README.md**: 프로젝트 소개 및 실행 가이드를 담고 있습니다.  

## 👥 Contributors 
- 김정윤 : Prompt Engineering, 보고서 생성 에이전트
- 김채연 : 경쟁사 분석 에이전트, 서비스 구조 도식화
- 신동연 : Prompt Engineering, 시장성 평가 에이전트, 기술 요약 에이전트(State 출력, 임베딩 수정)
- 원주혜 : 오케스트레이터 에이전트, Langgraph 기반 에이전트 통합
- 전혜민 : 투자 판단 에이전트, RAG용 raw data 구득 
- 정성희 : Prompt Engineering, 기술요약 에이전트
