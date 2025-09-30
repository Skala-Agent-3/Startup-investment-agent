# AI Startup Investment Evaluation Agent

> 본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.


## Overview

- Objective : 헬스케어 스타트업 후보 중 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석
- Method : AI Agent + Agentic RAG

## Features

- 헬스케어 기업 분석 보고서(PDF 파일) 기반 정보 추출
- 헬스케어 산업 최근 동향, 후보 스타트업 정밀 탐색 정보, 투자 추천 기업 요약 정보, 투자 판단 및 근거를 담은 보고서 작성
- 최종 보고서는 PDF로 다운로드 가능

## Tech Stack 

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini(OpenAI API), multilingual-e5-base(Huggingface) |
| Retrieval  | FAISS           |

## Agents
| Agents   | 설명                     |
|------------|------------------------------|
| 기술 요약 에이전트 | LangGraph, LangChain, Python |
| 시장성 평가 에이전트        | GPT-4o-mini(OpenAI API), multilingual-e5-base(Huggingface) |
| Retrieval  | FAISS           |


- 기술 요약 에이전트
- 시장성 평가 에이전트
- 경쟁사 분석 에이전트
- 투자 판단 에이전트
- 보고서 작성 에이전트

## Architecture
<img width="300" height="500" alt="Image" src="https://github.com/user-attachments/assets/b885eb88-c1eb-461e-8cd2-4d29e151d4c5" />

## 📂 Directory Structure
- **data/**: 분석할 스타트업 PDF 문서를 저장합니다.  
- **agents/**: 평가 기준(시장성, 기술력 등)에 따라 동작하는 개별 Agent 모듈이 위치합니다.  
- **prompts/**: LLM 실행 시 사용하는 프롬프트 템플릿을 관리합니다.  
- **outputs/**: 평가 실행 후 생성된 결과 파일이 저장됩니다.  
- **main.py**: 프로젝트 실행의 진입점 스크립트입니다.  
- **README.md**: 프로젝트 소개 및 실행 가이드를 담고 있습니다.  

## Contributors 
- 김정윤 : Prompt Engineering, 보고서 생성 에이전트
- 김채연 : 경쟁사 분석 에이전트, 서비스 구조 도식화
- 신동연 : Prompt Engineering, 시장성 평가 에이전트
- 원주혜 : Langgraph 기반 에이전트 통합
- 전혜민 : 투자 판단 에이전트, RAG용 raw data 구득 
- 정성희 : Prompt Engineering, 기술요약 에이전트
