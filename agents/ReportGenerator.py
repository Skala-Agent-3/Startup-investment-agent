from dotenv import load_dotenv
from langchain_teddynote import logging
from typing import Annotated,  TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from prompts.report_generaor_prompt import REPORT_PROMPT

load_dotenv()
logging.langsmith("Startup-report-generator-agent")

class GraphState(TypedDict):
    question: Annotated[str, "Question"]                  # 질문
    search_context: Annotated[str, "Context"]             # 검색 결과 (예: Tavily, PDF 등)
    rag_context: Annotated[str, "Context"]                # RAG 결과
    company_list: Annotated[List[str], "Companies"]       # 추출된 기업 리스트
    TechScribe: Annotated[str, "Context"]                 # 기술 요약 결과
    MarketEvaluator: Annotated[str, "Context"]            # 시장성 평가 결과
    CompetitorAnalyzer: Annotated[str, "Context"]         # 경쟁사 비교 결과
    InvestmentAdvisor: Annotated[str, "Context"]          # 투자 판단 결과
    answer: Annotated[str, "Answer"]                      # 최종 답변 / 보고서
    chat_history: Annotated[list, "Messages"]             # 누적 대화 로그

def report_generator_agent(state: GraphState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = LLMChain(llm=llm, prompt=REPORT_PROMPT)
    
    # LLM에 제공할 데이터
    formatted_data = {
        "companies": "\n".join(state["company_list"]),  # 기업 리스트
        "tech_scribe": state["TechScribe"],  # 기술 요약
        "market_evaluator": state["MarketEvaluator"],  # 시장성 평가
        "competitor_analyzer": state["CompetitorAnalyzer"],  # 경쟁사 분석
        "investment_advisor": state["InvestmentAdvisor"]  # 투자 판단
    }
    
    # LLM에 데이터 전달하여 보고서 생성
    report_markdown = chain.run(formatted_data)
    
    return report_markdown