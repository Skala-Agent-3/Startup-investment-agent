from dotenv import load_dotenv
from langchain_teddynote import logging
from typing import Annotated, Sequence, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

load_dotenv()
logging.langsmith("Startup-investment-agent")

# GraphState
class GraphState(TypedDict, total=False):
    question: Annotated[str, "Question"]                  # 질문
    search_context: Annotated[str, "Context"]             # 검색 결과 (예: Tavily, PDF 등)
    rag_context: Annotated[str, "Context"]                # RAG 결과
    company_list: Annotated[List[str], "Companies"]       # 추출된 기업 리스트
    TechScribe: Annotated[str, "Context"]                 # 기술 요약 결과
    MarketEvaluator: Annotated[str, "Context"]            # 시장성 평가 결과
    CompetitorAnalyzer: Annotated[str, "Context"]         # 경쟁사 비교 결과
    InvestmentAdvisor: Annotated[str, "Context"]          # 투자 판단 결과
    answer: Annotated[str, "Answer"]                      # 최종 답변 / 보고서
    chat_history: Annotated[list, "Messages"]             # 누적 대화 로그\

def investment_advise_agent(state, prompt_path="prompts/InvestmentAdvisor_sys_prompt.txt"):
    sys_prompt = Path(prompt_path).read_text(encoding="utf-8")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt
            ),
            ("human",
            "아래는 전체 입력입니다. 이를 바탕으로 스키마에 맞는 결론만 산출하세요. \n"
            "- 경쟁사 비교: {{ competitor }}\n"
            "- 시장 평가: {{ market_eval }}\n"
            )
        ],
        template_format="jinja2"
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "competitor": state["CompetitorAnalyzer"],
        "market_eval": state["MarketEvaluator"]
        })

    return response