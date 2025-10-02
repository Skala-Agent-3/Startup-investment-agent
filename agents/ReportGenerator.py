from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompts.report_generator import REPORT_SYSTEM_PROMPT, REPORT_USER_TEMPLATE
from dotenv import load_dotenv
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def ReportGenerator(state: GraphState) -> dict:
    """최종 투자 보고서를 생성하고 state['answer']에 저장한다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM_PROMPT),
        ("user", REPORT_USER_TEMPLATE),
    ])
    chain = prompt | llm

    response = chain.invoke(
        {
            "market_evaluator": state.get("MarketEvaluator", ""),
            "tech_scribe": state.get("TechScribe", ""),
            "competitor_analyzer": state.get("CompetitorAnalyzer", ""),
            "investment_advisor": state.get("InvestmentAdvisor", ""),
        }
    )

    print("[ReportGenerator] state snapshot")
    print(state)
    return {"answer": response.content.strip()}
