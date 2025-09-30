import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graph_state import GraphState

load_dotenv()

def CompetitorAnalyzer(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tech_scribe = state.get("TechScribe", "")

    with open("prompts/competitor_analyzer_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "기술 요약 (TechScribe 결과):\n{tech_scribe}\n\n위 가이드라인에 따라 경쟁사 분석 보고서를 생성해주세요.")
    ])

    result = (prompt | llm).invoke({"tech_scribe": tech_scribe})
    report = result.content if hasattr(result, "content") else str(result)
    
    return {
        "CompetitorAnalyzer": report
    }