import json
from typing import List
from dotenv import load_dotenv
from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def Orchestrator(state: GraphState) -> GraphState:
    """
    Orchestrator Agent:
    - 입력 질문을 LLM으로 열어 기업 리스트를 JSON으로 추출
    - 결과를 state["company_list"]에 반영
    """
    question = state["question"]

    # [1] 기업 리스트 추출 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": question})

    # [2] JSON 파싱
    try:
        data = json.loads(response.content)
        companies: List[str] = data.get("companies", []) if isinstance(data, dict) else []
    except Exception:
        companies = []

    # [3] state 업데이트
    print(state)
    state["company_list"] = companies
    return state