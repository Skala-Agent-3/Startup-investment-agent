import json
from typing import List
from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def Orchestrator(state: GraphState) -> GraphState:
    """
    Orchestrator Agent:
    - 유저 질문에서 기업명을 추출해 JSON 배열로 변환
    - 결과를 state["company_list"]에 추가 저장
    """
    question = state["question"]

    # [1] 기업 리스트 추출 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 JSON만 출력하는 도우미다. "
                   "사용자가 언급한 기업명을 JSON 배열로 추출해서 "
                   "{{\"companies\": [\"기업1\", \"기업2\", ...]}} 형식으로만 출력해."),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": question})

    # [2] JSON 파싱
    companies: List[str] = []
    try:
        text = response.content.strip()
        data = json.loads(text)
        if isinstance(data, dict) and "companies" in data:
            companies = data["companies"]
    except Exception as e:
        print(f"[Orchestrator] JSON 파싱 실패: {e}, 응답={response.content}")

    # [3] 기존 state를 보존하면서 company_list만 업데이트
    #state["company_list"] = companies

    print(f"[Orchestrator] 기업 추출 결과: {companies}")
    return {"company_list": ['다노', '루닛', '레몬헬스케어', '슬립큐(과거 웰트)']}