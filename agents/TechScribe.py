# agents/TechScribe.py
from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYS_PROMPT = """
당신은 VC 기술실사 애널리스트다.
주어진 컨텍스트만 사용하여 회사의 기술력을 투자 관점에서 요약하라.

요약은 아래 항목을 반드시 포함해야 한다:
0. **CEO(창업자)**: 창업자 이름 등의 요약 정보를 정리.
1. **핵심 기술 요약**: 회사가 보유한 주요 기술을 간단히 정리.
2. **강점**: 기술적/사업적 강점 2~3개.
3. **약점**: 부족하거나 리스크가 있는 부분 1~2개.
4. **핵심 모델/알고리즘**: 사용 중이거나 언급된 모델, 알고리즘.
5. **시스템 연동**: 병원 시스템(EMR), 웨어러블, 앱 등과의 연동 여부.
6. **규제/인증**: 의료 인증, 보안 규제, ISO, HIPAA 등.
7. **고객 수**: 추정치나 언급된 수치. 없으면 "정보 부족".
8. **관련 논문/출판물 수**: 기술적 신뢰성을 뒷받침하는 연구 결과. 없으면 "정보 부족".

⚠️ 반드시 항목별 소제목과 리스트 형식을 유지하고, JSON은 사용하지 말고 텍스트로만 작성하라.
"""

USER_PROMPT = """회사: {company}
<컨텍스트>
{context}
</컨텍스트>
"""

def make_techscribe_agent(state: GraphState) -> dict:
    """company_list 전체를 순회하며 기술 요약을 생성"""
    retriever = state.get("retriever")
    company_list = state.get("company_list", [])

    all_summaries = []
    for company in company_list:
        docs = retriever.get_relevant_documents(company) if retriever else []
        context = "\n\n".join([d.page_content for d in docs]) or "관련 문서 없음"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYS_PROMPT),
            ("user", USER_PROMPT)
        ])

        chain = prompt | llm
        response = chain.invoke({"company": company, "context": context})

        new_text = f"\n\n### {company}\n{response.content.strip()}"
        print(f"[TechScribe] {company} 처리 완료")
        all_summaries.append(new_text)
    print("여기가 첫번째꺼")
    print(state)
    return {"TechScribe": "\n".join(all_summaries)}
