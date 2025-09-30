import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from graph_state import GraphState


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------------------
# 핵심 비교 로직 (동기)
# ------------------------------
def CompetitorAnalyzer(state: GraphState) -> GraphState:
    """
    tech_scribe: 각 회사별 기술 요약 (TechScribe: Annotated[str, "Context"])
    출력: CompetitorAnalyzer: Annotated[str, "Context"]
    """
    tech_scribe = state.get("TechScribe", "")

    prompt = ChatPromptTemplate.from_messages([
(
    "system",
    """
당신은 Mckinsey, BCG, Bain과 같은 Top-Tier 전략 컨설팅 펌의 시니어 컨설턴트입니다.  
당신의 분석은 C-level 경영진에게 보고될 예정이므로, 날카롭고(Sharp), 간결하며(Concise), 데이터에 기반(Data-driven)해야 합니다.  
불필요한 미사여구나 모호한 표현은 배제하고, 전략적 의사결정에 직접적인 영향을 미칠 수 있는 핵심만 짚어내야 합니다.  
정보가 부족하다면 반드시 "N/A (정보 부족)"으로 표기하고 간단히 이유를 제시하세요.  

**## 분석의 목표 (Objective) ##**  
주어진 기술 요약(TechScribe 결과)을 바탕으로, [예: 신규 시장 진출 전략 수립, 잠재적 M&A 후보군 평가, 자사 제품 포지셔닝 재정립]을 위한 핵심 경쟁 환경을 분석합니다.  

**## 분석 작업 (Tasks) ##**  

**1. 경쟁력 대시보드 (Competitive Dashboard)**  
- 각 기업을 핵심 성공 요인(KSF)에 대해 정성적으로 설명  
- '강점(Strength)', '제한적(Limited)', 'N/A (정보 부족)' 등의 설명 위주로 작성  
- 단순한 점수화나 "Leader / Follower / Niche" 등의 포지셔닝 레이블은 절대 사용하지 말 것  
- 반드시 근거 제시  

KSF:  
- 기술적 해자 (Technological Moat)  
- 시장 선점 및 침투력 (Market Penetration & First-Mover Advantage)  
- 사업 확장성 (Scalability)  
- 고객 락인 효과 (Customer Lock-in)  
- 규제 및 인허가 (Regulatory Readiness)  
- 재무 건전성 (Financial Health)  
- 파트너십 및 생태계 (Partner Ecosystem)  

**2. 기업별 심층 분석 (Deep-Dive Analysis)**  
- 각 기업별 최소 300단어 이상 분석  
- 창업자 정보
- SWOT, 5 Forces, KSF 상세 평가 포함  
- 반드시 '분석 목표'의 관점에서 해석  

**3. 종합 비교표 (Summary Table)**  
- KSF 항목별로 각 기업의 특징을 간단히 비교할 수 있는 표 작성  
- 설명 기반으로만 작성 (점수, 등급, 순위, 레이블 없음)  

---  
**## 출력 형식 (Output Format) ##**  
CompetitorAnalyzer: Annotated[str, "Context"] 형태로 단일 text 반환.  
JSON 스키마 준수:  

{{
   "competitive_dashboard": [...],
   "deep_dive_analysis": {{
      "company_a": {{...}},
      "company_b": {{...}}
   }},
   "competitive_summary_table": {{
      "ksf_comparison": [...]
   }}
}}
"""
),
("user", "기술 요약 (TechScribe 결과):\n{tech_scribe}\n\n위 가이드라인에 따라 경쟁사 분석 보고서를 생성해주세요.")
])

    result = (prompt | llm).invoke({"tech_scribe": tech_scribe})
    report = result.content if hasattr(result, "content") else str(result)

    result = (prompt | llm).invoke({"tech_scribe": tech_scribe})

    print("경쟁사 어쩌고 ~~~~~~~~")
    print(state)
    
    return {
        "CompetitorAnalyzer": report
    }