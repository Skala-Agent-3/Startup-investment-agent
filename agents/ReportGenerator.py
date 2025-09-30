from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ReportGenerator(state: GraphState) -> dict:
    """최종 투자 보고서를 생성하고 state['answer']에 저장"""
    prompt_template = """
    당신은 투자 전문가입니다. 주어진 기업 및 기술 정보를 바탕으로 아래의 요구 사항을 충족하는 투자 평가 보고서를 작성해 주세요.
    
    # 보고서 형식
    1. **헬스케어 산업을 조망하다** (시장성 평가)
    - 선택된 산업의 시장 개요, 규모, 성장률, 트렌드, 리스크에 대한 분석을 제공합니다.
    2. **헬스케어 스타트업 동향** (기술 요약)
    - 선택된 스타트업별 핵심 기술 요약, 강점/약점 리스트, 핵심 모델/알고리즘 목록, 시스템 연동 목록, 규제/인증 목록, 고객 수, 관련 논문/출판물 수 등을 포함한 기술 요약을 제공합니다.
    3. **투자 추천 기업 분석** (경쟁사 분석 및 투자 판단)
    - 경쟁사와의 비교를 통해 각 기업의 포지셔닝을 평가하고, 시장/기술 정보를 종합하여 투자 여부를 판단합니다.

    ## 목차
    1. 헬스케어 산업을 조망하다 (시장성 평가)
    2. 헬스케어 스타트업 동향 (기술 요약)
    3. 투자 추천 기업 분석 (경쟁사 분석 및 투자 판단)

## 1. 헬스케어 산업을 조망하다 (MarketEvaluator)
{market_evaluator}

## 2. 헬스케어 스타트업 동향 (TechScribe)
{tech_scribe}

## 3. 투자 추천 기업 분석 (CompetitorAnalyzer)
{competitor_analyzer}

## 4. 투자 판단 (InvestmentAdvisor)
{investment_advisor}

    # 보고서 작성 지침
    1. **헬스케어 산업을 조망하다 (시장성 평가)**:
    - 주어진 산업의 시장 개요, 시장 규모, 성장률, 트렌드, 리스크 등을 포함하여 상세히 분석해 주세요.
    
    2. **헬스케어 스타트업 동향 (기술 요약)**:
    - 각 스타트업의 핵심 기술 요약(간결하고 직관적으로) 및 강점/약점 분석을 포함합니다.
    - 각 스타트업이 사용하는 핵심 모델, 알고리즘, 시스템 연동, 규제 및 인증, 고객 수, 관련 논문/출판물 수 등 기술 관련 정보를 작성해 주세요.
    
    3. **투자 추천 기업 분석 (경쟁사 분석)**:
    - 각 기업의 경쟁력 분석을 위해 주요 경쟁사와 비교 분석을 작성해 주세요.
    - 투자 판단 섹션에서는 시장, 기술, 경쟁사 분석을 기반으로 투자 여부를 신중하게 판단하고 결론을 제시해 주세요.
    """

    # LLM 체인 생성
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "너는 벤처캐피탈 심사역에게 제출될 투자 분석 보고서를 작성하는 전문 분석가야."
            "아래 항목을 참고해서 객관적이고 논리적인 보고서를 작성해."
            "보고서는 명확한 제목과 항목별 소제목으로 구성하고, 각 참고한 내용을 독자가 이해하기 쉽도록 구체적으로 설명해줘."
            "문체는 간결하고 비즈니스적이며 판단 근거가 잘 드러나도록 해."
        ),
        (
            "user",
            prompt_template
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
    "market_evaluator": state.get("MarketEvaluator", ""),
    "tech_scribe": state.get("TechScribe", ""),
    "competitor_analyzer": state.get("CompetitorAnalyzer", ""),
    "investment_advisor": state.get("InvestmentAdvisor", "")
})


    # 👉 state에 answer 키로 저장

    print("최종 보고서 ~~~~~~~~~~~~~~~~~~~~")
    print(state)
    return {"answer": response.content.strip()}