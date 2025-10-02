from typing import Dict, List

SYS_PROMPT = """
당신은 VC 기술투자 애널리스트다.
주어진 컨텍스트를 근거로 지시된 항목을 분석하라.

출력 규칙:
- 각 불릿에는 반드시 [출처: 파일명, p.xx] 형식의 근거를 포함한다.
- 근거가 없다면 해당 불릿은 쓰지 말고 "정보 부족"으로만 표기한다.
- 추측이나 확인되지 않은 정보는 금지한다.
"""

USER_PROMPT = """회사: {company}
분석 항목: {item_name}
지시문: {item_instruction}

<컨텍스트>
{context}
</컨텍스트>
위 컨텍스트만 근거로 분석하라."""

ANALYSIS_ITEMS: List[Dict[str, object]] = [
    {
        "name": "CEO(창업자)",
        "queries": [
            "{company} 창업자",
            "{company} CEO",
            "{company} 공동창업자",
            "{company} 경영진",
        ],
        "prompt": "창업자와 경영진의 이름, 배경, 경력을 [출처]와 함께 요약하라.",
        "doc_type": None,
    },
    {
        "name": "핵심 기술 요약",
        "queries": [
            "{company} 핵심 기술",
            "{company} 기술 요약",
            "{company} 주요 기술",
        ],
        "prompt": "회사가 보유한 주요 기술을 [출처]와 함께 3~4문장으로 정리하라.",
        "doc_type": "research",
    },
    {
        "name": "강점",
        "queries": [
            "{company} 강점",
            "{company} 경쟁우위",
            "{company} 차별화",
        ],
        "prompt": "기술/사업 강점을 [출처]와 함께 2~3개 불릿으로 작성하라.",
        "doc_type": None,
    },
    {
        "name": "약점",
        "queries": [
            "{company} 약점",
            "{company} 리스크",
            "{company} 문제점",
        ],
        "prompt": "위험요인 또는 약점을 [출처]와 함께 1~2개 불릿으로 작성하라.",
        "doc_type": None,
    },
    {
        "name": "핵심 모델/알고리즘",
        "queries": [
            "{company} 알고리즘",
            "{company} 모델",
            "{company} AI 기술",
        ],
        "prompt": "언급된 모델·알고리즘 명칭과 용도를 [출처]와 함께 열거하라.",
        "doc_type": "research",
    },
    {
        "name": "서비스/연동",
        "queries": [
            "{company} 서비스 연동",
            "{company} 플랫폼 연동",
            "{company} 파트너십",
        ],
        "prompt": "병원 시스템/파트너 연동 현황을 [출처]와 함께 요약하라.",
        "doc_type": "press_release",
    },
    {
        "name": "규제/인증",
        "queries": [
            "{company} 인증",
            "{company} 규제",
            "{company} 허가",
        ],
        "prompt": "의료·보안 인증 및 허가 현황을 [출처]와 함께 정리하라.",
        "doc_type": "regulation",
    },
    {
        "name": "고객 수",
        "queries": [
            "{company} 고객",
            "{company} 도입",
            "{company} 레퍼런스",
        ],
        "prompt": "고객/도입 병원 수나 대표 사례를 [출처]와 함께 제시하라.",
        "doc_type": "press_release",
    },
    {
        "name": "관련 논문/출판물",
        "queries": [
            "{company} 논문",
            "{company} 연구",
            "{company} 출판",
        ],
        "prompt": "기술 근거가 되는 연구·논문·발표를 [출처]와 함께 정리하라.",
        "doc_type": "research",
    },
]
