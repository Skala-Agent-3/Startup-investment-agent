from graph_state import GraphState

from dotenv import load_dotenv
from langchain_teddynote import logging
from typing import Annotated, Sequence, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from prompts.investment_advisor import INVESTMENT_ADVISOR_USER_TEMPLATE, INVESTMENT_ADVISOR_SYSTEM_TEMPLATE

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

def InvestmentAdvisor(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INVESTMENT_ADVISOR_SYSTEM_TEMPLATE),
            ("human", INVESTMENT_ADVISOR_USER_TEMPLATE)
        ],
        template_format="jinja2"
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "competitor": state.get("CompetitorAnalyzer", ""),
        "market_eval": state.get("MarketEvaluator", "")
    })


    report = response.content if hasattr(response, "content") else str(response)
    print("InvestmentAdvisor state snapshot")
    print(state)
    return {
        "InvestmentAdvisor": report
    }


