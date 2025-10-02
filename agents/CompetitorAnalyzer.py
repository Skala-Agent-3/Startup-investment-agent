import os
import logging

from dotenv import load_dotenv
from graph_state import GraphState
from langchain_openai import ChatOpenAI
from prompts.competitor_analyzer import COMPETITOR_PROMPT

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def CompetitorAnalyzer(state: GraphState) -> GraphState:
    """🎉 Generate a competitor analysis report based on the TechScribe summary."""
    tech_scribe = state.get("TechScribe", "")
    chain = COMPETITOR_PROMPT | llm
    response = chain.invoke({"tech_scribe": tech_scribe})
    report = response.content if hasattr(response, "content") else str(response)

    print("[CompetitorAnalyzer] state snapshot")
    print(state)

    return {"CompetitorAnalyzer": report}