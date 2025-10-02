from typing import Annotated, Any, List, TypedDict


class GraphState(TypedDict):
    question: Annotated[str, "Question"]                  # 🎉 Original user question
    search_context: Annotated[str, "Context"]             # 🎉 External search results (e.g., Tavily, PDFs)
    rag_context: Annotated[str, "Context"]                # 🎉 RAG output shared across agents
    company_list: Annotated[List[str], "Companies"]       # 🎉 Extracted company names
    retriever: Annotated[Any, "Retriever"]                # 🎉 Shared retriever instance for RAG
    TechScribe: Annotated[str, "Context"]                 # 🎉 TechScribe summary output
    MarketEvaluator: Annotated[str, "Context"]            # 🎉 Market analysis output
    CompetitorAnalyzer: Annotated[str, "Context"]         # 🎉 Competitor analysis output
    InvestmentAdvisor: Annotated[str, "Context"]          # 🎉 Investment recommendation output
    answer: Annotated[str, "Answer"]                      # 🎉 Final report content
    chat_history: Annotated[list, "Messages"]             # 🎉 Conversation history if needed
