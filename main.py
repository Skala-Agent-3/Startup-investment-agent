import os
import glob
import logging
from dotenv import load_dotenv

from typing import List

# Import Agents
from agents.Orchestrator import Orchestrator
from agents.TechScribe import make_techscribe_agent
from agents.MarketEvaluator import MarketEvaluator
from agents.CompetitorAnalyzer import CompetitorAnalyzer
from agents.InvestmentAdvisor import InvestmentAdvisor
from agents.ReportGenerator import ReportGenerator
from outputs.create_pdf import save_report_to_pdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from graph_state import GraphState
from langgraph.graph import StateGraph, START, END
from rag.advanced_retriever import create_advanced_retriever

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


def build_pdf_retriever(pdf_paths: List[str]):
    """Build the advanced parent-child retriever with dense/sparse/rerank stacks."""
    if not pdf_paths:
        raise ValueError("No PDF paths provided for retriever construction.")

    raw_documents: List[Document] = []
    for path in pdf_paths:
        print(f"[LOAD] {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            metadata = dict(doc.metadata)
            metadata.setdefault("page", metadata.get("page_number", 0))
            metadata["source_path"] = path
            metadata["source"] = os.path.basename(path)
            doc.metadata = metadata
            raw_documents.append(doc)
    print(f"[INFO] Total page documents: {len(raw_documents)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": False},
    )

    retriever = create_advanced_retriever(raw_documents, embeddings)
    print("[INFO] Advanced retriever ready")
    return retriever


def build_graph(company_list: List[str]):
    workflow = StateGraph(GraphState)

    workflow.add_node("techscribe", make_techscribe_agent)   # Company technology summary
    workflow.add_node("market_eval", MarketEvaluator)        # Market analysis
    workflow.add_node("competitor", CompetitorAnalyzer)      # Competitor comparison
    workflow.add_node("investment", InvestmentAdvisor)       # Investment recommendation
    workflow.add_node("report", ReportGenerator)             # Final report assembly

    workflow.add_edge(START, "techscribe")
    workflow.add_edge("techscribe", "market_eval")
    workflow.add_edge("market_eval", "competitor")
    workflow.add_edge("competitor", "investment")
    workflow.add_edge("investment", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def make_init_state(question: str) -> GraphState:
    """Prepare the default GraphState container."""
    return {
        "question": question,
        "company_list": [],
        "retriever": None,
        "search_context": "",
        "rag_context": "",
        "TechScribe": "",
        "MarketEvaluator": "",
        "CompetitorAnalyzer": "",
        "InvestmentAdvisor": "",
        "answer": "",
        "chat_history": [],
    }


def main():
    pdf_files = glob.glob("data/*.pdf")

    retriever = build_pdf_retriever(pdf_files)

    state = make_init_state(
        "헬스케어 스타트업인 다노, 루닛, 레몬헬스케어, 슬립큐(과거 웰트)를 투자 고민 중이야. 투자 분석 보고서를 작성해줘."
    )
    state["retriever"] = retriever

    state = Orchestrator(state)
    companies = state.get("company_list", [])
    print("Orchestrator extracted companies:", companies)

    graph = build_graph(companies)
    result = graph.invoke(state)

    output_path = os.path.join("outputs", "final_report.pdf")
    save_report_to_pdf(result["answer"], output_path)
    print(result["answer"])
    print("Report saved to outputs/final_report.pdf")


if __name__ == "__main__":
    main()
