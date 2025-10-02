import os
import certifi
import sys
from typing import List, Any, Optional, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from prompts.market_evaluator import MARKET_REPORT_TEMPLATE

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Import GraphState from the project when available
try:
    from graph_state import GraphState  # {'MarketEvaluator': str, 'search_context': str, ...}
except Exception:
    # Fallback definition for isolated runs or documentation builds
    from typing import TypedDict

    class GraphState(TypedDict, total=False):
        MarketEvaluator: str
        search_context: str

# Configure root logger for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("market_eval")

# Load environment variables for API keys and models
load_dotenv()

# Avoid importing heavy frameworks when not needed
os.environ["USE_TF"] = "0"   # Disable TensorFlow
os.environ["USE_JAX"] = "0"  # Disable JAX

# LLM configuration
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Utility: format retrieved documents into a compact context string
def format_docs(docs: List[Any], max_chars: int = 6000) -> str:
    """Stitch document excerpts into a concise text block."""
    parts, total_len = [], 0
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        meta: Dict[str, Any] = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("source_url") or meta.get("url")
        entry = f"[source] {src}\n{text}" if src else text
        if total_len + len(entry) > max_chars:
            break
        parts.append(entry)
        total_len += len(entry)
    return "\n\n".join(parts)

# Market report prompt definition
market_report_prompt = PromptTemplate(
    input_variables=["context"],
    template=MARKET_REPORT_TEMPLATE,
)

# Retrieve, load, and split web documents based on a search query
def web_search_docs(query: str, num_results: int = 5) -> List[Any]:
    """Run DuckDuckGo search, load pages, and split them into chunks."""
    try:
        ddg = DuckDuckGoSearchAPIWrapper(
            region="kr-kr",      # Region can be adjusted (e.g., "us-en")
            safesearch="Moderate",
            time="y",            # Options: d / w / m / y
        )
        results = ddg.results(query, max_results=num_results) or []
        logger.info(f"DDG results fetched: {len(results)}")
    except Exception as e:
        logger.error(f"DuckDuckGo error: {e!r}")
        return []

    urls: List[str] = []
    for item in results:
        url = item.get("link") or item.get("href")
        if url:
            urls.append(url)
    if not urls:
        logger.warning("No URLs extracted from DDG results.")
        return []

    try:
        loader = WebBaseLoader(web_paths=urls)
        raw_docs = loader.load()
        logger.info(f"WebBaseLoader loaded docs: {len(raw_docs)}")
    except Exception as e:
        logger.error(f"WebBaseLoader error: {e!r}")
        return []

    if not raw_docs:
        logger.warning("Loader returned empty documents.")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    split_docs = splitter.split_documents(raw_docs)
    logger.info(f"Split into chunks: {len(split_docs)}")
    return split_docs

# Build a vector store and RetrievalQA chain from documents
def create_market_eval_agent(docs: List[Any]) -> RetrievalQA:
    """Create a FAISS-backed RetrievalQA chain using a HuggingFace embedding model."""
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    logger.info(f"Using embedding model: {embedding_model}")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    except Exception as e:
        logger.error(
            "HuggingFaceEmbeddings init failed. "
            "Verify the model access rights or the HUGGINGFACEHUB_API_TOKEN. "
            f"error={e!r}"
        )
        raise

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    if not split_docs:
        logger.warning("No split_docs produced for vector store.")
        split_docs = docs  # Ensure we always have material for indexing

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": market_report_prompt},
        return_source_documents=False,
    )
    return chain

# LangGraph node: build market analysis and persist state
def MarketEvaluator(state: GraphState) -> GraphState:
    """Run web RAG on the healthcare scale-up market and store the summary."""
    query = "Healthcare scale-up market analysis"
    logger.info(f"[MarketEvaluator] Query: {query}")

    docs = web_search_docs(query, num_results=5)

    if not docs:
        state["MarketEvaluator"] = "No search results"
        state["search_context"] = ""
        logger.info("[MarketEvaluator] No docs; state updated with empty result.")
        return state

    logger.info(f"[MarketEvaluator] docs={len(docs)}; building agent...")
    try:
        agent = create_market_eval_agent(docs)
    except Exception as e:
        msg = f"Market evaluator initialization failed: {e!r}"
        logger.error(f"[MarketEvaluator] {msg}")
        state["MarketEvaluator"] = msg
        state["search_context"] = format_docs(docs)
        return state

    try:
        result = agent.invoke({"query": query})
        report = result.get("result") or result.get("output_text", "")
        logger.info("[MarketEvaluator] Chain invocation succeeded.")
    except Exception as e:
        report = f"Chain execution failed: {e!r}"
        logger.error(f"[MarketEvaluator] {report}")

    state["search_context"] = format_docs(docs)
    state["MarketEvaluator"] = report

    logger.info("[MarketEvaluator] State updated.")
    print("MarketEvaluator state snapshot")
    print(state)
    return state