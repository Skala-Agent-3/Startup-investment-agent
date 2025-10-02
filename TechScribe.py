from typing import Dict, List

from graph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompts.techscribe import ANALYSIS_ITEMS, SYS_PROMPT, USER_PROMPT

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _format_context(docs: List) -> str:
    if not docs:
        return "정보 부족"
    segments = []
    for doc in docs:
        citation = doc.metadata.get("citation", "[출처: 미상]")
        snippet = doc.page_content.strip()
        segments.append(f"{citation} {snippet}")
    return "\n\n".join(segments)


def _enforce_citations(text: str) -> str:
    lines = text.splitlines()
    enforced: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            enforced.append(line)
            continue
        if stripped.startswith(('-', '*')):
            if '[출처:' not in stripped and '정보 부족' not in stripped:
                enforced.append('- 정보 부족')
            else:
                enforced.append(line)
        else:
            enforced.append(line)
    return "\n".join(enforced)


def search_for_item(retriever, company: str, queries: List[str], doc_type: str | None, per_query_k: int = 4, max_docs: int = 8) -> str:
    formatted_queries = [query_template.format(company=company) for query_template in queries]
    docs = retriever.batch_search(
        formatted_queries,
        company=company,
        doc_type=doc_type,
        per_query_k=per_query_k,
        max_chunks=max_docs,
    )
    return _format_context(docs)


def make_techscribe_agent(state: GraphState) -> dict:
    """Iterate through each company and build structured RAG-backed analyses."""
    retriever = state.get("retriever")
    if retriever is None:
        raise ValueError("TechScribe2 requires a retriever in the graph state before running.")

    company_list = state.get("company_list", [])
    all_summaries: List[str] = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("user", USER_PROMPT),
    ])
    chain = prompt | llm

    for company in company_list:
        print("=" * 60)
        print(f"[TechScribe] {company} analysis start")
        print("=" * 60)

        company_summary = [f"### {company}\n"]
        for idx, item in enumerate(ANALYSIS_ITEMS, start=1):
            print(f"  [{idx}/9] {item['name']} in progress...")
            context = search_for_item(
                retriever,
                company,
                item["queries"],
                item.get("doc_type"),
            )
            response = chain.invoke(
                {
                    "company": company,
                    "item_name": item["name"],
                    "item_instruction": item["prompt"],
                    "context": context,
                }
            )
            answer = _enforce_citations(response.content.strip())
            company_summary.append(f"**{idx}. {item['name']}**")
            company_summary.append(answer)
            company_summary.append("")

        all_summaries.append("\n".join(company_summary))
        print(f"[TechScribe] {company} analysis complete\n")

    print("=" * 60)
    print("[TechScribe] Completed analysis for all companies")
    print("=" * 60)
    return {"TechScribe": "\n\n".join(all_summaries)}
