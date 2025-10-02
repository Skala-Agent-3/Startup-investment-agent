import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


@dataclass
class RetrievedParent:
    document: Document
    dense_score: float
    sparse_score: float
    rerank_score: float


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.array(vector, dtype="float32")
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[\w\-]+", text.lower())


def _infer_doc_type(source_path: str) -> str:
    name = os.path.basename(source_path or "").lower()
    if any(keyword in name for keyword in ["press", "release", "news"]):
        return "press_release"
    if any(keyword in name for keyword in ["regulation", "compliance", "policy"]):
        return "regulation"
    if any(keyword in name for keyword in ["whitepaper", "paper", "study", "report"]):
        return "research"
    return "unspecified"


def _detect_companies(text: str) -> List[str]:
    english_matches = re.findall(r"[A-Z][A-Za-z0-9&\-]{2,}", text)
    korean_matches = re.findall(r"[\uac00-\ud7af]{2,}", text)
    combined = {m.lower() for m in english_matches + korean_matches}
    return list(combined)


class AdvancedRetriever:
    """Compose dense, sparse, and cross-encoder reranking with parent-child context."""

    def __init__(
        self,
        parent_documents: List[Document],
        child_documents: List[Document],
        child_embeddings: np.ndarray,
        embedding_dimension: int,
        embedding_model,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 50,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        similarity_threshold: float = 0.35,
        rerank_threshold: float = 0.30,
        max_context_tokens: int = 4000,
    ) -> None:
        self.parent_documents = {doc.metadata["parent_id"]: doc for doc in parent_documents}
        self.child_documents = child_documents
        self.child_embeddings = child_embeddings.astype("float32")
        self.embedding_model = embedding_model
        self.fetch_k = fetch_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.similarity_threshold = similarity_threshold
        self.rerank_threshold = rerank_threshold
        self.max_context_tokens = max_context_tokens

        index = faiss.IndexFlatIP(embedding_dimension)
        if self.child_embeddings.size:
            index.add(self.child_embeddings)
        self.index = index

        tokenized_children = [_tokenize_for_bm25(doc.page_content) for doc in child_documents]
        self.bm25 = BM25Okapi(tokenized_children)

        self.cross_encoder = CrossEncoder(cross_encoder_model, device="cpu")

    def _mmr(self, candidate_ids: List[int], query_vector: np.ndarray, k: int, lambda_mult: float) -> List[int]:
        if not candidate_ids:
            return []
        selected: List[int] = []
        candidate_set = set(candidate_ids)
        query_vector = _normalize_vector(query_vector)
        while candidate_set and len(selected) < k:
            best_id = None
            best_score = -math.inf
            for candidate in candidate_set:
                if candidate >= len(self.child_embeddings):
                    continue
                candidate_vec = self.child_embeddings[candidate]
                similarity = float(np.dot(query_vector, candidate_vec))
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(
                        float(np.dot(candidate_vec, self.child_embeddings[chosen])) for chosen in selected
                    )
                mmr_score = lambda_mult * similarity - (1.0 - lambda_mult) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_id = candidate
            if best_id is None:
                break
            selected.append(best_id)
            candidate_set.remove(best_id)
        return selected

    def _candidate_scores(self, query: str) -> Tuple[np.ndarray, Dict[int, float], Dict[int, float]]:
        query_vector = _normalize_vector(self.embedding_model.embed_query(query))
        if self.child_embeddings.size:
            similarity_scores, indices = self.index.search(query_vector.reshape(1, -1), self.fetch_k)
            dense_lookup = {idx: float(score) for idx, score in zip(indices[0], similarity_scores[0])}
        else:
            dense_lookup = {}
        tokenized_query = _tokenize_for_bm25(query)
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_lookup = {idx: float(score) for idx, score in enumerate(sparse_scores)}
        return query_vector, dense_lookup, sparse_lookup

    def _combined_candidates(
        self,
        dense_lookup: Dict[int, float],
        sparse_lookup: Dict[int, float],
        query_vector: np.ndarray,
        lambda_mult: float,
    ) -> List[int]:
        dense_indices = [idx for idx, _ in sorted(dense_lookup.items(), key=lambda item: item[1], reverse=True)]
        dense_ids = self._mmr(dense_indices[: self.fetch_k], query_vector, self.fetch_k, lambda_mult)

        sorted_sparse = sorted(sparse_lookup.items(), key=lambda item: item[1], reverse=True)
        sparse_ids = [idx for idx, score in sorted_sparse if score > 0.0][: self.fetch_k]

        return list(dict.fromkeys(dense_ids + sparse_ids))

    def _apply_filters(self, doc: Document, company: Optional[str], doc_type: Optional[str]) -> bool:
        if doc_type and doc.metadata.get("doc_type") != doc_type:
            return False
        if company:
            lower_value = company.lower()
            companies_lower = doc.metadata.get("companies_lower", [])
            if lower_value not in companies_lower and lower_value not in doc.page_content.lower():
                return False
        return True

    def _build_parent_response(
        self,
        candidate_ids: Iterable[int],
        dense_lookup: Dict[int, float],
        sparse_lookup: Dict[int, float],
        query_vector: np.ndarray,
        query_text: str,
        company: Optional[str],
        doc_type: Optional[str],
        top_k: int,
    ) -> List[RetrievedParent]:
        parent_candidates: Dict[str, RetrievedParent] = {}
        max_sparse = max((value for value in sparse_lookup.values() if value > 0.0), default=1.0)

        for idx in candidate_ids:
            if idx >= len(self.child_documents):
                continue
            child = self.child_documents[idx]
            similarity = dense_lookup.get(idx, 0.0)
            if similarity < self.similarity_threshold:
                continue
            if not self._apply_filters(child, company, doc_type):
                continue

            parent_id = child.metadata["parent_id"]
            parent = self.parent_documents.get(parent_id)
            if not parent:
                continue

            sparse_score = sparse_lookup.get(idx, 0.0)
            scaled_sparse = sparse_score / max_sparse if max_sparse > 0.0 else 0.0
            combined = self.dense_weight * similarity + self.sparse_weight * scaled_sparse

            existing = parent_candidates.get(parent_id)
            if not existing or combined > existing.dense_score:
                parent_candidates[parent_id] = RetrievedParent(
                    document=Document(page_content=parent.page_content, metadata=dict(parent.metadata)),
                    dense_score=combined,
                    sparse_score=scaled_sparse,
                    rerank_score=0.0,
                )

        if not parent_candidates:
            return []

        ranked_ids = sorted(parent_candidates.keys(), key=lambda key: parent_candidates[key].dense_score, reverse=True)
        rerank_pool = ranked_ids[: max(top_k * 5, 30)]
        rerank_pairs = [
            (query_text, parent_candidates[parent_id].document.page_content)
            for parent_id in rerank_pool
        ]
        rerank_scores = self.cross_encoder.predict(rerank_pairs) if rerank_pairs else []

        for parent_id, score in zip(rerank_pool, rerank_scores):
            parent_candidates[parent_id].rerank_score = float(score)

        filtered = [
            parent_candidates[parent_id]
            for parent_id in rerank_pool
            if parent_candidates[parent_id].rerank_score >= self.rerank_threshold
        ]
        filtered.sort(key=lambda item: item.rerank_score, reverse=True)

        final_documents: List[RetrievedParent] = []
        seen_keys = set()
        for item in filtered[:top_k]:
            doc = item.document
            citation = f"[출처: {doc.metadata.get('source')}, {doc.metadata.get('pages')}]"
            doc.metadata["citation"] = citation
            doc.metadata["score"] = item.rerank_score
            doc.metadata["dense_score"] = item.dense_score
            doc.metadata["sparse_score"] = item.sparse_score
            token_estimate = len(doc.page_content) // 4
            if token_estimate > self.max_context_tokens:
                ratio = self.max_context_tokens / max(token_estimate, 1)
                truncated_length = max(200, int(len(doc.page_content) * ratio))
                doc.page_content = doc.page_content[: truncated_length]
            key = (doc.metadata.get("source"), doc.metadata.get("pages"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            final_documents.append(item)

        return final_documents

    def search(
        self,
        query: str,
        *,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: int = 8,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        query_vector, dense_lookup, sparse_lookup = self._candidate_scores(query)
        candidate_ids = self._combined_candidates(dense_lookup, sparse_lookup, query_vector, lambda_mult)
        parents = self._build_parent_response(
            candidate_ids,
            dense_lookup,
            sparse_lookup,
            query_vector,
            query,
            company,
            doc_type,
            top_k,
        )
        return [item.document for item in parents]

    def batch_search(
        self,
        queries: Sequence[str],
        *,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        per_query_k: int = 4,
        max_chunks: int = 16,
    ) -> List[Document]:
        collected: Dict[Tuple[str, str], Document] = {}
        for query in queries:
            docs = self.search(query, company=company, doc_type=doc_type, top_k=per_query_k)
            for doc in docs:
                key = (doc.metadata.get("source"), doc.metadata.get("pages"))
                if key not in collected:
                    collected[key] = doc
        return list(collected.values())[:max_chunks]


def build_parent_child_documents(
    raw_documents: List[Document],
    parent_chunk_size: int = 2200,
    child_chunk_size: int = 420,
) -> Tuple[List[Document], List[Document]]:
    """Split page-level documents into parent and child granularities."""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=60)

    parent_documents: List[Document] = []
    child_documents: List[Document] = []

    for raw_doc in raw_documents:
        source_path = raw_doc.metadata.get("source")
        page_num = raw_doc.metadata.get("page", 0)
        parent_chunks = parent_splitter.split_documents(
            [Document(page_content=raw_doc.page_content, metadata=dict(raw_doc.metadata))]
        )
        for parent_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent_metadata = {
                "parent_id": parent_id,
                "source": source_path,
                "doc_type": _infer_doc_type(source_path or ""),
                "page": page_num + 1,
                "pages": f"p.{page_num + 1}",
            }
            companies = _detect_companies(parent_chunk.page_content)
            parent_metadata["companies"] = companies
            parent_metadata["companies_lower"] = [value.lower() for value in companies]
            parent_doc = Document(page_content=parent_chunk.page_content, metadata=parent_metadata)
            parent_documents.append(parent_doc)

            child_chunks = child_splitter.split_documents([parent_doc])
            for child_chunk in child_chunks:
                child_metadata = dict(parent_metadata)
                child_metadata["doc_id"] = str(uuid.uuid4())
                child_metadata["parent_id"] = parent_id
                child_documents.append(Document(page_content=child_chunk.page_content, metadata=child_metadata))

    return parent_documents, child_documents


def create_advanced_retriever(
    raw_documents: List[Document],
    embedding_model,
    *,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    parent_chunk_size: int = 2200,
    child_chunk_size: int = 420,
) -> AdvancedRetriever:
    """Build an AdvancedRetriever from raw PDF documents and an embedding model."""
    parent_docs, child_docs = build_parent_child_documents(
        raw_documents,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
    )
    child_texts = [doc.page_content for doc in child_docs]
    if child_texts:
        child_embeddings = np.array(embedding_model.embed_documents(child_texts), dtype="float32")
        child_embeddings = np.array([_normalize_vector(vec) for vec in child_embeddings], dtype="float32")
        embedding_dimension = child_embeddings.shape[1]
    else:
        probe = np.array(embedding_model.embed_query("dimension probe"), dtype="float32")
        embedding_dimension = probe.shape[0]
        child_embeddings = np.empty((0, embedding_dimension), dtype="float32")
    return AdvancedRetriever(
        parent_documents=parent_docs,
        child_documents=child_docs,
        child_embeddings=child_embeddings,
        embedding_dimension=embedding_dimension,
        embedding_model=embedding_model,
        cross_encoder_model=cross_encoder_model,
    )
