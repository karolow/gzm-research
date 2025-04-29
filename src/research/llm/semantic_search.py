import os
from typing import Any

import faiss
import joblib
import numpy as np
from dotenv import load_dotenv

from src.research.config import get_config

# Load environment variables
load_dotenv(override=True)

# Get configuration
config = get_config()

# Initialize variables for lazy loading
_index = None
_examples = None


def _load_resources():
    """Lazily load FAISS index and examples when needed."""
    global _index, _examples

    if _index is None or _examples is None:
        try:
            _index = faiss.read_index(config.faiss_index_path)
            _examples = joblib.load(config.examples_path)
        except (RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to load FAISS index or examples: {e}. "
                f"Make sure to build the index first with build_faiss_index.py"
            ) from e


def embed_text(text: str) -> np.ndarray:
    """
    Create an embedding for a single text using the configured embedding model.
    Supports OpenAI or other embedding providers based on environment config.
    """
    api_key = config.embedding.api_key or os.getenv("OPENAI_API_KEY")

    if api_key:
        from openai import OpenAI

        # Use the new OpenAI v1.0+ client
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=config.embedding.model, input=[text])
        emb = np.array(response.data[0].embedding, dtype="float32")
        return emb
    else:
        raise ValueError(
            "No embedding API key configured. Set OPENAI_API_KEY in .env file"
        )


def semantic_search(
    query: str, top_k: int = 10, exclude_indices: list[int] = None
) -> list[int]:
    """
    Retrieve top_k indices for semantically similar examples, excluding any in exclude_indices
    and the case identical to the query.
    """
    _load_resources()

    q_emb = embed_text(query).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    # reserve extra slots accounting for exclusions
    search_k = top_k + (len(exclude_indices) if exclude_indices else 0) + 1
    distances, indices = _index.search(q_emb, search_k)
    # filter out excluded indices and the query itself
    filtered: list[int] = []
    for idx in indices[0]:
        if exclude_indices and idx in exclude_indices:
            continue
        case = _examples[idx]
        # Check if the case is the query itself
        if isinstance(case, dict):
            # Try common text fields: inputs, question, query, text
            text = (
                case.get("inputs", "")
                or case.get("question", "")
                or case.get("query", "")
                or case.get("text", "")
                or str(case)
            )
        else:
            text = str(case)

        if text == query:
            continue
        filtered.append(idx)
        if len(filtered) >= top_k:
            break
    return filtered


class JinaReranker:
    """Reranker based on Jina or alternative reranking models."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")

    def rerank(
        self, query: str, docs: list[str], top_n: int
    ) -> list[tuple[str, float]]:
        """
        Rerank documents based on relevance to query.
        Returns list of (document, score) tuples sorted by relevance.

        This is a simple implementation - in production, use a proper reranking model.
        """
        try:
            # If jina is available, use it
            from jina import Document, DocumentArray

            # Simple reranking using Jina
            docs_with_scores = []
            for doc in docs:
                # Simple similarity calculation as fallback
                # In real implementation, would use proper semantic similarity
                score = len(set(query.split()) & set(doc.split())) / len(
                    set(query.split())
                )
                docs_with_scores.append((doc, score))

            # Sort by score descending and return top_n
            return sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:top_n]

        except ImportError:
            # Fallback if jina not available - simple word overlap scoring
            docs_with_scores = []
            for doc in docs:
                # Simple similarity calculation as fallback
                score = len(set(query.split()) & set(doc.split())) / len(
                    set(query.split())
                )
                docs_with_scores.append((doc, score))

            # Sort by score descending and return top_n
            return sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:top_n]


def semantic_rerank(
    query: str,
    top_n: int = 10,
    exclude_indices: list[int] = None,
    api_key: str | None = None,
) -> list[dict[str, Any] | str]:
    """
    Perform FAISS-based retrieval then rerank top_n cases using JinaReranker.
    """
    _load_resources()

    idxs = semantic_search(query, top_n, exclude_indices)
    candidates = [_examples[i] for i in idxs]

    # Extract text from candidates for reranking
    input_texts = []
    for case in candidates:
        if isinstance(case, dict):
            # Try common text fields: inputs, question, query, text
            text = (
                case.get("inputs", "")
                or case.get("question", "")
                or case.get("query", "")
                or case.get("text", "")
                or str(case)
            )
        else:
            text = str(case)
        input_texts.append(text)

    reranker = JinaReranker(api_key)
    ranked = reranker.rerank(query, input_texts, top_n)

    # Map back to full case dicts or strings
    text_to_idx = {text: i for i, text in enumerate(input_texts)}
    similar_cases = []
    for doc, _ in ranked:
        if doc in text_to_idx:
            idx = text_to_idx[doc]
            similar_cases.append(candidates[idx])
    return similar_cases
