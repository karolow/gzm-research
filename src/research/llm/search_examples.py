#!/usr/bin/env python
"""
Search examples using semantic search.

This script demonstrates how to use the semantic search functionality
to find similar examples in the FAISS index.

Usage:
    uv run src/research/llm/search_examples.py --query "How many people participated in the survey?"
"""

import argparse
import sys

from research.llm.semantic_search import semantic_rerank, semantic_search


def search_examples(query: str, top_k: int = 5, rerank: bool = True):
    """Search for similar examples using semantic search."""
    try:
        if rerank:
            similar_cases = semantic_rerank(query, top_n=top_k)
            print(
                f"Found {len(similar_cases)} similar examples (ranked by relevance):\n"
            )

            for i, case in enumerate(similar_cases, 1):
                print(f"Example {i}:")
                if isinstance(case, dict):
                    # For dictionary examples
                    print(
                        f"  Query: {case.get('inputs', '') or case.get('question', '') or case.get('query', '') or str(case)}"
                    )
                    if "outputs" in case:
                        print(f"  SQL: {case.get('outputs', '')}")
                else:
                    # For string examples
                    print(f"  Query: {case}")
                print("-" * 50)
        else:
            # Import directly here to avoid circular imports
            from research.llm.semantic_search import _load_resources

            indices = semantic_search(query, top_k=top_k)
            print(f"Found {len(indices)} similar examples:")
            print(f"Indices: {indices}\n")

            # Load the examples if not already loaded
            _load_resources()
            from research.llm.semantic_search import _examples

            # Display the actual content
            for i, idx in enumerate(indices, 1):
                case = _examples[idx]
                print(f"Example {i} (index {idx}):")
                if isinstance(case, dict):
                    # For dictionary examples
                    print(
                        f"  Query: {case.get('inputs', '') or case.get('question', '') or case.get('query', '') or str(case)}"
                    )
                    if "outputs" in case:
                        print(f"  SQL: {case.get('outputs', '')}")
                else:
                    # For string examples
                    print(f"  Query: {case}")
                print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nMake sure you've built the FAISS index first with build_faiss_index.py."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Search examples using semantic search"
    )
    parser.add_argument(
        "--query", "-q", required=True, help="Query to search for similar examples"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of examples to return"
    )
    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        help="Use simple search without reranking",
    )

    args = parser.parse_args()

    search_examples(args.query, args.top_k, not args.simple)


if __name__ == "__main__":
    main()
