import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set up logging
logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Uses Jina AI's reranker to find similar examples based on semantic search.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Jina AI reranker.

        Args:
            api_key: Optional API key for Jina AI. If not provided, will read from JINA_API_KEY environment variable.
        """
        # Load API key from environment if not provided
        self.api_key = api_key or os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Jina API key is required but not provided and not found in environment variables"
            )

        self.url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.model = "jina-reranker-v2-base-multilingual"

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents based on similarity to the query.

        Args:
            query: The query to compare documents against
            documents: List of document texts to rank
            top_n: Number of top results to return

        Returns:
            List of tuples (document, score) sorted by relevance score in descending order
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        data = {
            "model": self.model,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            "return_documents": True,
        }

        try:
            logger.debug(
                f"Sending rerank request to {self.url} with {len(documents)} documents"
            )
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()

            # Log the raw response for debugging
            logger.debug(f"Received response: {json.dumps(result, indent=2)}")

            # Check if 'results' key exists in the response
            if "results" not in result:
                logger.error(
                    f"Invalid API response format, missing 'results' key: {result}"
                )
                return []

            ranked_results = []
            for item in result["results"]:
                # Try to extract document and score with robust error handling
                try:
                    # Handle the nested document.text structure in the new API format
                    if (
                        isinstance(item.get("document"), dict)
                        and "text" in item["document"]
                    ):
                        document = item["document"]["text"]
                    else:
                        document = item.get("document", "")

                    # Support both old 'score' and new 'relevance_score' fields
                    score = item.get("relevance_score", item.get("score", 0.0))

                    if document:  # Only include if document is not empty
                        ranked_results.append((document, float(score)))
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error processing result item {item}: {e}")

            return ranked_results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error during reranking request: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during reranking: {e}")
            return []


def find_similar_examples(
    query: str,
    eval_dataset_path: str | Path,
    top_n: int = 10,
    api_key: str | None = None,
    exclude_range: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Find similar examples from the evaluation dataset based on the query.

    Args:
        query: User query to find similar examples for
        eval_dataset_path: Path to the evaluation dataset JSON file
        top_n: Number of top examples to return
        api_key: Jina API key (optional)
        exclude_range: Range of cases to exclude in format "start-end" (1-indexed)

    Returns:
        List of top N examples from the dataset
    """
    # Load dataset
    try:
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading dataset from {eval_dataset_path}: {e}")
        return []

    cases = dataset.get("cases", [])
    if not cases:
        logger.warning(f"No cases found in dataset at {eval_dataset_path}")
        return []

    # Parse exclude_range if provided
    excluded_indices = set()
    if exclude_range:
        try:
            start, end = parse_range(exclude_range)
            excluded_indices = set(range(start, end))
            logger.info(
                f"Excluding cases in range {start + 1}-{end} from reranking examples"
            )
        except ValueError as e:
            logger.warning(f"Invalid exclude range format: {e}. Using all cases.")

    # Extract inputs for reranking
    documents = []
    cases_map = {}
    for idx, case in enumerate(cases):
        # Skip cases that are in the excluded range
        if idx in excluded_indices:
            continue

        input_text = case.get("inputs", "")
        if input_text:
            documents.append(input_text)
            cases_map[input_text] = case

    logger.info(
        f"Found {len(documents)} examples in dataset for reranking (after exclusions)"
    )

    if not documents:
        logger.warning("No valid documents found in dataset for reranking")
        return []

    # Perform reranking
    try:
        reranker = JinaReranker(api_key)
        ranked_results = reranker.rerank(query, documents, top_n=top_n)

        if not ranked_results:
            logger.warning("Reranker returned no results")
            return []

        # Log the top results
        logger.info(f"Top {len(ranked_results)} similar cases to query: {query}")
        for i, (doc, score) in enumerate(ranked_results):
            logger.info(f"{i + 1}. Score: {score:.4f} | Query: {doc}")

        # Return the top examples
        similar_cases = []
        for doc, _ in ranked_results:
            if doc in cases_map:
                similar_cases.append(cases_map[doc])
            else:
                logger.warning(f"Document not found in cases map: {doc[:50]}...")

        return similar_cases

    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        return []


def parse_range(range_str: str) -> tuple[int, int]:
    """
    Parse a range string in format 'start-end' to a tuple of integers.

    Examples:
        '1-5' -> (0, 5)
        '3-7' -> (2, 7)

    Args:
        range_str: Range string in format 'start-end'

    Returns:
        Tuple of (start, end) as zero-indexed integers

    Raises:
        ValueError: If the range string is invalid
    """
    if not range_str:
        raise ValueError("Range string cannot be empty")

    parts = range_str.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid range format: {range_str}. Expected format: 'start-end'"
        )

    try:
        # Convert to zero-indexed integers
        start = int(parts[0]) - 1
        end = int(parts[1])

        if start < 0:
            raise ValueError(f"Start index must be at least 1, got {start + 1}")
        if end <= start:
            raise ValueError("End index must be greater than start index")

        return start, end
    except ValueError as e:
        if "invalid literal for int" in str(e):
            raise ValueError(f"Range values must be integers: {range_str}")
        raise


def prepare_examples_for_prompt(similar_cases: List[Dict[str, Any]]) -> str:
    """
    Prepare examples for the SQL query prompt.

    Args:
        similar_cases: List of similar cases from the dataset

    Returns:
        Formatted string with example queries and SQL responses
    """
    if not similar_cases:
        logger.warning("No similar cases provided to prepare examples")
        return ""

    examples = []

    for case in similar_cases:
        input_text = case.get("inputs", "")
        expected_output = case.get("expected_output", {})
        sql_query = expected_output.get("sql_query", "")

        if input_text and sql_query:
            example = f'User question: "{input_text}"\n'
            example += "Generated query:\n\n```sql\n"
            example += f"{sql_query}\n"
            example += "```\n"
            examples.append(example)

    return "\n".join(examples)
