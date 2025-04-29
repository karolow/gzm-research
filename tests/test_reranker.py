#!/usr/bin/env python3
"""
Test script for the Jina AI reranker functionality.
"""

import json
import os
import sys
from pathlib import Path

import click
import dotenv
import requests

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.llm.reranker import (
    JinaReranker,
    find_similar_examples,
    prepare_examples_for_prompt,
)
from research.utils.logger import setup_logger

# Load environment variables
dotenv.load_dotenv(override=True)


@click.command()
@click.option("--query", "-q", help="Query to find similar examples for", required=True)
@click.option(
    "--dataset",
    "-d",
    help="Path to evaluation dataset",
    default="src/eval_dataset.json",
    type=click.Path(exists=True),
)
@click.option(
    "--top-n", "-n", help="Number of examples to return", default=10, type=int
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with detailed logging",
    default=False,
)
@click.option(
    "--api-key", help="Jina AI API key (defaults to JINA_API_KEY env variable)"
)
@click.option(
    "--direct-test",
    is_flag=True,
    help="Test the API directly with a simple request",
    default=False,
)
@click.option(
    "--exclude-range",
    help="Range of cases to exclude from reranking (format: 'start-end', 1-indexed)",
    type=str,
)
def main(
    query: str,
    dataset: str,
    top_n: int,
    debug: bool,
    api_key: str,
    direct_test: bool,
    exclude_range: str | None,
) -> None:
    """
    Test the reranker functionality by finding similar examples for a query.
    """
    # Configure logging based on debug flag
    log_level = "DEBUG" if debug else "INFO"
    logger = setup_logger("test_reranker", log_level)

    # Get API key from arguments or environment
    api_key = api_key or os.getenv("JINA_API_KEY")
    if not api_key:
        logger.error(
            "JINA_API_KEY not found in environment variables or command line arguments"
        )
        sys.exit(1)

    logger.info(f"Finding top {top_n} examples similar to: {query}")

    # Direct test of the API, bypassing the main functionality
    if direct_test:
        try:
            logger.info("Testing direct API call to Jina AI reranker")
            url = "https://api.jina.ai/v1/rerank"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            data = {
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "top_n": 3,
                "documents": [
                    "User participation in cultural events",
                    "Demographics of concert attendees",
                    "Use of Instagram among residents",
                ],
                "return_documents": True,
            }

            logger.debug(f"Request data: {json.dumps(data, indent=2)}")

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                logger.info("Direct API test successful")
                logger.debug(f"Response: {json.dumps(result, indent=2)}")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")

            # Exit after direct test
            return

        except Exception as e:
            logger.error(f"Error in direct API test: {e}", exc_info=debug)
            sys.exit(1)

    try:
        # Test creating the reranker directly
        reranker = JinaReranker(api_key)

        # Load some examples to test
        with open(dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
            sample_documents = [
                case.get("inputs", "")
                for case in data.get("cases", [])[:10]
                if case.get("inputs")
            ]

        if sample_documents:
            logger.info(
                f"Testing direct reranking with {len(sample_documents)} documents"
            )
            ranked_documents = reranker.rerank(
                query, sample_documents, top_n=min(3, len(sample_documents))
            )
            logger.info(f"Direct reranking result count: {len(ranked_documents)}")

            if ranked_documents:
                for i, (doc, score) in enumerate(ranked_documents):
                    logger.info(f"Result {i + 1}: {score:.4f} - {doc}")

        # Find similar examples using the main function
        logger.info("Testing find_similar_examples function")
        similar_cases = find_similar_examples(
            query=query,
            eval_dataset_path=dataset,
            top_n=top_n,
            api_key=api_key,
            exclude_range=exclude_range,
        )

        if not similar_cases:
            logger.warning("No similar cases found")
            return

        # Convert to examples format
        examples_text = prepare_examples_for_prompt(similar_cases)

        # Print the examples
        print("\nFormatted examples for prompt:\n")
        print(examples_text)

        # Save to a file for inspection
        output_file = Path(f"reranker_examples_{top_n}.txt")
        output_file.write_text(examples_text)
        logger.info(f"Examples saved to {output_file.resolve()}")

    except Exception as e:
        logger.error(f"Error testing reranker: {e}", exc_info=debug)
        raise


if __name__ == "__main__":
    main()
