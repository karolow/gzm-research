#!/usr/bin/env python
"""
Build a FAISS index from examples for semantic search functionality.

This script creates a FAISS index for fast similarity search based on embeddings
of example queries or cases. The index is stored in the same directory as this file.

Usage:
    uv run src/research/llm/build_faiss_index.py --examples path/to/examples.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.research.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Load environment variables
load_dotenv(override=True)


def get_openai_embedding(text: str) -> np.ndarray:
    """Create embedding using OpenAI API."""
    from openai import OpenAI

    api_key = config.embedding.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=config.embedding.model, input=[text])
        emb = np.array(response.data[0].embedding, dtype="float32")
        return emb
    except Exception as e:
        logger.error(f"Error getting OpenAI embedding: {e}")
        raise


def build_index(examples: list[dict[str, Any] | str], output_dir: Path = None) -> None:
    """
    Build a FAISS index from examples and save it to disk.

    Args:
        examples: List of examples (can be dictionaries with 'inputs' key, strings, or other formats)
        output_dir: Directory to save the index and examples, defaults to directory from config
    """
    if output_dir is None:
        # Use the directory part of the configured paths
        output_dir = Path(os.path.dirname(config.faiss_index_path))

    # Extract text from examples, handling different formats
    texts = []
    logger.info(f"Processing {len(examples)} examples...")

    # Check the structure of the first example to determine format
    if examples and isinstance(examples[0], dict):
        # Dictionary format - look for common text fields
        if "inputs" in examples[0]:
            logger.info("Detected dictionary format with 'inputs' field")
            texts = [ex.get("inputs", "") for ex in examples]
        elif "question" in examples[0]:
            logger.info("Detected dictionary format with 'question' field")
            texts = [ex.get("question", "") for ex in examples]
        elif "query" in examples[0]:
            logger.info("Detected dictionary format with 'query' field")
            texts = [ex.get("query", "") for ex in examples]
        elif "text" in examples[0]:
            logger.info("Detected dictionary format with 'text' field")
            texts = [ex.get("text", "") for ex in examples]
        else:
            # Try to find a string value in the dict
            keys = list(examples[0].keys())
            if keys and isinstance(examples[0][keys[0]], str):
                logger.info(f"Using first string field: '{keys[0]}'")
                texts = [ex.get(keys[0], "") for ex in examples]
            else:
                logger.warning(
                    "Could not determine text field in dictionary, using string representation"
                )
                texts = [str(ex) for ex in examples]
    elif examples and isinstance(examples[0], str):
        # Simple string format
        logger.info("Detected string format")
        texts = examples
    else:
        # Fallback to string representation
        logger.warning("Unknown example format, using string representation")
        texts = [str(ex) for ex in examples]

    logger.info(f"Creating embeddings for {len(texts)} examples...")

    # Create embeddings for all texts
    embeddings = []
    for text in tqdm(texts, desc="Creating embeddings"):
        emb = get_openai_embedding(text)
        embeddings.append(emb)

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype="float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)

    # Create FAISS index
    logger.info("Building FAISS index...")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(
        dim
    )  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings_array)

    # Save index and examples
    index_path = (
        Path(config.faiss_index_path)
        if os.path.basename(config.faiss_index_path)
        else output_dir / os.path.basename(config.faiss_index_path)
    )
    examples_path = (
        Path(config.examples_path)
        if os.path.basename(config.examples_path)
        else output_dir / os.path.basename(config.examples_path)
    )

    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    logger.info(f"Saving examples to {examples_path}")
    joblib.dump(examples, str(examples_path))

    logger.info(f"Done! Index built with {len(examples)} examples")


def load_examples(examples_path: str) -> list[dict[str, Any] | str]:
    """
    Load examples from a JSON file.

    The file can contain:
    - A list of dictionaries with text fields
    - A list of strings
    - A dictionary with lists or objects
    """
    with open(examples_path, "r") as f:
        data = json.load(f)

    # Handle different data structures
    if isinstance(data, list):
        # Already a list of examples
        return data
    elif isinstance(data, dict):
        # Try to extract examples from dictionary
        # Look for lists in the dict that might contain the examples
        for key, value in data.items():
            if isinstance(value, list) and value:
                logger.info(f"Using examples from key '{key}'")
                return value

        # If no lists found, use the dict keys as examples
        logger.info("Using dictionary keys as examples")
        return list(data.keys())
    else:
        # Unexpected format
        raise ValueError(f"Unsupported data format in {examples_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for semantic search"
    )
    parser.add_argument(
        "--examples", type=str, required=True, help="Path to JSON file with examples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the index and examples (default: directory from config)",
    )

    args = parser.parse_args()

    examples = load_examples(args.examples)
    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info(f"Building index from {len(examples)} examples")
    build_index(examples, output_dir)


if __name__ == "__main__":
    main()
