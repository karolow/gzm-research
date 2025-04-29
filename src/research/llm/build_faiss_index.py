#!/usr/bin/env python
"""
Build or update a FAISS index from examples for semantic search functionality.

This script creates or updates a FAISS index for fast similarity search based on embeddings
of example queries or cases. The index is stored in the specified directory.

Usage:
    # Build a new index:
    uv run src/research/llm/build_faiss_index.py --examples path/to/examples.json

    # Update an existing index:
    uv run src/research/llm/build_faiss_index.py --examples path/to/new_examples.json --update
"""

import argparse
import json
import logging
import os
from pathlib import Path

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


def extract_texts_from_examples(examples: list[dict | str]) -> list[str]:
    """
    Extract text from examples, expecting only the 'inputs' field.

    Args:
        examples: List of examples (dictionaries with 'inputs' key or strings)

    Returns:
        List of text strings extracted from examples
    """
    texts = []

    if not examples:
        return texts

    # Check the structure of examples
    if isinstance(examples[0], dict):
        # Dictionary format - only use 'inputs' field
        texts = [ex.get("inputs", "") for ex in examples]
        logger.info("Extracting text from 'inputs' field")

        # Check if any texts are empty
        empty_count = sum(1 for t in texts if not t)
        if empty_count > 0:
            logger.warning(f"{empty_count} examples missing 'inputs' field")
    elif isinstance(examples[0], str):
        # Simple string format
        logger.info("Examples are already in string format")
        texts = examples
    else:
        # Fallback to string representation
        logger.warning("Unknown example format, using string representation")
        texts = [str(ex) for ex in examples]

    return texts


def create_embeddings(texts: list[str]) -> np.ndarray:
    """
    Create embeddings for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        Normalized numpy array of embeddings
    """
    logger.info(f"Creating embeddings for {len(texts)} texts...")

    # Create embeddings for all texts
    embeddings = []
    for text in tqdm(texts, desc="Creating embeddings"):
        emb = get_openai_embedding(text)
        embeddings.append(emb)

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype="float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)

    return embeddings_array


def build_index(
    examples: list[dict | str], output_dir: Path = None, update: bool = False
) -> None:
    """
    Build or update a FAISS index from examples and save it to disk.

    Args:
        examples: List of examples (dictionaries with 'inputs' key or strings)
        output_dir: Directory to save the index and examples, defaults to directory from config
        update: Whether to update an existing index or create a new one
    """
    if output_dir is None:
        # Use the directory part of the configured paths
        output_dir = Path(os.path.dirname(config.faiss_index_path))

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    index_path = (
        Path(config.faiss_index_path)
        if os.path.isabs(config.faiss_index_path)
        else output_dir / os.path.basename(config.faiss_index_path)
    )
    examples_path = (
        Path(config.examples_path)
        if os.path.isabs(config.examples_path)
        else output_dir / os.path.basename(config.examples_path)
    )

    # Extract texts from examples
    texts = extract_texts_from_examples(examples)

    if not texts:
        logger.error("No valid texts found in examples")
        return

    # Create embeddings for current examples
    embeddings_array = create_embeddings(texts)
    dim = embeddings_array.shape[1]

    # If updating, load the existing index and examples
    existing_examples = []
    if update and index_path.exists() and examples_path.exists():
        logger.info(f"Loading existing index from {index_path}")
        try:
            index = faiss.read_index(str(index_path))
            existing_examples = joblib.load(str(examples_path))
            logger.info(f"Loaded index with {index.ntotal} vectors")

            # Add new vectors to the existing index
            logger.info(f"Adding {len(examples)} new examples to index")
            index.add(embeddings_array)

            # Merge examples
            existing_examples.extend(examples)

        except Exception as e:
            logger.error(f"Error updating index: {e}")
            logger.warning("Creating a new index instead")
            update = False

    # Create a new index if not updating or if update failed
    if not update:
        logger.info("Building new FAISS index...")
        index = faiss.IndexFlatIP(
            dim
        )  # Inner product for cosine similarity with normalized vectors
        index.add(embeddings_array)
        existing_examples = examples

    # Save index and examples
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    logger.info(f"Saving examples to {examples_path}")
    joblib.dump(existing_examples, str(examples_path))

    logger.info(
        f"Done! Index now has {index.ntotal} vectors from {len(existing_examples)} examples"
    )


def load_examples(examples_path: str) -> list[dict | str]:
    """
    Load examples from a JSON file.

    The file can contain:
    - A list of dictionaries with 'inputs' field
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
        description="Build or update FAISS index for semantic search"
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
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update an existing index with new examples",
    )

    args = parser.parse_args()

    examples = load_examples(args.examples)
    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info(f"Processing {len(examples)} examples")
    build_index(examples, output_dir, args.update)


if __name__ == "__main__":
    main()
