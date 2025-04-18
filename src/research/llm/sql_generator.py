"""
Module for generating SQL queries using LLM (Google's Gemini).
"""

import json
import re
from pathlib import Path
from typing import Optional

import jinja2

from research.config import get_config
from research.llm.gemini import GeminiClient
from research.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_metadata(metadata_path: str) -> str:
    """
    Load the survey metadata as a JSON string.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        JSON string representation of the metadata

    Raises:
        FileNotFoundError: If the metadata file doesn't exist
        json.JSONDecodeError: If the metadata file contains invalid JSON
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return json.dumps(metadata, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise


def create_prompt(metadata_json: str, template_path: str) -> str:
    """
    Create system instruction using the template and metadata.

    Args:
        metadata_json: JSON string containing survey metadata
        template_path: Path to Jinja2 template file

    Returns:
        System instruction string

    Raises:
        FileNotFoundError: If the template file doesn't exist
        jinja2.exceptions.TemplateError: If there's an error rendering the template
    """
    try:
        # Load template
        template_dir = Path(template_path).parent
        template_file = Path(template_path).name

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        template = env.get_template(template_file)

        # Render the system prompt with metadata
        system_instruction = template.render(METADATA_JSON=metadata_json)
        logger.debug(f"System instruction generated from template: {template_path}")
        return system_instruction

    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise
    except jinja2.exceptions.TemplateError as e:
        logger.error(f"Template error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise


def extract_sql_from_response(response_text: str) -> str:
    """
    Extract SQL query from LLM response text.

    Args:
        response_text: Text response from LLM

    Returns:
        Extracted SQL query
    """
    # Extract from code blocks if present
    if "```" in response_text:
        # Extract content between triple backticks
        sql_blocks = response_text.split("```")
        # If there are multiple code blocks, find the SQL one
        for i in range(1, len(sql_blocks), 2):
            block = sql_blocks[i]
            # Remove 'sql' or other language identifiers if present
            if block.lower().startswith("sql"):
                block = re.sub(r"^sql\s*", "", block, flags=re.IGNORECASE).strip()

            # Process the block for comment replacement
            sql_query = replace_hash_comments(block)
            if sql_query:
                return sql_query

    # If no code blocks found or they're empty, process the full response
    if response_text:
        return replace_hash_comments(response_text)

    return ""


def replace_hash_comments(text: str) -> str:
    """
    Replace # comments with -- comments for DuckDB compatibility.

    Args:
        text: SQL query text

    Returns:
        SQL query with standardized comments
    """
    sql_query = ""
    for line in text.split("\n"):
        comment_pos = line.find("#")
        if comment_pos >= 0:
            line = line[:comment_pos] + "--" + line[comment_pos + 1 :]
        sql_query += line + "\n"
    return sql_query.strip()


def natural_language_to_sql(
    question: str,
    metadata_path: Optional[str] = None,
    template_path: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    verbose: bool = False,
) -> str:
    """
    Convert a natural language question to a SQL query using LLM.

    Args:
        question: Natural language question
        metadata_path: Path to the survey metadata JSON file
        template_path: Path to the prompt template
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        verbose: Enable verbose logging

    Returns:
        Generated SQL query
    """
    if verbose:
        logger.setLevel("DEBUG")

    # Get configuration
    config = get_config()

    # Use provided paths or defaults from config
    metadata_path = metadata_path or config.survey_metadata
    template_path = template_path or config.sql_prompt_template
    temperature = temperature if temperature is not None else config.llm.temperature
    max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

    # Load metadata
    metadata_json = load_metadata(metadata_path)
    logger.debug(f"Loaded metadata from {metadata_path}")

    # Create the system instruction
    system_instruction = create_prompt(metadata_json, template_path)

    # Initialize Gemini client
    client = GeminiClient(
        api_key=config.llm.api_key,
        model=config.llm.model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Generate SQL query
    logger.debug(f"Generating SQL query for question: {question}")
    response = client.generate_content(system_instruction, question)

    # Extract SQL from response
    sql_query = extract_sql_from_response(response)

    if not sql_query:
        logger.warning("Failed to extract SQL query from model response")

    return sql_query
