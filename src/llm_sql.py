#!/usr/bin/env python3
"""
Module for generating SQL queries using LLM (Google's Gemini).
"""

import json
import logging
import os
from pathlib import Path

import jinja2
from dotenv import load_dotenv
from google import genai
from google.genai import types


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_gemini_client() -> tuple[genai.Client, str]:
    """
    Load the Gemini client and model specified in the environment.

    Returns:
        tuple: (Gemini client, model name)
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)

    logging.info(f"Using model: {model_name}")
    return client, model_name


def load_metadata(metadata_path: str) -> str:
    """Load the survey metadata as a JSON string."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return json.dumps(metadata, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        raise


def create_prompt(metadata_json: str, template_path: str) -> str:
    """
    Create system instruction and user prompt using the template and metadata.

    Returns:
        tuple: (system_instruction, user_question)
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
        logging.info(f"System instruction:\n{system_instruction}")
        return system_instruction

    except Exception as e:
        logging.error(f"Error creating prompt: {e}")
        raise


def generate_sql_query(
    system_instruction: str, user_question: str, client: genai.Client, model_name: str
) -> str:
    """Generate a SQL query using the Gemini model."""
    try:
        # Send the request to the Gemini API with system instruction
        response = client.models.generate_content(
            model=model_name,
            contents=user_question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                top_p=0.95,
                top_k=20,
                max_output_tokens=1024,
            ),
        )

        # Extract the SQL query from the response
        response_text = response.text if response and hasattr(response, "text") else ""

        # Extract SQL query from markdown code block if present
        if response_text and "```" in response_text:
            # Extract content between triple backticks
            sql_blocks = response_text.split("```")
            # If there are multiple code blocks, find the SQL one
            for i in range(1, len(sql_blocks), 2):
                block = sql_blocks[i]
                # Remove 'sql' or other language identifiers if present
                if block.startswith("sql"):
                    block = block[3:].strip()

                # Replace # comments with -- comments for DuckDB compatibility
                sql_query = ""
                for line in block.split("\n"):
                    comment_pos = line.find("#")
                    if comment_pos >= 0:
                        line = line[:comment_pos] + "--" + line[comment_pos + 1 :]
                    sql_query += line + "\n"

                return sql_query.strip()

        # If no code blocks found, process the full response for comment replacement
        if response_text:
            sql_query = ""
            for line in response_text.split("\n"):
                comment_pos = line.find("#")
                if comment_pos >= 0:
                    line = line[:comment_pos] + "--" + line[comment_pos + 1 :]
                sql_query += line + "\n"
            return sql_query.strip()

        return ""
    except Exception as e:
        logging.error(f"Error generating SQL query: {e}")
        raise


def natural_language_to_sql(
    question: str, metadata_path: str, template_path: str, verbose: bool = False
) -> str:
    """
    Convert a natural language question to a SQL query using LLM.

    Args:
        question: Natural language question
        metadata_path: Path to the survey metadata JSON file
        template_path: Path to the prompt template
        verbose: Enable verbose logging

    Returns:
        Generated SQL query
    """
    setup_logging(verbose)

    # Load the model
    client, model_name = load_gemini_client()

    # Load metadata
    metadata_json = load_metadata(metadata_path)

    # Create the system instruction and user prompt
    system_instruction = create_prompt(metadata_json, template_path)

    if verbose:
        logging.debug("Prompt created, generating SQL query...")

    # Generate the SQL query
    sql_query = generate_sql_query(system_instruction, question, client, model_name)

    return sql_query
