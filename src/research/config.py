"""
Configuration settings for the GZM package.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv(override=True)


def _require_env(var: str) -> str:
    value = os.environ.get(var)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable '{var}' is not set.")
    return value


class LLMConfig(BaseModel):
    """LLM configuration settings.
    - provider: 'api' (default) or 'ollama'
    - base_url: only needed for ollama or custom endpoints
    """

    api_key: str = Field(default_factory=lambda: _require_env("GEMINI_API_KEY"))
    model: str = Field(default_factory=lambda: _require_env("GEMINI_MODEL"))
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "api"))
    base_url: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_BASE_URL"))


class EmbeddingConfig(BaseModel):
    """Embedding model configuration settings."""

    model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    default_path: str = Field(default_factory=lambda: _require_env("GZM_DATABASE"))


class EvalConfig(BaseModel):
    """Evaluation configuration settings."""

    dataset_path: str = Field(
        default_factory=lambda: os.getenv("GZM_EVAL_DATASET", "src/eval_dataset.json")
    )
    rate_limit: int = Field(
        default_factory=lambda: int(os.getenv("GZM_RATE_LIMIT", "0"))
    )
    max_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("GZM_MAX_CONCURRENCY", "5"))
    )


class Config(BaseModel):
    """
    Main configuration for the GZM package.

    Required (must be set in .env or environment):
      - GEMINI_API_KEY
      - GEMINI_MODEL
      - GZM_DATABASE
    Optional (defaulted, can be overridden in .env):
      - GZM_EVAL_DATASET
      - GZM_SQL_PROMPT_TEMPLATE
      - GZM_SPSS_SURVEY_METADATA
      - GZM_SURVEY_METADATA
      - GZM_DATA_DIR
      - GZM_RATE_LIMIT
      - GZM_MAX_CONCURRENCY
      - LLM_TEMPERATURE
      - LLM_MAX_TOKENS
      - EMBEDDING_MODEL
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # Paths (optional, with defaults)
    root_dir: Path = Path(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("GZM_DATA_DIR", "data"))
    )
    sql_prompt_template: str = Field(
        default_factory=lambda: os.getenv(
            "GZM_SQL_PROMPT_TEMPLATE",
            "src/research/llm/prompts/sql_query_prompt.jinja2",
        )
    )
    # used while loading SPSS survey metadata only
    spss_survey_metadata: str = Field(
        default_factory=lambda: os.getenv(
            "GZM_SPSS_SURVEY_METADATA",
            "src/research/data_preprocessing/survey_metadata_spss.json",
        )
    )
    survey_metadata: str = Field(
        default_factory=lambda: os.getenv(
            "GZM_SURVEY_METADATA",
            "src/survey_metadata_queries_simple.md",
        )
    )
    # Semantic search paths
    faiss_index_path: str = Field(
        default_factory=lambda: os.getenv(
            "GZM_FAISS_INDEX_PATH", "src/research/llm/faiss.index"
        )
    )
    examples_path: str = Field(
        default_factory=lambda: os.getenv(
            "GZM_EXAMPLES_PATH", "src/research/llm/examples.joblib"
        )
    )
    project_name: str = "gzm"


@lru_cache()
def get_config() -> Config:
    """
    Get configuration settings from environment variables only.
    Returns:
        Config object with configuration settings
    """
    return Config()
