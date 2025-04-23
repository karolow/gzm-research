# GZM Survey Analysis

This project provides tools for analyzing survey data of GZM and Katowice residents regarding cultural participation.

## Features

- **Database Operations**: DuckDB operations for storing and querying survey data
- **Natural Language Queries**: Convert natural language to SQL using Google's Gemini AI
- **Evaluation Framework**: Comprehensive evaluation of SQL query generation performance

## Requirements

- Python 3.13+
- Dependencies listed in pyproject.toml

## Configuration and Environment Variables

This project uses a robust, config-driven pattern for all settings. All configuration is loaded from environment variables (typically via a `.env` file) using `src/research/config.py`. CLI arguments can override config values, but there are no hardcoded defaults in the codebase.

### Required Environment Variables (must be set in `.env` or environment):
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: The Gemini model to use (e.g., `gemini-2.0-flash`)
- `GZM_DATABASE`: Path to your DuckDB database file

### Optional Environment Variables (with defaults):
- `GZM_EVAL_DATASET` (default: `src/eval_dataset.json`)
- `GZM_SQL_PROMPT_TEMPLATE` (default: `src/research/llm/prompts/sql_query_prompt.jinja2`)
- `GZM_SPSS_SURVEY_METADATA` (default: `src/research/data_preprocessing/survey_metadata_spss.json`)
- `GZM_SURVEY_METADATA` (default: `src/survey_metadata_queries.json`)
- `GZM_DATA_DIR` (default: `data`)
- `GZM_RATE_LIMIT` (default: `0`)
- `GZM_MAX_CONCURRENCY` (default: `5`)
- `LLM_TEMPERATURE` (default: `0.0`)
- `LLM_MAX_TOKENS` (default: `1024`)

### How it works
- All scripts and package modules use `get_config()` from `src/research/config.py` to load settings.
- CLI arguments (e.g., `--database`, `--model`, etc.) override config values if provided.
- There are **no hardcoded defaults** in the codebase.
- `.env.example` provides a template for your `.env` file, but you can set variables however you like.

### Example: Setting environment variables
You can use a `.env` file (recommended for local development), or set variables in your shell or CI/CD environment:

**Using a .env file:**
```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GZM_DATABASE=research.db
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
GZM_EVAL_DATASET=src/eval_dataset.json
GZM_RATE_LIMIT=0
GZM_MAX_CONCURRENCY=5
GZM_DATA_DIR=data
GZM_SQL_PROMPT_TEMPLATE=src/research/llm/prompts/sql_query_prompt.jinja2
GZM_SPSS_SURVEY_METADATA=src/research/data_preprocessing/survey_metadata_spss.json
GZM_SURVEY_METADATA=src/survey_metadata_queries.json
```

**Or using shell export:**
```bash
export GEMINI_API_KEY=your_api_key_here
export GEMINI_MODEL=gemini-2.0-flash
export GZM_DATABASE=research.db
```

### Usage
- To run any CLI or evaluation script, ensure your `.env` is set up, or pass required variables as environment variables or CLI arguments.
- Example:
  ```bash
  uv venv
  uv sync
  cp .env.example .env  # and edit as needed
  llm-query --database my.db --model gemini-2.0-flash
  db --help
  evaluate --help
  ```

### Notes
- All evaluation scripts in `/evals` and all package code use this config pattern.
- If you want to tweak advanced LLM settings (e.g., `top_p`, `top_k`), edit the code that instantiates the LLM client. Defaults are set to Gemini API recommendations (0.95, 20).

## Usage

### Database Operations

```bash
# List tables in the database
db list-tables --database research.db

# Query the database
db query --database research.db --query "SELECT * FROM participation_survey LIMIT 10"

# Describe a table
db describe-table --database research.db --table participation_survey
```

### Natural Language Queries

```bash
# Ask a question in natural language
llm_query ask --database research.db --question "How many men and women participated in the survey?"

# Generate SQL without executing
llm-query generate --question "Show me the distribution of age groups"
```

### Running Evaluations

```bash
# Run evaluation on a set of test cases
evaluate --eval-cases src/eval_dataset.json --database research.db
```

