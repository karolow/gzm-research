# GZM Survey Analysis

This project provides tools for analyzing survey data of GZM and Katowice residents regarding cultural participation.

## Features

- **Database Operations**: DuckDB operations for storing and querying survey data
- **Natural Language Queries**: Convert natural language to SQL using Google's Gemini AI
- **Evaluation Framework**: Comprehensive evaluation of SQL query generation performance
- **Semantic Search**: Find semantically similar examples using FAISS and text embeddings

## Requirements

- Python 3.13+
- Dependencies listed in pyproject.toml

## Configuration and Environment Variables

This project uses a robust, config-driven pattern for all settings. All configuration is loaded from environment variables (typically via a `.env` file) using `src/research/config.py`. CLI arguments can override config values, but there are no hardcoded defaults in the codebase.

### Required Environment Variables (must be set in `.env` or environment):
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: The Gemini model to use (e.g., `gemini-2.0-flash`)
- `GZM_DATABASE`: Path to your DuckDB database file
- `OPENAI_API_KEY`: Your OpenAI API key (required for semantic search embeddings)

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
- `JINA_API_KEY`: API key for Jina AI (optional, for enhanced reranking)
- `EMBEDDING_MODEL` (default: `text-embedding-3-large`)
- `GZM_FAISS_INDEX_PATH` (default: `src/research/llm/faiss.index`)
- `GZM_EXAMPLES_PATH` (default: `src/research/llm/examples.joblib`)

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
export OPENAI_API_KEY=your_openai_api_key_here
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
  evals --help
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
evals evaluate --eval-cases src/eval_dataset.json --database research.db --model gemini-2.0-flash

# Configure the evaluation settings
evals config show

# Ask a single SQL question without evaluation
evals ask "How many men and women participated in the survey?" --database research.db
```

## Semantic Search

The project includes semantic search functionality that lets you find similar examples using vector similarity. This is useful for finding similar queries, identifying patterns in survey data, or implementing RAG (Retrieval-Augmented Generation) approaches.

### Building the FAISS Index

Before using semantic search, you need to build a FAISS index from your examples:

1. Prepare a JSON file with your examples. The script supports multiple formats:
   
   - List of dictionaries with text fields (it will automatically detect common fields like "inputs", "question", "query", etc.):
     ```json
     [
       {"inputs": "How many people participated in cultural events?"},
       {"inputs": "What's the age distribution of museum visitors?"}
     ]
     ```

   - Simple list of strings:
     ```json
     [
       "How many people participated in cultural events?",
       "What's the age distribution of museum visitors?"
     ]
     ```

   - Dictionary with nested lists:
     ```json
     {
       "examples": [
         "How many people participated in cultural events?",
         "What's the age distribution of museum visitors?"
       ]
     }
     ```

2. Build the index:
   ```bash
   uv run src/research/llm/build_faiss_index.py --examples src/eval_dataset.json
   ```

   This will create two files:
   - `src/research/llm/faiss.index`: The FAISS index for fast similarity search
   - `src/research/llm/examples.joblib`: The serialized examples data

### Using Semantic Search

Once the index is built, you can use the semantic search functionality in your code:

```python
from research.llm.semantic_search import semantic_search, semantic_rerank

# Simple search - returns indices of similar examples
indices = semantic_search("How many women visited museums?", top_k=5)

# Reranked search - returns the actual example dictionaries, reranked for better relevance
similar_cases = semantic_rerank("How many women visited museums?", top_n=5)

# Print the similar cases
for case in similar_cases:
    if isinstance(case, dict):
        print(f"Query: {case.get('inputs', str(case))}")
    else:
        print(f"Query: {case}")
    print("-" * 50)
```

### Customizing the Search

- You can specify `exclude_indices` to exclude certain examples from search results
- The `JinaReranker` class provides reranking capabilities to improve result relevance
