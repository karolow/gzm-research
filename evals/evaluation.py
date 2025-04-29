import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import click
import dotenv
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from rich.console import Console
from rich.table import Table

from research.config import get_config
from research.db.operations import SQLExecutionError, SQLParsingError, query_duckdb
from research.utils.logger import setup_logger

# Load environment variables
dotenv.load_dotenv(override=True)

logger = setup_logger("evaluation")
logfire.configure()


class Result(BaseModel):
    sql_query: str | None
    result: str


@dataclass
class ResultWeighted(Evaluator):
    def evaluate(self, ctx: EvaluatorContext[str, Result]) -> bool:
        return ctx.output.sql_query is not None and "waga_proby" in ctx.output.sql_query


@dataclass
class ResultEquals(Evaluator):
    def extract_values(self, text: str | Result) -> list[str]:
        # Handle Result objects by extracting the result string
        if isinstance(text, Result):
            text = text.result

        # Split by newlines and then by separator (pipe or comma)
        all_values: list[str] = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Skip the first line (header) if there are multiple lines
        data_lines = lines[1:] if len(lines) > 1 else lines

        for line in data_lines:
            # First split by pipe which is the primary column separator
            if "|" in line:
                columns = [col.strip() for col in line.split("|")]
                for col in columns:
                    # Check if this column has a format like "text,number"
                    if "," in col and any(c.isdigit() for c in col):
                        # In cases like "value1,12.34", extract both parts separately
                        parts = [part.strip() for part in col.split(",")]
                        all_values.extend(parts)
                    else:
                        all_values.append(col)
            # If no pipe, try comma as separator
            elif "," in line:
                all_values.extend([v.strip() for v in line.split(",")])
            else:
                all_values.append(line.strip())

        return all_values

    def extract_numeric_values(self, text: str | Result) -> list[str]:
        """Extract only numeric values from the result."""
        values = self.extract_values(text)
        # Filter to keep only numeric values
        numeric_values: list[str] = []
        for val in values:
            # Try to determine if it's a numeric value
            # This handles both integers and floats
            val = val.strip()
            try:
                float(val)  # Check if convertible to float
                numeric_values.append(val)
            except ValueError:
                pass

        return numeric_values

    async def evaluate(self, ctx: EvaluatorContext[str, Result]) -> float:
        # Handle empty strings and None values
        if not ctx.expected_output or not ctx.output:
            return 0.0

        # Extract numeric values only
        expected_numbers = self.extract_numeric_values(ctx.expected_output)
        output_numbers = self.extract_numeric_values(ctx.output)

        logger.debug(f"Expected numeric values: {expected_numbers}")
        logger.debug(f"Output numeric values: {output_numbers}")

        # If no values to check, return 0
        if not expected_numbers:
            return 0.0

        # Count matches - numeric values should match exactly or very closely
        matches = 0
        for expected_num in expected_numbers:
            # Try with exact matches first, then with close matches for floating point
            if any(expected_num == output_num for output_num in output_numbers):
                matches += 1
            else:
                # Try with close match for floating point
                try:
                    expected_float = float(expected_num)
                    for output_num in output_numbers:
                        try:
                            output_float = float(output_num)
                            # Allow small difference (0.01 or 0.1%)
                            if abs(expected_float - output_float) < max(
                                0.01, abs(expected_float * 0.001)
                            ):
                                matches += 1
                                break
                        except ValueError:
                            continue
                except ValueError:
                    continue

        # Calculate base match ratio for correct numbers
        if len(expected_numbers) > 0:
            match_ratio = matches / len(expected_numbers)
        else:
            match_ratio = 0.0

        # Penalize for extra numbers (noise)
        # If we have more numbers than expected, reduce the score
        if len(output_numbers) > len(expected_numbers):
            extra_numbers = len(output_numbers) - len(expected_numbers)
            # Penalty grows with the number of extra values, but caps at 0.5
            penalty = min(0.5, extra_numbers / len(expected_numbers) * 0.5)
            match_ratio *= 1 - penalty

        # Bonus for exact match of all numeric values and count
        if matches == len(expected_numbers) and len(output_numbers) == len(
            expected_numbers
        ):
            match_ratio = 1.0

        return match_ratio


# Rate limiter class to control LLM API calls
class RateLimiter:
    """Limits the rate of API calls using a token bucket algorithm."""

    def __init__(self, rate_limit_per_minute: int = 60):
        """
        Initialize the rate limiter.

        Args:
            rate_limit_per_minute: Maximum number of requests per minute
        """
        self.rate_limit = rate_limit_per_minute
        self.tokens = rate_limit_per_minute
        self.last_refill_time = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

        # Calculate token refill rate (tokens per second)
        self.refill_rate = rate_limit_per_minute / 60.0

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        while True:
            async with self.lock:
                # Refill tokens based on time elapsed
                current_time = asyncio.get_event_loop().time()
                time_elapsed = current_time - self.last_refill_time
                new_tokens = time_elapsed * self.refill_rate

                if new_tokens > 0:
                    self.tokens = min(self.rate_limit, self.tokens + new_tokens)
                    self.last_refill_time = current_time

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            # If no tokens available, wait a bit before trying again
            wait_time = (
                1 / self.refill_rate
            ) * 0.8  # Wait for ~80% of the time to get a new token
            logger.debug("Rate limit reached, waiting for next available token")
            await asyncio.sleep(wait_time)


def parse_range(range_str: str) -> tuple[int, int]:
    """
    Parse a range string in format 'start-end' to a tuple of integers.

    Examples:
        '1-5' -> (1, 5)
        '3-7' -> (3, 7)

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


def filter_cases_by_range(
    cases: Sequence[Case], case_range: Optional[str] = None
) -> list[Case]:
    """
    Filter cases by the specified range.

    Args:
        cases: The sequence of cases to filter
        case_range: Optional range string in format 'start-end' (1-indexed)

    Returns:
        Filtered list of cases
    """
    if not case_range:
        return list(cases)

    try:
        start_idx, end_idx = parse_range(case_range)
        # Ensure indices are within bounds
        if start_idx >= len(cases):
            logger.warning(
                f"Start index {start_idx + 1} exceeds number of cases ({len(cases)})"
            )
            return []

        end_idx = min(end_idx, len(cases))
        selected_cases = list(cases[start_idx:end_idx])
        logger.info(
            f"Selected cases {start_idx + 1} to {end_idx} (out of {len(cases)} total cases)"
        )
        return selected_cases
    except ValueError as e:
        logger.error(f"Invalid range: {e}")
        return list(cases)


def export_report_to_markdown(
    report: Any, table: Table, model_name: str, output_dir: Optional[str | Path] = None
) -> Path:
    """
    Export evaluation report to a markdown file with timestamp and model name.

    Args:
        report: The evaluation report object
        table: The rich table object representing the report
        model_name: Name of the model used for evaluation
        output_dir: Optional directory path for output file (defaults to current dir)

    Returns:
        Path object pointing to the saved file
    """
    from io import StringIO

    # Create a string buffer and console to capture the table output
    string_io = StringIO()
    markdown_console = Console(file=string_io, width=120)
    markdown_console.print(table)
    table_output = string_io.getvalue()

    # Generate timestamped filename with model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filename (remove special characters)
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    filename = f"eval_report_{safe_model_name}_{timestamp}.md"

    # Create output path
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / filename
    else:
        output_path = Path(filename)

    # Create markdown content with model name and proper formatting
    markdown_content = f"""# Evaluation Report

    ## Model: {model_name}
    ## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    {table_output}
    """

    # Save to file
    output_path.write_text(markdown_content, encoding="utf-8")

    return output_path


# Create a Click group
@click.group()
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (can be used multiple times)",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    """
    SQL Evaluation Tool - Evaluate LLM's ability to generate SQL queries.

    Run 'eval_sql evaluate' to perform evaluation or 'eval_sql config' to manage configuration.
    """
    # Ensure we have a context object
    ctx.ensure_object(dict)

    # Store verbose level in context
    ctx.obj["verbose"] = verbose

    # Set log level based on verbosity
    if verbose == 0:
        log_level = "INFO"
    elif verbose == 1:
        log_level = "DEBUG"
    else:
        log_level = "DEBUG"  # More verbose debug if needed

    # Initialize global logger
    global logger
    logger = setup_logger("evaluation", log_level)


# Evaluation options decorators grouped by category
def add_input_options(command):
    """Add input file options to a command."""
    decorators = [
        click.option(
            "--eval-cases",
            "-e",
            help="Path to evaluation cases JSON file",
            type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
            default=lambda: get_config().eval.dataset_path,
            show_default="from config",
            envvar="SQL_EVAL_CASES",
        ),
        click.option(
            "--metadata-json",
            "-m",
            help="Path to metadata JSON file: used in system prompt",
            type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
            default=lambda: get_config().survey_metadata,
            show_default="from config",
            envvar="SQL_METADATA_JSON",
        ),
        click.option(
            "--template-path",
            "-t",
            help="Path to prompt template file",
            type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
            default=lambda: get_config().sql_prompt_template,
            show_default="from config",
            envvar="SQL_TEMPLATE_PATH",
        ),
        click.option(
            "--database",
            "-d",
            help="Path to the database file",
            type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
            default=lambda: get_config().db.default_path,
            show_default="from config",
            envvar="SQL_DATABASE_PATH",
        ),
    ]

    # Apply all decorators to the command
    for decorator in decorators:
        command = decorator(command)
    return command


def add_model_options(command):
    """Add model configuration options to a command."""
    decorators = [
        click.option(
            "--model",
            "-M",
            help="LLM model to use",
            default=lambda: get_config().llm.model,
            show_default="from config",
            envvar="SQL_EVAL_MODEL",
        ),
        click.option(
            "--temperature",
            "-T",
            help="Temperature setting for the model",
            type=float,
            default=lambda: get_config().llm.temperature,
            show_default="from config",
            envvar="SQL_EVAL_TEMPERATURE",
        ),
        click.option(
            "--provider",
            help="LLM provider: 'api' (default) or 'ollama' (local)",
            default=lambda: get_config().llm.provider,
            show_default="from config",
            envvar="SQL_EVAL_PROVIDER",
        ),
        click.option(
            "--base-url",
            help="Base URL for LLM provider (only needed for ollama or custom endpoints)",
            default=lambda: get_config().llm.base_url,
            show_default="from config",
            envvar="SQL_EVAL_BASE_URL",
        ),
    ]

    for decorator in decorators:
        command = decorator(command)
    return command


def add_execution_options(command):
    """Add execution control options to a command."""
    decorators = [
        click.option(
            "--range",
            "-R",
            help="Range of cases to evaluate (format: 'start-end', 1-indexed, inclusive-exclusive)",
            type=str,
            envvar="SQL_EVAL_RANGE",
        ),
        click.option(
            "--rate-limit",
            "-r",
            help="Rate limit for LLM API calls (requests per minute, 0 = no limit)",
            type=int,
            default=lambda: get_config().eval.rate_limit,
            show_default="from config",
            envvar="SQL_EVAL_RATE_LIMIT",
        ),
        click.option(
            "--max-concurrency",
            "-c",
            help="Maximum number of concurrent evaluations (0 = unlimited)",
            type=int,
            default=lambda: get_config().eval.max_concurrency,
            show_default="from config",
            envvar="SQL_EVAL_MAX_CONCURRENCY",
        ),
    ]

    for decorator in decorators:
        command = decorator(command)
    return command


def add_reranker_options(command):
    """Add reranker and semantic search options to a command."""
    decorators = [
        click.option(
            "--use-reranker",
            is_flag=True,
            help="Use Jina AI reranker to refine examples for the prompt",
            default=False,
            envvar="SQL_USE_RERANKER",
        ),
        click.option(
            "--use-semantic-search",
            is_flag=True,
            help="Use FAISS semantic search to fetch examples for the prompt",
            default=False,
            envvar="SQL_USE_SEMANTIC_SEARCH",
        ),
        click.option(
            "--num-examples",
            type=int,
            help="Number of examples to include in the prompt (default: 10)",
            default=10,
            show_default=True,
            envvar="SQL_NUM_EXAMPLES",
        ),
    ]
    for decorator in decorators:
        command = decorator(command)
    return command


@cli.command()
@add_input_options
@add_model_options
@add_execution_options
@add_reranker_options
@click.option(
    "--output-dir",
    help="Directory to save evaluation reports",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    envvar="SQL_EVAL_OUTPUT_DIR",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    eval_cases: str,
    metadata_json: str,
    template_path: str,
    model: str,
    temperature: float,
    database: str,
    range: Optional[str],
    rate_limit: int,
    max_concurrency: int,
    provider: str,
    base_url: Optional[str],
    use_reranker: bool,
    use_semantic_search: bool,
    num_examples: int,
    output_dir: Optional[str],
) -> None:
    """
    Run model evaluation against SQL queries.

    This tool evaluates the model's ability to generate SQL queries
    from natural language questions and produce correct results.

    Examples:
      eval_sql evaluate --model gemini-2.0-flash --range 1-10
      eval_sql evaluate --database custom.db --use-reranker
      eval_sql evaluate -M claude-3-5-sonnet -T 0.2 -c 5
    """
    # Get verbosity level from context
    verbose = ctx.obj.get("verbose", 0)

    # Initialize context dictionary
    evaluation_ctx = {
        "eval_cases_path": eval_cases,
        "use_reranker": use_reranker,
        "use_semantic_search": use_semantic_search,
        "num_examples": num_examples,
        "metadata_json_path": metadata_json,
        "template_path": template_path,
        "range_filter": range,
        "rate_limiter": None,
    }

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(f"Using database at: {database}")
    logger.info(f"Loading evaluation cases from: {eval_cases}")

    if use_reranker:
        logger.info(
            f"Jina AI reranker enabled, using {num_examples} examples per query"
        )

    # Load metadata and SQL prompt template only when needed
    from research.llm.sql_generator import create_system_prompt, load_metadata

    try:
        # Show progress for data loading
        console = Console()
        with console.status("[bold green]Loading evaluation dataset...") as status:
            full_dataset = Dataset[str, Result, Any].from_file(
                eval_cases,
                fmt="json",
                custom_evaluator_types=[ResultWeighted, ResultEquals],
            )
            status.update(
                f"[bold green]Loaded {len(full_dataset.cases)} cases for evaluation"
            )

        logger.info(f"Loaded {len(full_dataset.cases)} cases for evaluation.")

        if range:
            filtered_cases = filter_cases_by_range(full_dataset.cases, range)
            if not filtered_cases:
                logger.error("No cases to evaluate after applying range filter")
                return
            dataset = Dataset[str, Result, Any](
                cases=filtered_cases,
                evaluators=full_dataset.evaluators,
            )
            logger.info(
                f"Evaluating {len(dataset.cases)} cases (filtered by range: {range})"
            )
        else:
            dataset = full_dataset
            logger.info(f"Evaluating all {len(dataset.cases)} cases")

    except FileNotFoundError:
        logger.error(f"Error: Evaluation cases file not found at {eval_cases}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {eval_cases}: {e}")
        return
    except ValueError as e:
        logger.error(f"Error loading dataset from {eval_cases}: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading cases: {e}")
        return

    # Initialize rate limiter if needed
    if rate_limit > 0:
        evaluation_ctx["rate_limiter"] = RateLimiter(rate_limit_per_minute=rate_limit)
        logger.info(f"Rate limiter enabled: {rate_limit} requests per minute")

    async def query_generator(
        question: str, rate_limiter: Optional[RateLimiter] = None
    ) -> Result:
        if rate_limiter is not None:
            await rate_limiter.acquire()

        examples_text = ""
        if evaluation_ctx["use_semantic_search"] or evaluation_ctx["use_reranker"]:
            try:
                # Fetch examples based on flags
                if evaluation_ctx["use_semantic_search"]:
                    from research.llm.semantic_search import (
                        _load_resources,
                        semantic_search,
                    )

                    _load_resources()
                    indices = semantic_search(
                        question, top_k=10
                    )  # Always fetch 10 for semantic search
                    from research.llm.semantic_search import _examples

                    similar_cases = [_examples[idx] for idx in indices]
                else:
                    # Fall back to reranker-only mode (no semantic search)
                    from research.llm.reranker import find_similar_examples

                    similar_cases = find_similar_examples(
                        query=question,
                        eval_dataset_path=evaluation_ctx["eval_cases_path"],
                        top_n=10,  # Fetch 10 for reranker if no semantic search
                    )

                # Rerank if enabled
                if evaluation_ctx["use_reranker"] and similar_cases:
                    from research.llm.reranker import rerank_examples

                    similar_cases = rerank_examples(question, similar_cases)

                # Limit to --num-examples
                similar_cases = similar_cases[: evaluation_ctx["num_examples"]]

                # Prepare examples for the prompt
                if similar_cases:
                    from research.llm.reranker import prepare_examples_for_prompt

                    examples_text = prepare_examples_for_prompt(similar_cases)
                    logger.debug(f"Added {len(similar_cases)} examples to the prompt")

            except Exception as e:
                logger.error(
                    f"Error retrieving examples (continuing without them): {e}"
                )

        # Create the prompt with or without examples
        system_prompt = create_system_prompt(
            load_metadata(evaluation_ctx["metadata_json_path"]),
            evaluation_ctx["template_path"],
            examples=examples_text,
        )

        if verbose >= 2:
            logger.debug(f"System prompt: {system_prompt}")

        # Dynamically create the model and agent for each question
        if provider == "ollama":
            model_instance = OpenAIModel(
                model_name=model,
                provider=OpenAIProvider(base_url=base_url),
            )
        else:
            model_instance = model

        agent = Agent(
            model_instance,
            output_type=str,
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=get_config().llm.max_tokens,
            ),
        )
        output = await agent.run(question)
        response = output.output

        # Robustly strip code fencing (```sql ... ```, with or without whitespace)
        import re

        sql_query = response.strip()
        # Remove any markdown code blocks (handles both ```sql and just ```)
        # First attempt to find and extract content from a full markdown code block
        code_block_pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, sql_query, re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()

        # In case there are still backticks at the beginning or end, remove them
        sql_query = re.sub(r"^```(?:sql)?", "", sql_query)
        sql_query = re.sub(r"```$", "", sql_query)
        sql_query = sql_query.strip()

        logger.info(f"Generated SQL query: {sql_query}")
        try:
            result_df = query_duckdb(database, sql_query)
            result_str = result_df.to_csv(sep="|", index=False)
            result = Result(sql_query=sql_query, result=result_str)
        except SQLParsingError as e:
            logger.error(f"SQL parsing error during evaluation: {e}")
            result = Result(sql_query=sql_query, result="SQL PARSING ERROR")
        except SQLExecutionError as e:
            logger.error(f"SQL execution error during evaluation: {e}")
            result = Result(sql_query=sql_query, result="SQL EXECUTION ERROR")
        return result

    # Wrap query_generator to inject rate_limiter
    async def wrapped_query_generator(question: str) -> Result:
        return await query_generator(
            question, rate_limiter=evaluation_ctx["rate_limiter"]
        )

    # Show evaluation progress
    console = Console()
    with console.status(
        f"[bold green]Evaluating {len(dataset.cases)} cases..."
    ) as status:
        report = dataset.evaluate_sync(
            wrapped_query_generator,
            max_concurrency=max_concurrency if max_concurrency > 0 else None,
        )
        status.update("[bold green]Evaluation completed!")

    logger.info("Evaluation completed. Generating report...")
    table = report.console_table(
        include_input=True,
        include_output=True,
        include_expected_output=True,
        include_durations=True,
        include_metadata=True,
    )

    console = Console()
    console.print(table)

    # After creating and printing the table
    output_path = export_report_to_markdown(
        report, table, model_name=model, output_dir=output_dir
    )
    logger.info(f"Report saved to {output_path.resolve()}")


@cli.command()
@click.argument("action", type=click.Choice(["show", "get", "set"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.pass_context
def config(
    ctx: click.Context, action: str, key: Optional[str], value: Optional[str]
) -> None:
    """
    Manage configuration settings.

    ACTIONS:
      show     Display all configuration settings
      get      Get a specific configuration value (e.g., 'llm.model')
      set      Set a specific configuration value (e.g., 'llm.model gpt-4')

    Examples:
      eval_sql config show
      eval_sql config get llm.model
      eval_sql config set llm.temperature 0.7
    """
    config = get_config()
    console = Console()

    if action == "show":
        # Display all configuration in a nice table
        table = Table(title="Configuration Settings")
        table.add_column("Section", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Value", style="yellow")

        # This assumes get_config() returns a dict-like object with nested sections
        # Adjust as needed based on your actual config structure
        for section_name, section in config.__dict__.items():
            if hasattr(section, "__dict__"):
                for key, value in section.__dict__.items():
                    if not key.startswith("_"):  # Skip private attributes
                        table.add_row(section_name, key, str(value))

        console.print(table)

    elif action == "get":
        # Get a specific configuration value
        if not key:
            console.print("[bold red]Error:[/] Key is required for 'get' action")
            return

        try:
            section, setting = key.split(".", 1)
            value = getattr(getattr(config, section), setting)
            console.print(f"[cyan]{key}:[/] [yellow]{value}[/]")
        except (AttributeError, ValueError):
            console.print(f"[bold red]Error:[/] Invalid configuration key: {key}")

    elif action == "set":
        # Set a specific configuration value
        if not key or not value:
            console.print(
                "[bold red]Error:[/] Both key and value are required for 'set' action"
            )
            return

        try:
            section, setting = key.split(".", 1)
            # Note: This implementation depends on your config system supporting this style of setting
            # You'll need to modify this to work with your actual config system
            console.print(
                "[yellow]Note:[/] Setting configuration values requires implementation specific to your config system"
            )
            console.print(f"Would set [cyan]{key}[/] to [green]{value}[/]")

            # Example implementation (pseudocode):
            # config.set_value(section, setting, value)
            # config.save()

        except (AttributeError, ValueError):
            console.print(f"[bold red]Error:[/] Invalid configuration key: {key}")


@cli.command()
@click.argument("question")
@add_input_options
@add_model_options
@add_reranker_options
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    eval_cases: str,
    metadata_json: str,
    template_path: str,
    model: str,
    temperature: float,
    database: str,
    provider: str,
    base_url: Optional[str],
    use_reranker: bool,
    use_semantic_search: bool,
    num_examples: int,
) -> None:
    """
    Ask a single SQL question without full evaluation.

    Useful for testing and debugging prompts.

    Examples:
      eval_sql ask "How many users are there in the database?"
      eval_sql ask "What is the average order value?" --database orders.db
      eval_sql ask "Find top 10 products by sales" --use-semantic-search --num-examples 3
      eval_sql ask "Calculate revenue by region" --use-reranker --use-semantic-search
    """
    import asyncio

    from research.llm.sql_generator import create_system_prompt, load_metadata

    async def run_query():
        # Initialize console for rich output
        console = Console()

        # Handle examples retrieval (semantic search/reranking)
        examples_text = ""
        if use_semantic_search or use_reranker:
            try:
                console.print("[bold cyan]Retrieving relevant examples...[/]")

                # Fetch examples based on flags
                if use_semantic_search:
                    from research.llm.semantic_search import (
                        _load_resources,
                        semantic_search,
                    )

                    _load_resources()
                    indices = semantic_search(question, top_k=10)  # Always fetch 10
                    from research.llm.semantic_search import _examples

                    similar_cases = [_examples[idx] for idx in indices]
                    console.print(
                        f"[bold cyan]Found {len(similar_cases)} examples via semantic search[/]"
                    )
                else:
                    # Fall back to reranker-only mode
                    from research.llm.reranker import find_similar_examples

                    similar_cases = find_similar_examples(
                        query=question,
                        eval_dataset_path=eval_cases,
                        top_n=10,  # Fetch 10 for reranker
                    )
                    console.print(
                        f"[bold cyan]Found {len(similar_cases)} examples from dataset[/]"
                    )

                # Rerank if enabled
                if use_reranker and similar_cases:
                    from research.llm.reranker import rerank_examples

                    similar_cases = rerank_examples(question, similar_cases)
                    console.print("[bold cyan]Reranked examples by relevance[/]")

                # Limit to num_examples
                similar_cases = similar_cases[:num_examples]

                # Prepare examples for the prompt
                if similar_cases:
                    from research.llm.reranker import prepare_examples_for_prompt

                    examples_text = prepare_examples_for_prompt(similar_cases)
                    console.print(
                        f"[bold cyan]Added {len(similar_cases)} examples to the prompt[/]"
                    )
            except Exception as e:
                console.print(f"[bold red]Error retrieving examples: {e}[/]")
                console.print("[yellow]Continuing without examples...[/]")

        # Create system prompt
        system_prompt = create_system_prompt(
            load_metadata(metadata_json),
            template_path,
            examples=examples_text,
        )
        print(system_prompt)

        # Set up model
        if provider == "ollama":
            model_instance = OpenAIModel(
                model_name=model,
                provider=OpenAIProvider(base_url=base_url),
            )
        else:
            model_instance = model

        # Create agent and execute query
        agent = Agent(
            model_instance,
            output_type=str,
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=get_config().llm.max_tokens,
            ),
        )

        # Show progress
        with console.status(
            "[bold green]Generating SQL query for question..."
        ) as status:
            output = await agent.run(question)
            response = output.output

            # Extract SQL query from response
            import re

            sql_query = response.strip()

            # Remove any markdown code blocks
            code_block_pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
            match = re.search(code_block_pattern, sql_query, re.IGNORECASE)
            if match:
                sql_query = match.group(1).strip()

            # Remove any remaining backticks
            sql_query = re.sub(r"^```(?:sql)?", "", sql_query)
            sql_query = re.sub(r"```$", "", sql_query)
            sql_query = sql_query.strip()

            status.update("[bold green]Executing SQL query...")

            try:
                # Execute the query against the database
                result_df = query_duckdb(database, sql_query)

                # Update status
                status.update("[bold green]Query execution completed")
            except (SQLParsingError, SQLExecutionError) as e:
                console.print(f"[bold red]Error executing query:[/] {e}")
                return

        # Display the results in a table
        console.print("\n[bold cyan]Generated SQL Query:[/]")
        console.print(f"```sql\n{sql_query}\n```")

        console.print("\n[bold cyan]Query Results:[/]")

        # Create a rich table from the dataframe
        if not result_df.empty:
            rich_table = Table(show_header=True, header_style="bold magenta")

            # Add columns
            for col in result_df.columns:
                rich_table.add_column(str(col))

            # Add rows
            for _, row in result_df.iterrows():
                rich_table.add_row(*[str(cell) for cell in row])

            # Print the table
            console.print(rich_table)
        else:
            console.print("[yellow]Query returned no results[/]")

    # Run the async function
    asyncio.run(run_query())


if __name__ == "__main__":
    cli()
