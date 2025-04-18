import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import click
import dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from research.config import get_config
from research.db.operations import query_duckdb
from research.llm.sql_generator import create_prompt, load_metadata
from research.utils.logging import setup_logger

dotenv.load_dotenv(override=True)

logger = setup_logger("evaluation")


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


@click.command()
@click.option(
    "--eval-cases",
    "-e",
    help="Path to evaluation cases JSON file",
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
    default=lambda: get_config().eval.dataset_path,
)
@click.option(
    "--metadata-json",
    "-m",
    help="Path to metadata JSON file",
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
    default=lambda: get_config().survey_metadata,
)
@click.option(
    "--template-path",
    "-t",
    help="Path to prompt template file",
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
    default=lambda: get_config().sql_prompt_template,
)
@click.option(
    "--model", "-M", help="LLM model to use", default=lambda: get_config().llm.model
)
@click.option(
    "--temperature",
    "-T",
    help="Temperature setting for the model",
    type=float,
    default=lambda: get_config().llm.temperature,
)
@click.option(
    "--database",
    "-d",
    help="Path to the database file",
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=False),
    default=lambda: get_config().db.default_path,
)
@click.option(
    "--log-level",
    "-l",
    help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    type=click.Choice(
        [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        case_sensitive=False,
    ),
    default="INFO",
)
@click.option(
    "--rate-limit",
    "-r",
    help="Rate limit for LLM API calls (requests per minute, 0 = no limit)",
    type=int,
    default=lambda: get_config().eval.rate_limit,
)
@click.option(
    "--range",
    "-R",
    help="Range of cases to evaluate (format: 'start-end', 1-indexed, inclusive-exclusive)",
    type=str,
)
@click.option(
    "--max-concurrency",
    "-c",
    help="Maximum number of concurrent evaluations (0 = unlimited)",
    type=int,
    default=lambda: get_config().eval.max_concurrency,
)
@click.option(
    "--provider",
    help="LLM provider: 'api' (default) or 'ollama' (local)",
    default=lambda: get_config().llm.provider,
)
@click.option(
    "--base-url",
    help="Base URL for LLM provider (only needed for ollama or custom endpoints)",
    default=lambda: get_config().llm.base_url,
)
def main(
    eval_cases: str,
    metadata_json: str,
    template_path: str,
    model: str,
    temperature: float,
    database: str,
    log_level: str,
    rate_limit: int,
    range: Optional[str],
    max_concurrency: int,
    provider: str,
    base_url: Optional[str],
) -> None:
    """
    Run model evaluation against SQL queries.

    This tool evaluates the model's ability to generate SQL queries
    from natural language questions and produce correct results.
    """
    global logger
    logger = setup_logger("evaluation", log_level)

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(f"Using database at: {database}")
    logger.info(f"Loading evaluation cases from: {eval_cases}")

    try:
        full_dataset = Dataset[str, Result, Any].from_file(
            eval_cases,
            fmt="json",
            custom_evaluator_types=[ResultWeighted, ResultEquals],
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
    rate_limiter = None
    if rate_limit > 0:
        rate_limiter = RateLimiter(rate_limit_per_minute=rate_limit)

    async def query_generator(
        question: str, rate_limiter: RateLimiter | None = None
    ) -> Result:
        # Use the rate limiter if provided
        if rate_limiter is not None:
            await rate_limiter.acquire()

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
            system_prompt=create_prompt(load_metadata(metadata_json), template_path),
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
        result_df = query_duckdb(database, sql_query)
        result_str = result_df.to_csv(sep="|", index=False)
        result = Result(sql_query=sql_query, result=result_str)
        return result

    # Wrap query_generator to inject rate_limiter
    async def wrapped_query_generator(question: str) -> Result:
        return await query_generator(question, rate_limiter=rate_limiter)

    report = dataset.evaluate_sync(
        wrapped_query_generator,
        max_concurrency=max_concurrency if max_concurrency > 0 else None,
    )

    logger.info("Evaluation completed. Report:")
    report.print(
        include_input=True,
        include_output=True,
        include_expected_output=True,
        include_durations=True,
        include_metadata=True,
    )


if __name__ == "__main__":
    main()
