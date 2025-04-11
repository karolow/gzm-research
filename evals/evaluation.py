from dataclasses import dataclass
from typing import Any

import dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from db_operations import query_duckdb
from src.llm_sql import create_prompt, load_metadata

dotenv.load_dotenv(override=True)

metadata_json_path = "src/survey_metadata_queries.json"
template_path = "src/sql_query_prompt.jinja2"
eval_cases_json_path = "src/eval_dataset.json"


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
            # Try different separators
            if "|" in line:
                values = [v.strip() for v in line.split("|")]
            elif "," in line:
                values = [v.strip() for v in line.split(",")]
            else:
                values = [line.strip()]
            all_values.extend(values)

        return all_values

    async def evaluate(self, ctx: EvaluatorContext[str, Result]) -> float:
        # Handle empty strings and None values
        if not ctx.expected_output or not ctx.output:
            return 0.0

        # Exact match case
        if ctx.expected_output == ctx.output:
            return 1.0

        expected_values = self.extract_values(ctx.expected_output)
        output_values = self.extract_values(ctx.output)

        print(f"Expected values: {expected_values}")
        print(f"Output values: {output_values}")

        # If no values to check, return 0
        if not expected_values:
            return 0.0

        # Count how many expected values are in the output
        matches = 0
        for expected_val in expected_values:
            normalized_expected = expected_val.lower().strip()
            if any(
                normalized_expected == out_val.lower().strip()
                or normalized_expected in out_val.lower()
                or out_val.lower() in normalized_expected
                for out_val in output_values
            ):
                matches += 1

        # Calculate the match ratio
        match_ratio = matches / len(expected_values)

        # Scale to 0.9 max for partial matches
        if 0 < match_ratio < 1:
            return match_ratio * 1.0

        return match_ratio


case1 = Case(
    name="117",
    inputs="Czy mieszkańcy Tarnowskich Gór częściej chodzą na koncery niż mieszkańcy innych miejscowości w powiecie tarnogórskim?",
    expected_output=Result(
        result="miejsce|procent_koncerty_rocznie\nTarnowskie Góry|25.88\nInne gminy w powiecie TG|40.92",
        sql_query=None,
    ),
    metadata={
        "complexity": "medium",
        "ambiguity_note": "Uses `gmina_miejscowosc` to differentiate within `miasto_powiat` = 'Powiat tarnogórski'. Compares annual concert attendance (at least once).",
    },
    evaluators=(ResultWeighted(),),
)
case2 = Case[str, Result, Any](
    name="118",
    inputs="Jaki jest średni miesięczny wydatek na kulturę wśród studentów w Katowicach w porównaniu do studentów w Gliwicach?",
    expected_output=Result(
        result="miasto_powiat|sredni_wydatek_studenci\nKatowicki|101.84\nGliwicki|65.47",
        sql_query=None,
    ),
    metadata={
        "complexity": "medium",
        "ambiguity_note": "",
    },
    evaluators=(ResultWeighted(),),
)


# dataset = Dataset[str, Result, Any](cases=[case2])


# dataset.add_evaluator(ResultEquals())

# Load the metadata content properly
metadata_content = load_metadata(metadata_json_path)
system_prompt = create_prompt(metadata_content, template_path)

print(system_prompt)

model_settings = ModelSettings(
    temperature=0.2,
    top_p=0.95,
    max_tokens=1024,
)


agent = Agent(
    # "groq:llama-3.3-70b-versatile",
    # "google-gla:gemini-2.5-pro-exp-03-25",
    "google-gla:gemini-2.0-flash",
    result_type=str,
    system_prompt=system_prompt,
    model_settings=model_settings,
)


async def mock_evaluation(question: str) -> str:
    return "miejsce|procent_koncerty_rocznie\nInne gminy w powiecie TG|40.92\nTarnowskie Góry|25.88\n"


async def generate_sql_query(question: str) -> Result:
    output = await agent.run(question)
    response = output.data

    # Extract SQL query from markdown code block if present
    if response.startswith("```sql") and response.endswith("```"):
        sql_query = response[6:-3].strip()  # Remove ```sql and ```
    else:
        sql_query = response.strip()

    print(sql_query)
    # Execute query against the database
    database_path = "research.db"
    result_df = query_duckdb(database_path, sql_query)

    # Convert result to string format expected by the evaluator
    result_str = result_df.to_csv(sep="|", index=False)
    result = Result(sql_query=sql_query, result=result_str)
    return result


def main():
    try:
        # --- Load directly using Dataset.from_file ---
        # Specify the generic types: Input=str, Output=Result, Metadata=Any
        # Register custom evaluators needed for deserialization
        dataset = Dataset[str, Result, Any].from_file(
            eval_cases_json_path,
            fmt="json",
            custom_evaluator_types=[ResultWeighted, ResultEquals],
        )
        print(f"Loaded {len(dataset.cases)} cases for evaluation.")

    except FileNotFoundError:
        print(f"Error: Evaluation cases file not found at {eval_cases_json_path}")
        return
    except ValueError as e:  # Catches JSON errors and validation errors
        print(f"Error loading dataset from {eval_cases_json_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading cases: {e}")
        return

    # Evaluate the model
    report = dataset.evaluate_sync(generate_sql_query)

    # Print evaluation report
    report.print(
        include_input=True,
        include_output=True,
        include_expected_output=True,
        include_durations=True,
        include_metadata=True,
    )


if __name__ == "__main__":
    main()
