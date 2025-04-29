from typing import Any

import pytest
from pydantic_evals.evaluators import EvaluatorContext

# Import the actual implementation from evaluation.py
from evals.evaluation import Result, ResultEquals


def create_test_context(
    name: str,
    inputs: str,
    expected_output: Result | None,
    output: Result,
    metadata: dict[str, Any] | None = None,
) -> EvaluatorContext[str, Result, Any]:
    """Create an EvaluatorContext for testing."""
    kwargs: dict[str, Any] = {
        "name": name,
        "inputs": inputs,
        "expected_output": expected_output,
        "output": output,
        "metadata": metadata or {},
        "duration": None,
        "_span_tree": None,
        "attributes": None,
        "metrics": None,
    }
    return EvaluatorContext(**kwargs)


@pytest.fixture
def evaluator() -> ResultEquals:
    """Fixture providing a ResultEquals evaluator instance."""
    return ResultEquals()


def test_extract_numeric_values(evaluator: ResultEquals) -> None:
    """Test extraction of numeric values from various formats."""
    # Test with string input
    text = "header1|header2\nvalue1|42.5\nvalue2|10"
    numbers = evaluator.extract_numeric_values(text)
    assert numbers == ["42.5", "10"]

    # Test with Result object
    result = Result(
        sql_query="SELECT * FROM table",
        result="column1|column2\ntext|123.45\nanother|67.89",
    )
    numbers = evaluator.extract_numeric_values(result)
    assert numbers == ["123.45", "67.89"]

    # Test with no numeric values
    text = "column1|column2\ntext|text\nmore|stuff"
    numbers = evaluator.extract_numeric_values(text)
    assert numbers == []

    # Test with mixed formats - this is the key test case for our implementation
    text = "value1,12.34|value2,56.78"
    numbers = evaluator.extract_numeric_values(text)
    assert numbers == ["12.34", "56.78"]


@pytest.mark.asyncio
async def test_perfect_match(evaluator: ResultEquals) -> None:
    """Test scenario where all expected numbers match exactly."""
    ctx = create_test_context(
        name="test1",
        inputs="query1",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname1|42.5\nname2|10.3",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 1.0


@pytest.mark.asyncio
async def test_close_match(evaluator: ResultEquals) -> None:
    """Test scenario where numbers are very close but not exact."""
    ctx = create_test_context(
        name="test2",
        inputs="query2",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname1|42.505\nname2|10.298",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 1.0  # Should tolerate small differences


@pytest.mark.asyncio
async def test_partial_match(evaluator: ResultEquals) -> None:
    """Test scenario where only some expected numbers match."""
    ctx = create_test_context(
        name="test3",
        inputs="query3",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname1|42.5\nname2|99.9",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 0.5  # Only 1 of 2 numbers match


@pytest.mark.asyncio
async def test_extra_numbers_penalty(evaluator: ResultEquals) -> None:
    """Test scenario with correct numbers but extra ones too."""
    ctx = create_test_context(
        name="test4",
        inputs="query4",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2|col3\nname1|42.5|30.0\nname2|10.3|20.0",
        ),
    )

    score = await evaluator.evaluate(ctx)
    # 2 matches but 2 extra numbers, so penalty applied
    expected_score = 2 / 2 * (1 - min(0.5, 2 / 2 * 0.5))
    assert score == expected_score


@pytest.mark.asyncio
async def test_missing_numbers(evaluator: ResultEquals) -> None:
    """Test scenario with missing expected numbers."""
    ctx = create_test_context(
        name="test5",
        inputs="query5",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3\nname3|99.9"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname1|42.5\nname2|10.3",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 2 / 3  # 2 of 3 expected numbers found


@pytest.mark.asyncio
async def test_different_order(evaluator: ResultEquals) -> None:
    """Test scenario with numbers in different order."""
    ctx = create_test_context(
        name="test6",
        inputs="query6",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname2|10.3\nname1|42.5",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 1.0  # Order shouldn't matter


@pytest.mark.asyncio
async def test_empty_results(evaluator: ResultEquals) -> None:
    """Test scenario with empty results."""
    ctx = create_test_context(
        name="test7",
        inputs="query7",
        expected_output=Result(
            sql_query=None, result="col1|col2\nname1|42.5\nname2|10.3"
        ),
        output=Result(sql_query="SELECT * FROM table", result=""),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 0.0  # No matches


@pytest.mark.asyncio
async def test_none_results(evaluator: ResultEquals) -> None:
    """Test scenario with None results."""
    ctx = create_test_context(
        name="test8",
        inputs="query8",
        expected_output=None,
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nname1|42.5\nname2|10.3",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 0.0  # No expected output


@pytest.mark.asyncio
async def test_real_world_example(evaluator: ResultEquals) -> None:
    """Test with a real-world example from our actual cases."""
    ctx = create_test_context(
        name="real_example",
        inputs="Jaki jest średni miesięczny wydatek na kulturę wśród studentów w Katowicach w porównaniu do studentów w Gliwicach?",
        expected_output=Result(
            sql_query=None,
            result="miasto_powiat|sredni_wydatek_studenci\nKatowicki|101.84\nGliwicki|65.47",
        ),
        output=Result(
            sql_query="SELECT subregion_metropolii_gzm, ROUND(AVG(wydatki_na_kulture_kwota), 2) AS srednia FROM participation_survey WHERE zatrudnienie_student = TRUE AND (subregion_metropolii_gzm = 'KATOWICE' OR subregion_metropolii_gzm = 'GLIWICKI') GROUP BY subregion_metropolii_gzm",
            result="subregion_metropolii_gzm|srednia\nKATOWICE|101.84\nGLIWICKI|65.47",
        ),
    )

    score = await evaluator.evaluate(ctx)
    assert score == 1.0  # Should be a perfect match


@pytest.mark.asyncio
async def test_mixed_format_columns(evaluator: ResultEquals) -> None:
    """Test the specific case where columns contain mixed text and numeric values."""
    result_obj = Result(sql_query=None, result="col1|col2\nvalue1,12.34|value2,56.78")

    # Check that the numeric extraction works correctly
    numbers = evaluator.extract_numeric_values(result_obj)
    assert numbers == ["12.34", "56.78"]

    # Test the evaluation with identical inputs/outputs
    ctx = create_test_context(
        name="mixed_format",
        inputs="query_mixed",
        expected_output=result_obj,
        output=Result(
            sql_query="SELECT * FROM table",
            result="col1|col2\nvalue1,12.34|value2,56.78",
        ),
    )

    # Then check that score is 1.0 for identical results
    score = await evaluator.evaluate(ctx)
    assert score == 1.0
