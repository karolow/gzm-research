import io
import json
import re
import sys
from typing import Any

import click
import pandas as pd
import pyreadstat
from pandas import DataFrame

from research.db.operations import save_to_duckdb
from research.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)


# --- Helper Functions ---


def load_sav(sav_file_path: str) -> tuple[DataFrame | None, Any | None]:
    """Loads data from an SPSS SAV file."""
    logger.info(f"Attempting to load SAV file: {sav_file_path}")
    try:
        df, sav_meta = pyreadstat.read_sav(
            sav_file_path, apply_value_formats=False, user_missing=True
        )
        logger.info(
            f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df, sav_meta
    except FileNotFoundError:
        logger.error(f"SAV file not found at {sav_file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load SAV file: {e}", exc_info=True)
        return None, None


def load_metadata(
    metadata_path: str,
) -> tuple[list[dict[str, Any]] | None, dict[str, dict[str, Any]] | None]:
    """Loads JSON metadata file."""
    logger.info(f"Attempting to load JSON metadata: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        metadata_dict = {item["original_symbol"]: item for item in metadata_list}
        logger.info("Metadata loaded successfully.")
        return metadata_list, metadata_dict
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {metadata_path}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON metadata: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logger.error(f"Error processing metadata file: {e}", exc_info=True)
        return None, None


def rename_columns(
    df: pd.DataFrame, metadata_list: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Renames DataFrame columns based on metadata."""
    logger.info("Renaming columns...")
    rename_map = {}
    final_name_map = {}  # Map original_symbol to its final name
    original_columns = df.columns.tolist()
    processed_symbols = set()
    rename_count = 0

    for item in metadata_list:
        original_symbol = item["original_symbol"]
        if (
            original_symbol in processed_symbols
            or original_symbol not in original_columns
        ):
            if (
                original_symbol not in original_columns
                and original_symbol not in processed_symbols
            ):
                processed_symbols.add(original_symbol)
            continue

        processed_symbols.add(original_symbol)
        semantic_name = item["semantic_name"]
        final_name = semantic_name

        # Refined Renaming for T2B/T3B columns
        if original_symbol.startswith("T2B_"):
            base_name = re.sub(r"_top2$", "", semantic_name)
            final_name = f"{base_name}_ocena_top_bottom_2"
        elif original_symbol.startswith("T3B_"):
            base_name = re.sub(r"_top3$", "", semantic_name)
            final_name = f"{base_name}_ocena_top_mid_bottom_3"

        if original_symbol != final_name:
            if final_name in df.columns and final_name != original_symbol:
                logger.warning(
                    f"Target column name '{final_name}' already exists. Skipping rename for '{original_symbol}'."
                )
                final_name_map[original_symbol] = original_symbol  # Keep original name
            else:
                rename_map[original_symbol] = final_name
                final_name_map[original_symbol] = final_name
                rename_count += 1
        else:
            final_name_map[original_symbol] = original_symbol  # No rename needed

    df.rename(columns=rename_map, inplace=True)
    logger.info(f"Renamed {rename_count} columns.")
    return df, final_name_map


def apply_value_labels(
    df: pd.DataFrame,
    metadata_dict: dict[str, dict[str, Any]],
    final_name_map: dict[str, str],
) -> pd.DataFrame:
    """Applies value labels from metadata to DataFrame columns."""
    logger.info("Applying value labels...")
    mapped_count = 0
    processed_symbols = set()

    for original_symbol, item in metadata_dict.items():
        if (
            original_symbol in processed_symbols
            or original_symbol not in final_name_map
        ):
            continue
        processed_symbols.add(original_symbol)

        final_col_name = final_name_map[original_symbol]
        value_mapping = item.get("value_mapping")

        if value_mapping and isinstance(value_mapping, dict):
            if final_col_name not in df.columns:
                logger.warning(
                    f"Column '{final_col_name}' (from '{original_symbol}') not found. Skipping mapping."
                )
                continue

            current_dtype = df[final_col_name].dtype
            converted_map = {}
            try:
                # Convert map keys based on DataFrame column dtype
                if pd.api.types.is_float_dtype(
                    current_dtype
                ) or pd.api.types.is_integer_dtype(current_dtype):
                    for k, v in value_mapping.items():
                        try:
                            converted_map[float(k)] = v
                        except (ValueError, TypeError):
                            converted_map[k] = v
                else:
                    converted_map = {str(k): v for k, v in value_mapping.items()}

                # Apply map and handle unmapped/missing
                df[final_col_name] = df[final_col_name].map(converted_map).fillna(pd.NA)
                mapped_count += 1

            except Exception as e:
                logger.error(
                    f"Error mapping column '{final_col_name}' (from '{original_symbol}'): {e}",
                    exc_info=True,
                )

    logger.info(f"Applied labels to {mapped_count} columns.")
    return df


def convert_data_types(
    df: pd.DataFrame,
    metadata_dict: dict[str, dict[str, Any]],
    final_name_map: dict[str, str],
) -> pd.DataFrame:
    """Converts columns to specified data types."""
    logger.info("Converting data types...")
    converted_count = 0
    processed_symbols = set()
    tak_nie_conversion_count = 0

    for original_symbol, item in metadata_dict.items():
        if (
            original_symbol in processed_symbols
            or original_symbol not in final_name_map
        ):
            continue
        processed_symbols.add(original_symbol)

        final_col_name = final_name_map[original_symbol]
        data_type = item["data_type"]

        if final_col_name not in df.columns:
            continue

        try:
            current_type = str(df[final_col_name].dtype)
            target_type_str = ""  # For logging

            # Handle Tak/Nie conversion to boolean
            if (data_type == "text" or data_type == "categorical") and (
                "possible_values" in item or "value_mapping" in item
            ):
                # Check for possible_values field (common for "text" type)
                if "possible_values" in item:
                    possible_values = item.get("possible_values", [])
                    if possible_values and isinstance(possible_values, list):
                        # Debug logging to see what's happening
                        if "Tak" in possible_values and "Nie" in possible_values:
                            logger.info(
                                f"Found Tak/Nie column: {final_col_name}, possible_values: {possible_values}"
                            )
                            if "Trudno powiedzieć" not in possible_values:
                                logger.info(f"Converting to boolean: {final_col_name}")
                                # First replace None values with pd.NA
                                df[final_col_name] = df[final_col_name].fillna(pd.NA)
                                # Create a mask for "Tak" and "Nie" values
                                tak_mask = df[final_col_name] == "Tak"
                                nie_mask = df[final_col_name] == "Nie"
                                # Apply the boolean transformation
                                df.loc[tak_mask, final_col_name] = True
                                df.loc[nie_mask, final_col_name] = False
                                # Convert to boolean type
                                df[final_col_name] = df[final_col_name].astype(
                                    "boolean"
                                )
                                target_type_str = "boolean (Tak/Nie conversion)"
                                tak_nie_conversion_count += 1
                                converted_count += 1
                                logger.info(
                                    f"Successfully converted '{final_col_name}' from Tak/Nie to boolean"
                                )

                # Check for value_mapping field (common for "categorical" type)
                elif "value_mapping" in item:
                    value_mapping = item.get("value_mapping", {})
                    if value_mapping and isinstance(value_mapping, dict):
                        # Check if this is a Tak/Nie mapping
                        values_set = set(value_mapping.values())
                        if (
                            "Tak" in values_set
                            and "Nie" in values_set
                            and len(values_set) == 2
                        ):
                            logger.info(
                                f"Found categorical Tak/Nie column: {final_col_name}, values: {values_set}"
                            )

                            # First replace None values with pd.NA
                            df[final_col_name] = df[final_col_name].fillna(pd.NA)

                            # Create a mask for "Tak" and "Nie" values
                            tak_mask = df[final_col_name] == "Tak"
                            nie_mask = df[final_col_name] == "Nie"

                            # Apply the boolean transformation
                            df.loc[tak_mask, final_col_name] = True
                            df.loc[nie_mask, final_col_name] = False

                            # Convert to boolean type
                            df[final_col_name] = df[final_col_name].astype("boolean")
                            target_type_str = "boolean (categorical Tak/Nie conversion)"
                            tak_nie_conversion_count += 1
                            converted_count += 1
                            logger.info(
                                f"Successfully converted '{final_col_name}' from categorical Tak/Nie to boolean"
                            )

            # Standard data type conversions
            elif data_type in [
                "numeric",
                "weight",
            ] and not pd.api.types.is_numeric_dtype(df[final_col_name].dtype):
                df[final_col_name] = pd.to_numeric(df[final_col_name], errors="coerce")
                target_type_str = "numeric"
                converted_count += 1
            elif data_type == "boolean" and current_type != "boolean":
                bool_map = {"Wymieniono": True, "Nie wymieniono": False}
                # Check if mapping is needed (i.e., if it contains the string labels)
                if df[final_col_name].dropna().isin(bool_map.keys()).any():
                    df[final_col_name] = df[final_col_name].map(
                        bool_map, na_action="ignore"
                    )
                # Convert to nullable boolean type regardless
                df[final_col_name] = df[final_col_name].astype("boolean")
                target_type_str = "boolean"
                converted_count += 1
            elif data_type == "identifier" and original_symbol == "UID":
                try:
                    df[final_col_name] = df[final_col_name].astype(pd.Int64Dtype())
                    target_type_str = "Int64"
                except (TypeError, ValueError):
                    df[final_col_name] = df[final_col_name].astype(str)
                    target_type_str = "string (fallback)"
                converted_count += 1

            # Optional: Log the conversion attempt
            if target_type_str:
                logger.debug(
                    f"Converted '{final_col_name}' from {current_type} to {target_type_str}"
                )

        except Exception as e:
            logger.error(
                f"Error converting type for column '{final_col_name}' (target: {data_type}): {e}",
                exc_info=True,
            )

    logger.info(
        f"Attempted type conversions for {converted_count} columns (including {tak_nie_conversion_count} Tak/Nie to boolean conversions)."
    )
    return df


def clean_text_columns(
    df: pd.DataFrame,
    metadata_dict: dict[str, dict[str, Any]],
    final_name_map: dict[str, str],
) -> pd.DataFrame:
    """Cleans text columns (strip whitespace, handle empty strings)."""
    logger.info("Cleaning text columns...")
    cleaned_count = 0
    processed_symbols = set()

    for original_symbol, item in metadata_dict.items():
        if (
            original_symbol in processed_symbols
            or original_symbol not in final_name_map
        ):
            continue
        processed_symbols.add(original_symbol)

        final_col_name = final_name_map[original_symbol]
        data_type = item["data_type"]

        if data_type in ["text", "text_detail"]:
            if final_col_name in df.columns and df[final_col_name].dtype == "object":
                try:
                    # Ensure conversion to string before applying string methods, handle potential NA
                    df[final_col_name] = df[final_col_name].astype(str).str.strip()
                    # Replace empty/whitespace/common missing strings with pd.NA
                    df[final_col_name] = df[final_col_name].replace(
                        {r"^\s*$": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA},
                        regex=True,
                    )
                    cleaned_count += 1
                except Exception as e:
                    logger.error(
                        f"Error cleaning text column '{final_col_name}': {e}",
                        exc_info=True,
                    )

    logger.info(f"Cleaned {cleaned_count} text columns.")
    return df


def drop_redundant_columns(
    df: pd.DataFrame, final_name_map: dict[str, str]
) -> pd.DataFrame:
    """Drops specified redundant columns."""
    logger.info("Dropping redundant columns...")
    columns_to_drop = []
    # Example: Drop the column derived from GminaPlec
    gmina_plec_orig_symbol = "GminaPlec"
    if gmina_plec_orig_symbol in final_name_map:
        gmina_plec_final_name = final_name_map[gmina_plec_orig_symbol]
        if gmina_plec_final_name in df.columns:
            columns_to_drop.append(gmina_plec_final_name)
            logger.info(f"Marked '{gmina_plec_final_name}' for dropping.")
        else:
            logger.warning(
                f"Column '{gmina_plec_final_name}' (orig: '{gmina_plec_orig_symbol}') not found for dropping."
            )
    else:
        logger.warning(
            f"Original symbol '{gmina_plec_orig_symbol}' not in metadata map, cannot drop."
        )

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
        logger.info(
            f"Dropped {len(columns_to_drop)} columns: {', '.join(columns_to_drop)}"
        )
    else:
        logger.info("No columns marked for dropping.")
    return df


def log_final_overview(df: pd.DataFrame) -> None:
    """Logs summary information about the final DataFrame."""
    logger.info("--- Final Data Overview ---")
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=False, show_counts=True)  # Concise info
    logger.info("DataFrame Info:\n" + buffer.getvalue())

    missing_counts = df.isna().sum()
    missing_filtered = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if not missing_filtered.empty:
        logger.info(
            "Missing Values per Column (Top 20 with missing > 0):\n"
            + missing_filtered.head(20).to_string()
        )
    else:
        logger.info("No missing values found in the final DataFrame.")


def save_output(df: pd.DataFrame, output_path: str | None) -> bool:
    """Saves the DataFrame to a CSV file."""
    if not output_path:
        logger.info("No output file specified. Skipping save.")
        return True  # Indicate success (no save attempted)

    logger.info(f"Attempting to save cleaned data to {output_path}...")
    try:
        # Use utf-8-sig for better Excel compatibility
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Data saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}", exc_info=True)
        return False


# --- Main Orchestrator Function ---
def preprocess_survey_data_orchestrator(
    sav_file_path: str,
    metadata_path: str,
    output_csv_path: str | None = None,
    db_path: str | None = None,
    table_name: str | None = None,
    replace_table: bool = False,
) -> DataFrame | None:
    """Orchestrates the preprocessing steps."""

    df, _ = load_sav(sav_file_path)
    if df is None:
        return None

    metadata_list, metadata_dict = load_metadata(metadata_path)
    if metadata_list is None or metadata_dict is None:
        return None

    df, final_name_map = rename_columns(df, metadata_list)
    df = apply_value_labels(df, metadata_dict, final_name_map)
    df = convert_data_types(df, metadata_dict, final_name_map)
    df = clean_text_columns(df, metadata_dict, final_name_map)
    df = drop_redundant_columns(df, final_name_map)

    log_final_overview(df)

    # Save to CSV if specified
    if not save_output(df, output_csv_path):
        logger.warning("Failed to save to CSV. Continuing with other outputs.")

    # Save to DuckDB if specified
    if db_path and table_name:
        logger.info(f"Saving data to DuckDB table '{table_name}'...")
        try:
            if save_to_duckdb(df, db_path, table_name, replace=replace_table):
                logger.info(f"Data successfully saved to DuckDB table '{table_name}'")
            else:
                logger.error("Failed to save to DuckDB")
        except Exception as e:
            logger.error(f"Error saving to DuckDB: {e}", exc_info=True)

    logger.info("--- Preprocessing finished ---")
    return df


# --- CLI Definition ---
@click.command()
@click.argument(
    "sav_file_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument(
    "metadata_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Optional path to save the cleaned data as a CSV file.",
)
@click.option(
    "-d",
    "--db-path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to DuckDB database file. If provided, data will be saved to DuckDB.",
)
@click.option(
    "-t",
    "--table-name",
    type=str,
    default="participation_survey",
    help="Name of the table to save data to in DuckDB. Default: 'participation_survey'",
)
@click.option(
    "--replace-table/--append-table",
    default=False,
    help="Whether to replace the table if it exists. Default: append",
)
def cli(
    sav_file_path: str,
    metadata_path: str,
    output: str | None,
    db_path: str | None,
    table_name: str,
    replace_table: bool,
) -> None:
    """
    Cleans and preprocesses survey data from an SPSS SAV file using JSON metadata.

    Can save output to CSV and/or DuckDB database.

    SAV_FILE_PATH: Path to the input SPSS (.sav) file.

    METADATA_PATH: Path to the input metadata (.json) file describing the columns and value labels.
    """
    cleaned_df = preprocess_survey_data_orchestrator(
        sav_file_path,
        metadata_path,
        output,
        db_path,
        table_name if db_path else None,
        replace_table,
    )

    if cleaned_df is None:
        click.echo(
            click.style("Preprocessing failed. Check logs for details.", fg="red"),
            err=True,
        )
        sys.exit(1)  # Exit with error code

    # Print head if no output file specified
    if output is None and db_path is None:
        click.echo("\nCleaned DataFrame head (no output file specified):")
        # Use pandas option to prevent truncation for better display in terminal
        with pd.option_context(
            "display.max_rows", 10, "display.max_columns", None, "display.width", 1000
        ):
            click.echo(cleaned_df.head())


if __name__ == "__main__":
    cli()
