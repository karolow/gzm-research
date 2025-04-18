#!/usr/bin/env python3
"""
Launch Gemini fine-tuning jobs using the Google Generative AI API.
"""

import json
import logging
import os
import sys
from typing import Any, Optional, Protocol, Tuple, cast, runtime_checkable

import click
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import Model
from google.cloud.storage import Blob


# Define protocol classes for the methods we need to work with
@runtime_checkable
class BucketProtocol(Protocol):
    def blob(self, blob_name: str, **kwargs) -> Blob: ...


@runtime_checkable
class BlobProtocol(Protocol):
    def upload_from_filename(self, filename: str, **kwargs) -> None: ...


@runtime_checkable
class GenAIConfigureProtocol(Protocol):
    def __call__(self, *, api_key: str = None) -> None: ...


@runtime_checkable
class ModelUploadProtocol(Protocol):
    @classmethod
    def upload_tuned_model(cls, **kwargs) -> Model: ...


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """
    Load configuration from .env file.

    Returns:
        dict[str, Any]: Configuration dictionary

    Raises:
        SystemExit: If required environment variables are missing
    """
    # Load environment variables from .env file
    load_dotenv()

    # Required configuration values
    required_vars = [
        "GCP_PROJECT_ID",
        "GCP_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCS_BUCKET_NAME",
        "GEMINI_BASE_MODEL",
        "GEMINI_TUNED_MODEL_NAME",
        "TRAIN_DATA_PATH",
    ]

    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.error("Please set these in your .env file")
        sys.exit(1)

    # Create config dictionary
    config = {
        "project_id": os.getenv("GCP_PROJECT_ID", ""),
        "location": os.getenv("GCP_LOCATION", ""),
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        "bucket_name": os.getenv("GCS_BUCKET_NAME", ""),
        "base_model": os.getenv("GEMINI_BASE_MODEL", ""),
        "tuned_model_name": os.getenv("GEMINI_TUNED_MODEL_NAME", ""),
        "train_data_path": os.getenv("TRAIN_DATA_PATH", ""),
        "test_data_path": os.getenv("TEST_DATA_PATH", ""),
        "validation_data_path": os.getenv("VALIDATION_DATA_PATH", ""),
        "epochs": int(os.getenv("EPOCHS", "3")),
        "learning_rate_multiplier": float(os.getenv("LEARNING_RATE_MULTIPLIER", "1.0")),
        "batch_size": int(os.getenv("BATCH_SIZE", "8")),
    }

    return config


def upload_to_gcs(
    bucket_name: str, source_file_path: str, destination_blob_name: str
) -> str:
    """
    Upload a file to Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name
        source_file_path: Path to the source file
        destination_blob_name: Name of the destination blob in GCS

    Returns:
        str: GCS URI of the uploaded file (gs://bucket/blob)
    """
    # Initialize client
    storage_client = storage.Client()

    # Get bucket and create blob reference
    # Using Protocol classes to handle complex types
    bucket = cast(BucketProtocol, storage_client.bucket(bucket_name))
    blob = cast(BlobProtocol, bucket.blob(destination_blob_name))

    # Upload file with explicit parameter name
    blob.upload_from_filename(filename=source_file_path)
    logger.info(
        f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}"
    )

    # Return GCS URI
    return f"gs://{bucket_name}/{destination_blob_name}"


def upload_dataset_files(
    config: dict[str, Any],
    train_file: str,
    test_file: Optional[str] = None,
    validation_file: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Upload dataset files to Google Cloud Storage.

    Args:
        config: Configuration dictionary
        train_file: Path to training data file
        test_file: Path to test data file (optional)
        validation_file: Path to validation data file (optional)

    Returns:
        Tuple[str, Optional[str], Optional[str]]: GCS URIs for train, test and validation files
    """
    bucket_name = cast(str, config["bucket_name"])

    # Upload training data
    train_basename = os.path.basename(train_file)
    train_destination = f"finetune_data/{train_basename}"
    train_uri = upload_to_gcs(bucket_name, train_file, train_destination)

    # Upload test data if provided
    test_uri = None
    if test_file:
        test_basename = os.path.basename(test_file)
        test_destination = f"finetune_data/{test_basename}"
        test_uri = upload_to_gcs(bucket_name, test_file, test_destination)

    # Upload validation data if provided
    validation_uri = None
    if validation_file:
        validation_basename = os.path.basename(validation_file)
        validation_destination = f"finetune_data/{validation_basename}"
        validation_uri = upload_to_gcs(
            bucket_name, validation_file, validation_destination
        )

    return train_uri, test_uri, validation_uri


def validate_jsonl_file(file_path: str) -> bool:
    """
    Validate a JSONL file for Gemini fine-tuning.

    Args:
        file_path: Path to the JSONL file

    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Check each line is valid JSON
            line_count = 0
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    line_count += 1

                    # Validate required fields
                    if "messages" not in data:
                        logger.error(f"Line {line_num}: Missing 'messages' field")
                        return False

                    messages = data["messages"]
                    if not isinstance(messages, list) or not messages:
                        logger.error(
                            f"Line {line_num}: 'messages' must be a non-empty list"
                        )
                        return False

                    # Validate message format
                    for msg_idx, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            logger.error(
                                f"Line {line_num}, message {msg_idx}: Message must be an object"
                            )
                            return False

                        if "role" not in msg or "content" not in msg:
                            logger.error(
                                f"Line {line_num}, message {msg_idx}: Message missing 'role' or 'content'"
                            )
                            return False

                        if msg["role"] not in ["user", "model"]:
                            logger.error(
                                f"Line {line_num}, message {msg_idx}: Invalid role '{msg['role']}'"
                            )
                            return False

                except json.JSONDecodeError:
                    logger.error(f"Line {line_num}: Invalid JSON")
                    return False

            if line_count == 0:
                logger.error("File is empty or contains no valid JSON lines")
                return False

            logger.info(f"Successfully validated {line_count} examples in {file_path}")
            return True

    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False


def launch_tuning_job(
    config: dict[str, Any],
    train_uri: str,
    test_uri: Optional[str] = None,
    validation_uri: Optional[str] = None,
) -> str:
    """
    Launch a Gemini fine-tuning job.

    Args:
        config: Configuration dictionary
        train_uri: GCS URI for training data
        test_uri: GCS URI for test data
        validation_uri: GCS URI for validation data

    Returns:
        str: The resource name of the launched job
    """
    # Initialize Vertex AI
    aiplatform.init(
        project=config["project_id"],
        location=config["location"],
    )

    # Set API key for generative AI if available
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # Using protocol to handle genai.configure
        configure_func = cast(GenAIConfigureProtocol, genai.configure)
        configure_func(api_key=api_key)

    # Prepare tuning job parameters
    tuning_options = {
        "epochs": config["epochs"],
        "learning_rate_multiplier": config["learning_rate_multiplier"],
        "batch_size": config["batch_size"],
    }

    logger.info(f"Starting fine-tuning job for model: {config['tuned_model_name']}")
    logger.info(f"Training options: {tuning_options}")

    # Create and launch tuning job using the Vertex AI API
    tuning_params = {
        "steps": tuning_options["epochs"],
        "learning_rate": tuning_options["learning_rate_multiplier"],
        "batch_size": tuning_options["batch_size"],
    }

    # Use Vertex AI to launch tuning job
    # Using Protocol to handle Model.upload_tuned_model
    model_cls = cast(ModelUploadProtocol, Model)

    model = model_cls.upload_tuned_model(
        base_model=config["base_model"],
        tuned_model_display_name=config["tuned_model_name"],
        training_dataset=train_uri,
        validation_dataset=validation_uri,
        evaluation_dataset=test_uri,
        tuning_parameters=tuning_params,
    )

    return str(model.resource_name)


@click.group()
def cli():
    """Command-line interface for Gemini fine-tuning."""
    pass


@cli.command("launch")
@click.option(
    "--train-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the training data JSONL file",
)
@click.option(
    "--test-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the test data JSONL file",
)
@click.option(
    "--validation-file",
    default=None,
    help="Path to the validation data JSONL file",
)
@click.option(
    "--base-model",
    default=None,
    help="Base model to fine-tune (overrides .env config)",
)
@click.option(
    "--model-name",
    default=None,
    help="Name for the fine-tuned model (overrides .env config)",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Number of epochs for training (overrides .env config)",
)
@click.option(
    "--learning-rate",
    type=float,
    default=None,
    help="Learning rate multiplier (overrides .env config)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for training (overrides .env config)",
)
@click.option(
    "--project-id",
    default=None,
    help="GCP project ID (overrides .env config)",
)
@click.option(
    "--location",
    default=None,
    help="GCP location (overrides .env config)",
)
@click.option(
    "--bucket",
    default=None,
    help="GCS bucket name (overrides .env config)",
)
def launch_command(
    train_file,
    test_file,
    validation_file,
    base_model,
    model_name,
    epochs,
    learning_rate,
    batch_size,
    project_id,
    location,
    bucket,
):
    """Launch a Gemini fine-tuning job."""
    # Load config from .env
    config = load_config()
    if config is None:
        sys.exit(1)

    # Override config with command-line options
    if base_model:
        config["base_model"] = base_model
    if model_name:
        config["tuned_model_name"] = model_name
    if epochs:
        config["epochs"] = epochs
    if learning_rate:
        config["learning_rate_multiplier"] = learning_rate
    if batch_size:
        config["batch_size"] = batch_size
    if project_id:
        config["project_id"] = project_id
    if location:
        config["location"] = location
    if bucket:
        config["bucket_name"] = bucket

    # Get file paths from config or override
    train_file = train_file or config["train_data_path"]
    test_file = test_file or config.get("test_data_path")
    validation_file = validation_file or config.get("validation_data_path")

    # Validate dataset files
    logger.info("Validating dataset files")
    if not validate_jsonl_file(train_file):
        logger.error("Invalid training data file")
        sys.exit(1)

    if test_file and not validate_jsonl_file(test_file):
        logger.error("Invalid test data file")
        sys.exit(1)

    if validation_file and not validate_jsonl_file(validation_file):
        logger.error("Invalid validation data file")
        sys.exit(1)

    # Upload dataset files to GCS
    logger.info("Uploading dataset files to GCS")
    train_uri, test_uri, validation_uri = upload_dataset_files(
        config, train_file, test_file, validation_file
    )

    # Launch tuning job
    try:
        job_name = launch_tuning_job(config, train_uri, test_uri, validation_uri)
        logger.info(f"Fine-tuning job launched: {job_name}")
    except Exception as e:
        logger.error(f"Failed to launch fine-tuning job: {e}")
        sys.exit(1)


@cli.command("validate")
@click.argument("jsonl_file", type=click.Path(exists=True))
def validate_command(jsonl_file):
    """Validate a JSONL file for Gemini fine-tuning."""
    valid = validate_jsonl_file(jsonl_file)
    if valid:
        logger.info(f"File is valid: {jsonl_file}")
    else:
        logger.error(f"File is invalid: {jsonl_file}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
