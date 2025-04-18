import argparse
from pathlib import Path
from typing import Optional


def upload_to_gcs(
    bucket_name: str,
    source_directory: str,
    destination_directory: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    """
    Upload files from a local directory to Google Cloud Storage.

    Args:
        bucket_name: Name of the GCS bucket
        source_directory: Path to local directory containing files to upload
        destination_directory: Optional directory in the bucket to upload files to
        project_id: Optional Google Cloud Project ID
    """
    try:
        from google.cloud import storage
    except ImportError:
        print("Error: google-cloud-storage package is required.")
        print("Install it with: pip install google-cloud-storage")
        return

    # Authenticate and create a client
    client = storage.Client(project=project_id) if project_id else storage.Client()

    # Get the bucket
    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        return

    # Get all JSONL files from the source directory
    source_path = Path(source_directory)
    jsonl_files = list(source_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {source_directory}")
        return

    # Upload each file
    for file_path in jsonl_files:
        destination_blob_name = file_path.name
        if destination_directory:
            destination_blob_name = f"{destination_directory}/{destination_blob_name}"

        blob = bucket.blob(destination_blob_name)

        print(f"Uploading {file_path} to gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(str(file_path))

    print(
        f"Successfully uploaded {len(jsonl_files)} files to gs://{bucket_name}/{destination_directory or ''}"
    )
    print("\nYou can now use these files for Gemini fine-tuning.")
    print("\nExample using the Vertex AI SDK:")
    print(f"""
import vertexai
from google.cloud import aiplatform

# Initialize Vertex AI
vertexai.init(project='YOUR_PROJECT_ID', location='us-central1')

# Create a tuning job
tuning_job = aiplatform.TuningJob.create(
    base_model='gemini-2.0-flash',
    training_data_uri=f'gs://{bucket_name}/{destination_directory or ""}/train.jsonl',
    validation_data_uri=f'gs://{bucket_name}/{destination_directory or ""}/validation.jsonl',
    tuned_model_display_name='my-tuned-gemini-model'
)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload JSONL files to Google Cloud Storage for Gemini fine-tuning"
    )
    parser.add_argument(
        "--bucket", required=True, help="GCS bucket name (without gs:// prefix)"
    )
    parser.add_argument(
        "--source",
        default="./",
        help="Local directory containing JSONL files (default: current directory)",
    )
    parser.add_argument(
        "--destination", help="Destination directory in the bucket (optional)"
    )
    parser.add_argument(
        "--project",
        help="Google Cloud Project ID (optional, defaults to authenticated project)",
    )

    args = parser.parse_args()

    upload_to_gcs(
        bucket_name=args.bucket,
        source_directory=args.source,
        destination_directory=args.destination,
        project_id=args.project,
    )
