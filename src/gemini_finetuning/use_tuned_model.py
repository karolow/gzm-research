#!/usr/bin/env python3
"""
Example script demonstrating how to use a fine-tuned Gemini model.
"""

import argparse
from typing import Optional


def use_tuned_model(
    endpoint_id: str,
    query: str,
    project_id: Optional[str] = None,
    location: str = "us-central1",
) -> None:
    """
    Use a fine-tuned Gemini model to generate a response.

    Args:
        endpoint_id: The endpoint ID of the fine-tuned model
        query: The query to process
        project_id: Optional Google Cloud Project ID
        location: Google Cloud region where the model is deployed
    """
    try:
        import vertexai
        from vertexai.preview.generative_models import GenerativeModel
    except ImportError:
        print("Error: Google AI SDK is not installed.")
        print("Install it with: pip install google-cloud-aiplatform")
        return

    # Initialize Vertex AI with the given project and location
    vertexai.init(project=project_id, location=location)

    print(f"Query: {query}\n")
    print("Generating response...\n")

    # Load the model from the endpoint
    model = GenerativeModel(endpoint_name=endpoint_id)

    # Generate content
    response = model.generate_content(query)

    print("Response:")
    print("-" * 80)
    print(response.text)
    print("-" * 80)


def main() -> None:
    """Parse command line arguments and call the main function."""
    parser = argparse.ArgumentParser(description="Use a fine-tuned Gemini model")
    parser.add_argument(
        "--endpoint", required=True, help="Endpoint ID of the fine-tuned model"
    )
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--project", help="Google Cloud Project ID (optional)")
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud region (default: us-central1)",
    )

    args = parser.parse_args()

    use_tuned_model(
        endpoint_id=args.endpoint,
        query=args.query,
        project_id=args.project,
        location=args.location,
    )


if __name__ == "__main__":
    main()
