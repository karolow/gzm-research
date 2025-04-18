# Gemini Fine-tuning

This module contains scripts and utilities for fine-tuning Google's Gemini models using the Vertex AI Supervised Tuning feature.

## Overview

The scripts in this directory help convert your existing dataset into the format required by Gemini for fine-tuning and split it into training, testing, and validation sets.

### Gemini Fine-tuning Format

For fine-tuning, Gemini requires data in a specific JSONL format where each line contains a complete example. Each example should have the following structure:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "User input text"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "Expected model output text"
        }
      ]
    }
  ]
}
```

## Quick Start

The easiest way to prepare your data is to use the provided shell script:

```bash
# Make the script executable
chmod +x src/gemini_finetuning/prepare_gemini_data.sh

# Run the script
./src/gemini_finetuning/prepare_gemini_data.sh [BUCKET_NAME]
```

The script will:
1. Install required dependencies
2. Convert the dataset to Gemini format
3. Validate the created files
4. Upload to GCS if a bucket name is provided

## Scripts

### `convert_dataset.py`

This script converts the original dataset in `src/eval_dataset.json` to the format required by Gemini and splits it into training, testing, and validation sets using a 70/15/15 ratio.

#### Usage

```bash
python src/gemini_finetuning/convert_dataset.py
```

#### Output

The script generates three JSONL files in the `src/gemini_finetuning` directory:
- `train.jsonl`: Contains 70% of the examples for model training.
- `test.jsonl`: Contains 15% of the examples for testing.
- `validation.jsonl`: Contains 15% of the examples for validation during training.

### `test_conversion.py`

This script validates that the generated JSONL files are correctly formatted for Gemini fine-tuning.

#### Usage

```bash
python src/gemini_finetuning/test_conversion.py
```

### `upload_to_gcs.py`

This script uploads the generated JSONL files to a Google Cloud Storage bucket for fine-tuning.

#### Usage

```bash
python src/gemini_finetuning/upload_to_gcs.py --bucket YOUR_BUCKET_NAME [--source SOURCE_DIR] [--destination DEST_DIR] [--project PROJECT_ID]
```

Parameters:
- `--bucket`: (Required) Name of the GCS bucket without the `gs://` prefix
- `--source`: (Optional) Local directory containing JSONL files (default: current directory)
- `--destination`: (Optional) Destination directory in the bucket
- `--project`: (Optional) Google Cloud Project ID (defaults to authenticated project)

### `prepare_gemini_data.sh`

A convenience shell script that runs all the necessary steps to prepare data for Gemini fine-tuning.

#### Usage

```bash
./src/gemini_finetuning/prepare_gemini_data.sh [BUCKET_NAME]
```

### `use_tuned_model.py`

This script demonstrates how to use your fine-tuned Gemini model after training is complete.

#### Usage

```bash
python src/gemini_finetuning/use_tuned_model.py --endpoint ENDPOINT_ID --query "Your query here" [--project PROJECT_ID] [--location LOCATION]
```

Parameters:
- `--endpoint`: (Required) Endpoint ID of the fine-tuned model
- `--query`: (Required) Query to process
- `--project`: (Optional) Google Cloud Project ID
- `--location`: (Optional) Google Cloud region (default: us-central1)

Example:
```bash
python src/gemini_finetuning/use_tuned_model.py --endpoint projects/123456789/locations/us-central1/endpoints/9876543210 --query "Jak często mieszkańcy centrum Katowic i Piotrowic chodzą do galerii sztuki?"
```

## Fine-tuning Process

To fine-tune a Gemini model on Google Cloud Platform:

1. Generate the training and validation files:
   ```bash
   python src/gemini_finetuning/convert_dataset.py
   ```

2. Upload the generated JSONL files to Google Cloud Storage:
   ```bash
   python src/gemini_finetuning/upload_to_gcs.py --bucket YOUR_BUCKET_NAME
   ```

3. Use the Vertex AI console or SDK to create a supervised tuning job:
   ```python
   import vertexai
   from google.cloud import aiplatform
   
   # Initialize Vertex AI
   vertexai.init(project='YOUR_PROJECT_ID', location='us-central1')
   
   # Create a tuning job
   tuning_job = aiplatform.TuningJob.create(
       base_model='gemini-2.0-flash',
       training_data_uri='gs://YOUR_BUCKET_NAME/train.jsonl',
       validation_data_uri='gs://YOUR_BUCKET_NAME/validation.jsonl',
       tuned_model_display_name='my-tuned-gemini-model'
   )
   ```

4. Monitor the tuning job progress in the Google Cloud Console.

5. After training is complete, use your fine-tuned model:
   ```bash
   python src/gemini_finetuning/use_tuned_model.py --endpoint ENDPOINT_ID --query "Your query here"
   ```

For complete documentation, see [Google's official Gemini tuning documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning).

## Limitations

- Gemini 2.0 Flash/Flash-Lite supports up to 131,072 tokens for input and output.
- Maximum training dataset size is 1M text-only examples.
- Maximum validation dataset size is 5,000 examples.
- Maximum file size is 1GB for JSONL files.

## Requirements

- Python 3.8+
- Google Cloud account with Vertex AI access
- Sufficient quota for Gemini tuning jobs

### Python Packages

Install the required packages:

```bash
cd src/gemini_finetuning
pip install -r requirements.txt
```

Or using uv:

```bash
cd src/gemini_finetuning
uv pip install -r requirements.txt
``` 