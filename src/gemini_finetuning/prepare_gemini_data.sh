#!/bin/bash

# Script to prepare data for Gemini fine-tuning

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Gemini fine-tuning data preparation...${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Current directory is the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo -e "\n${GREEN}1. Installing dependencies...${NC}"
cd "$SCRIPT_DIR"

# Try to use uv first, fall back to pip if not available
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    python -m pip install -r requirements.txt
fi

echo -e "\n${GREEN}2. Converting dataset to Gemini format...${NC}"
python "$SCRIPT_DIR/convert_dataset.py"

echo -e "\n${GREEN}3. Validating created files...${NC}"
python "$SCRIPT_DIR/test_conversion.py"

# Check if a bucket name was provided
if [ "$#" -gt 0 ]; then
    BUCKET_NAME=$1
    echo -e "\n${GREEN}4. Uploading files to GCS bucket $BUCKET_NAME...${NC}"
    python "$SCRIPT_DIR/upload_to_gcs.py" --bucket "$BUCKET_NAME" --source "$SCRIPT_DIR"
else
    echo -e "\n${YELLOW}To upload files to GCS, run:${NC}"
    echo -e "python $SCRIPT_DIR/upload_to_gcs.py --bucket YOUR_BUCKET_NAME"
fi

echo -e "\n${GREEN}Data preparation complete! 🎉${NC}"
echo -e "\nNext steps:"
echo -e "1. Upload files to GCS (if not done already)"
echo -e "2. Create a Gemini fine-tuning job using the Vertex AI console or SDK"
echo -e "3. Wait for the model to finish training"
echo -e "4. Use your custom fine-tuned model" 