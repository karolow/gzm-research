# OpenAI API Integration for GZM

This document describes the integration of the `openai-api-blueprint` package with the GZM research project, creating a local OpenAI-compatible API server powered by Google Gemini.

## Overview

The integration provides:

- **Local OpenAI-compatible API server** using FastAPI and the openai-api-blueprint
- **Gemini backend** that translates OpenAI chat completion requests to Google Gemini API calls
- **Enhanced CLI support** in `llm-query` for both direct Gemini and OpenAI-compatible endpoints
- **Streaming support** for real-time response generation
- **Production-ready features** including authentication, rate limiting, and error handling

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   llm-query     │    │  OpenAI-Compatible│    │  Google Gemini  │
│      CLI        │───▶│   FastAPI Server  │───▶│      API        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         └─────────────▶│ Direct Gemini    │
                        │ Integration      │
                        └──────────────────┘
```

## Setup

### 1. Install Dependencies

The integration requires the `openai-api-blueprint` and `openai` packages, which have been added to `pyproject.toml`:

```bash
# Install/update dependencies
uv sync
```

### 2. Configuration

#### Environment Variables

Make sure your main `.env` file contains the required Gemini configuration:

```bash
# Required for Gemini backend
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
```

#### API Server Configuration (Optional)

For custom API server settings, create or modify `.env.api`:

```bash
# API server specific settings
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8001
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60

# Authentication tokens (auto-generated in development if not provided)
API_AUTH_TOKENS=your-custom-token-12345678901234567890
```

## Usage

### Starting the API Server

#### Method 1: Using the CLI script

```bash
# Start the API server
api-server
```

#### Method 2: Direct Python execution

```bash
python -m research.api.app
```

#### Method 3: Using uvicorn directly

```bash
uvicorn research.api.app:app --host 127.0.0.1 --port 8001 --reload
```

The server will start on `http://127.0.0.1:8001` by default.

**Available endpoints:**
- API Documentation: `http://127.0.0.1:8001/docs`
- Chat Completions: `POST http://127.0.0.1:8001/v1/chat/completions`
- Models List: `GET http://127.0.0.1:8001/v1/models`
- Health Check: `GET http://127.0.0.1:8001/health`
- Service Info: `GET http://127.0.0.1:8001/info`

### Using the Enhanced CLI

The `llm-query` command now supports both direct Gemini integration and OpenAI-compatible endpoints.

#### Direct Gemini Usage (Default)

```bash
# Generate and execute SQL using direct Gemini integration
llm-query ask --db research.db --question "Show me all survey responses from 2023"

# Generate SQL only (no execution)
llm-query generate --question "Count responses by gender"
```

#### OpenAI-Compatible API Usage

```bash
# Start the API server first
api-server

# In another terminal, use the API endpoint
llm-query ask --db research.db --question "Show me all survey responses from 2023" --use-api

# With custom API settings
llm-query ask \
  --db research.db \
  --question "Count responses by gender" \
  --use-api \
  --api-url "http://127.0.0.1:8001/v1" \
  --api-key "test-key" \
  --model "gzm-research"

# Generate only using API
llm-query generate --question "Show table schema" --use-api
```

#### CLI Options for API Mode

- `--use-api`: Enable OpenAI-compatible API mode
- `--api-url`: API base URL (default: `http://127.0.0.1:8001/v1`)
- `--api-key`: API authentication key (default: `test-key`)
- `--model`: Model name (default: `gzm-research`)

### Direct API Usage

You can also use the API directly with any OpenAI-compatible client:

#### Using curl

```bash
# Non-streaming completion
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "gzm-research",
    "messages": [
      {
        "role": "user",
        "content": "Generate SQL to count all records in the users table"
      }
    ]
  }'

# Streaming completion
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "gzm-research",
    "messages": [
      {
        "role": "user",
        "content": "Generate SQL to show top 10 users by score"
      }
    ],
    "stream": true
  }'
```

#### Using Python OpenAI Client

```python
from openai import OpenAI

# Initialize client for local server
client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="test-key"
)

# Generate completion
response = client.chat.completions.create(
    model="gzm-research",
    messages=[
        {"role": "user", "content": "Generate SQL to show all tables"}
    ]
)

print(response.choices[0].message.content)

# Streaming example
stream = client.chat.completions.create(
    model="gzm-research",
    messages=[
        {"role": "user", "content": "Generate SQL to count records"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Configuration Options

### API Server Settings

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ENVIRONMENT` | `development` | Server environment mode |
| `HOST` | `127.0.0.1` | Server host address |
| `PORT` | `8001` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RATE_LIMIT_PER_MINUTE` | `60` | Request rate limit |
| `API_AUTH_TOKENS` | Auto-generated | Comma-separated API keys |

### Gemini Backend Settings

The API server uses your existing Gemini configuration from the main `.env` file:

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GEMINI_MODEL` | Yes | Gemini model name |
| `LLM_TEMPERATURE` | No | Model temperature (default: 0.0) |
| `LLM_MAX_TOKENS` | No | Max response tokens (default: 2048) |

## Benefits of This Integration

### 1. **Flexibility**
- Use either direct Gemini integration or OpenAI-compatible API
- Switch between modes without changing your application logic
- Support for both local and remote API endpoints

### 2. **Compatibility**
- Works with any OpenAI-compatible client or tool
- Maintains existing Gemini functionality
- No breaking changes to current workflows

### 3. **Production Ready**
- Authentication and rate limiting
- Proper error handling and logging
- Health checks and monitoring endpoints
- Streaming support for real-time responses

### 4. **Development Friendly**
- Auto-generated API keys in development mode
- Interactive API documentation
- Detailed logging and debugging support

## Troubleshooting

### Common Issues

#### 1. API Server Won't Start

**Error**: `Address already in use`
**Solution**: Change the port in `.env.api` or kill the process using port 8001:

```bash
lsof -ti:8001 | xargs kill -9
```

#### 2. Authentication Errors

**Error**: `401 Unauthorized`
**Solution**: Check that you're using the correct API key. In development mode, check the logs for the auto-generated key.

#### 3. Gemini API Errors

**Error**: `Failed to generate completion`
**Solution**: Verify your `GEMINI_API_KEY` and `GEMINI_MODEL` settings in your main `.env` file.

#### 4. Model Not Found

**Error**: `Model 'xyz' is not available`
**Solution**: Use `gzm-research` as the model name when calling the API.

### Debugging

Enable verbose logging:

```bash
# For CLI
llm-query ask --question "test" --use-api --verbose

# For API server, set in .env.api:
LOG_LEVEL=DEBUG
```

Check server logs for detailed error information.

### Health Checks

Verify the server is running properly:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/info
```

## Next Steps

1. **Custom Models**: Extend the service to support multiple Gemini models
2. **Enhanced Prompting**: Integrate survey metadata and templates into the API calls
3. **Caching**: Add response caching for improved performance
4. **Monitoring**: Add metrics and monitoring for production deployment
5. **Docker**: Create Docker containers for easy deployment