"""
OpenAI-compatible API module for GZM research project.

This module provides a FastAPI-based OpenAI-compatible API server
powered by Google Gemini, allowing the use of standard OpenAI clients
with the existing Gemini backend.

Main components:
- GeminiChatService: Bridge service between OpenAI format and Gemini API
- FastAPI app: OpenAI-compatible API server with authentication and rate limiting

Usage:
    # Start the API server
    python -m research.api.app
    
    # Or use the CLI command
    api-server
"""

from research.api.app import app, create_gzm_api_app, main
from research.api.gemini_chat_service import GeminiChatService

__all__ = [
    "app",
    "create_gzm_api_app", 
    "main",
    "GeminiChatService",
]