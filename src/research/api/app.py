"""
FastAPI application integrating OpenAI API blueprint with Gemini backend.
This creates a local OpenAI-compatible API server powered by Google Gemini.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi

from openai_api_blueprint.core.errors import register_exception_handlers

from research.api.gemini_chat_service import GeminiChatService
from research.config import get_config
from research.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize our custom Gemini chat service
gemini_chat_service = GeminiChatService()

# Replace the chat service before importing the router
import openai_api_blueprint.services.chat_service
openai_api_blueprint.services.chat_service.chat_service = gemini_chat_service

# Now import the router after the service is replaced
from openai_api_blueprint.api.v1.router import v1_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting GZM OpenAI-compatible API server with Gemini backend")
    yield
    logger.info("Shutting down GZM API server")


def create_gzm_api_app() -> FastAPI:
    """
    Create FastAPI application with OpenAI blueprint integration.

    Returns:
        Configured FastAPI application
    """
    config = get_config()

    # Create app with lifespan
    app = FastAPI(
        title="GZM OpenAI-Compatible API",
        description="OpenAI-compatible API powered by Google Gemini for GZM research project",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure OpenAPI schema with security
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="GZM OpenAI-Compatible API",
            version="0.1.0",
            description="OpenAI-compatible API powered by Google Gemini for GZM research project",
            routes=app.routes,
        )
        openapi_schema["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter your API key (e.g., gzm-dev-key-12345678901234567890)"
            }
        }
        openapi_schema["security"] = [{"bearerAuth": []}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers for OpenAI-compatible error responses
    register_exception_handlers(app)

    # Include the OpenAI blueprint v1 router
    app.include_router(v1_router)

    # Health check endpoint
    @app.get("/health", tags=["Management"], status_code=status.HTTP_200_OK)
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "gzm-openai-api",
            "backend": "gemini",
            "model": config.llm.model
        }

    # Info endpoint
    @app.get("/info", tags=["Management"])
    async def info() -> dict[str, str]:
        """Service information endpoint."""
        return {
            "service": "GZM OpenAI-Compatible API",
            "backend": "Google Gemini",
            "model": config.llm.model,
            "provider": config.llm.provider,
        }

    logger.info("GZM API application created successfully")
    return app


# Create the app instance
app = create_gzm_api_app()

# Security scheme for Swagger UI
security_scheme = HTTPBearer(
    scheme_name="Bearer Authentication",
    description="Enter your API key with the 'Bearer ' prefix, e.g. 'Bearer your-api-key'",
)


def main() -> None:
    """Main function to start the API server."""
    import uvicorn

    config = get_config()

    # Default server configuration
    host = "127.0.0.1"
    port = 8001  # Different from default FastAPI port to avoid conflicts

    logger.info(f"Starting GZM OpenAI-compatible API server on {host}:{port}")
    logger.info(f"Using Gemini model: {config.llm.model}")
    logger.info(f"API docs available at: http://{host}:{port}/docs")

    uvicorn.run(
        "research.api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
