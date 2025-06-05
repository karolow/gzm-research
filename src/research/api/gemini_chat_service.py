"""
Bridge service to integrate Gemini with OpenAI API blueprint.
This service translates between OpenAI chat completion format and Gemini API,
always treating requests as SQL generation using survey metadata and prompts.
"""

import asyncio
import re
import time
import uuid
from typing import AsyncGenerator, List, Optional

from openai_api_blueprint.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    UsageInfo,
)
from openai_api_blueprint.services.chat_service import ChatService

from research.config import get_config
from research.db.operations import query_duckdb
from research.llm.gemini import GeminiClient
from research.llm.sql_generator import (
    load_metadata,
    create_system_prompt,
    extract_sql_from_response,
)
from research.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiChatService(ChatService):
    """
    Chat service that bridges OpenAI chat completion format with Google Gemini API.
    """

    def __init__(self):
        """Initialize the Gemini chat service."""
        self.config = get_config()
        self.client = GeminiClient(
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        logger.info(f"Initialized GeminiChatService with model: {self.config.llm.model}")

    def _clean_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text

    def _convert_messages_to_gemini_format(self, messages: List[dict]) -> tuple[str, str]:
        """
        Convert OpenAI messages format to Gemini format with SQL generation context.

        Args:
            messages: List of OpenAI format messages

        Returns:
            Tuple of (system_instruction, user_message)
        """
        system_messages = []
        user_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                system_messages.append(content)
            elif role == "user":
                user_messages.append(f"User: {content}")
            elif role == "assistant":
                user_messages.append(f"Assistant: {content}")

        # Always use SQL generation system prompt with survey metadata
        try:
            # Load survey metadata and create SQL generation prompt
            metadata_content = load_metadata(self.config.survey_metadata)
            sql_system_prompt = create_system_prompt(
                metadata_content,
                self.config.sql_prompt_template
            )
            system_instruction = sql_system_prompt
        except Exception as e:
            logger.warning(f"Failed to load SQL context: {e}, falling back to basic prompt")
            system_instruction = """You are a SQL expert. Generate clean, efficient SQL queries for DuckDB.
            Focus on survey data analysis. Return only the SQL query without explanations."""

        # Combine user messages into conversation format
        user_message = "\n".join(user_messages) if user_messages else ""

        return system_instruction, user_message

    async def generate_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion response using Gemini.

        Args:
            request: OpenAI format chat completion request

        Returns:
            OpenAI format chat completion response
        """
        try:
            # Always process as SQL generation request
            messages_dict = [msg.model_dump() for msg in request.messages]
            
            logger.info("Processing request as SQL generation")

            # Convert messages to Gemini format with SQL context
            system_instruction, user_message = self._convert_messages_to_gemini_format(
                messages_dict
            )

            logger.debug(f"System instruction length: {len(system_instruction)} chars")
            logger.debug(f"User message: {user_message}")

            # Run Gemini client in executor to avoid blocking
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None,
                self.client.generate_content,
                system_instruction,
                user_message
            )

            # Extract and clean the SQL from response
            sql_query = extract_sql_from_response(response_text)
            if sql_query:
                # Clean HTML tags from SQL query
                sql_query = self._clean_html_tags(sql_query)
                logger.info(f"Extracted SQL query: {sql_query[:100]}...")
                
                # Execute the SQL query and format results as markdown
                try:
                    result_df = query_duckdb(self.config.db.default_path, sql_query)
                    
                    if result_df.empty:
                        response_text = "## SQL Query Executed\n\n"
                        response_text += "```sql\n"
                        response_text += sql_query + "\n"
                        response_text += "```\n\n"
                        response_text += "### Result\n"
                        response_text += "Query executed successfully but returned no results."
                    else:
                        # Format results as markdown with table
                        row_count = len(result_df)
                        col_count = len(result_df.columns)
                        
                        # Convert DataFrame to markdown table with better formatting
                        df_display = result_df.head(50) if row_count > 50 else result_df
                        
                        # Create markdown table manually for better control
                        table_lines = []
                        # Header
                        headers = "| " + " | ".join(str(col) for col in df_display.columns) + " |"
                        table_lines.append(headers)
                        # Separator
                        separator = "|" + "|".join("---" for _ in df_display.columns) + "|"
                        table_lines.append(separator)
                        # Data rows
                        for _, row in df_display.iterrows():
                            row_str = "| " + " | ".join(str(val) if val is not None else "" for val in row) + " |"
                            table_lines.append(row_str)
                        
                        markdown_table = "\n".join(table_lines)
                        
                        result_text = "## SQL Query Executed\n\n"
                        result_text += "```sql\n"
                        result_text += sql_query + "\n"
                        result_text += "```\n\n\n\n"
                        result_text += "### Results\n"
                        result_text += f"Query returned **{row_count}** rows with **{col_count}** columns.\n\n"
                        result_text += markdown_table
                        
                        if row_count > 50:
                            result_text += f"\n\n*... and {row_count - 50} more rows.*"
                        
                        response_text = result_text
                        
                except Exception as e:
                    logger.error(f"Failed to execute SQL query: {e}")
                    response_text = "## SQL Query Generated\n\n"
                    response_text += "```sql\n"
                    response_text += sql_query + "\n"
                    response_text += "```\n\n"
                    response_text += "### Error\n"
                    response_text += f"❌ **Error executing query:** {str(e)}"

            # Create OpenAI format response
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())

            response = ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=created_timestamp,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatCompletionResponseMessage(
                            role="assistant",
                            content=response_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=len(user_message.split()),  # Rough estimate
                    completion_tokens=len(response_text.split()),  # Rough estimate
                    total_tokens=len(user_message.split()) + len(response_text.split())
                )
            )

            logger.info(f"Generated completion with {len(response_text)} characters")
            return response

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    async def generate_streaming_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a streaming chat completion response using Gemini.

        Args:
            request: OpenAI format chat completion request

        Yields:
            Dictionary chunks formatted as OpenAI streaming responses
        """
        try:
            # Always process as SQL generation request
            messages_dict = [msg.model_dump() for msg in request.messages]
            
            # Convert messages to Gemini format with SQL context
            system_instruction, user_message = self._convert_messages_to_gemini_format(
                messages_dict
            )

            # Generate full response first (Gemini doesn't support true streaming)
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None,
                self.client.generate_content,
                system_instruction,
                user_message
            )

            # Extract and clean the SQL from response
            sql_query = extract_sql_from_response(response_text)
            if sql_query:
                # Clean HTML tags from SQL query
                sql_query = self._clean_html_tags(sql_query)
                
                # Execute the SQL query and format results for streaming as markdown
                try:
                    result_df = query_duckdb(self.config.db.default_path, sql_query)
                    
                    if result_df.empty:
                        response_text = "## SQL Query Executed\n\n"
                        response_text += "```sql\n"
                        response_text += sql_query + "\n"
                        response_text += "```\n\n"
                        response_text += "### Result\n"
                        response_text += "Query executed successfully but returned no results."
                    else:
                        # Format results as markdown with table
                        row_count = len(result_df)
                        col_count = len(result_df.columns)
                        
                        # Convert DataFrame to markdown table with better formatting
                        df_display = result_df.head(50) if row_count > 50 else result_df
                        
                        # Create markdown table manually for better control
                        table_lines = []
                        # Header
                        headers = "| " + " | ".join(str(col) for col in df_display.columns) + " |"
                        table_lines.append(headers)
                        # Separator
                        separator = "|" + "|".join("---" for _ in df_display.columns) + "|"
                        table_lines.append(separator)
                        # Data rows
                        for _, row in df_display.iterrows():
                            row_str = "| " + " | ".join(str(val) if val is not None else "" for val in row) + " |"
                            table_lines.append(row_str)
                        
                        markdown_table = "\n".join(table_lines)
                        
                        result_text = "## SQL Query Executed\n\n"
                        result_text += "```sql\n"
                        result_text += sql_query + "\n"
                        result_text += "```\n\n"
                        result_text += "### Results\n"
                        result_text += f"Query returned **{row_count}** rows with **{col_count}** columns.\n\n"
                        result_text += markdown_table
                        
                        if row_count > 50:
                            result_text += f"\n\n*... and {row_count - 50} more rows.*"
                        
                        response_text = result_text
                        
                except Exception as e:
                    logger.error(f"Failed to execute SQL query: {e}")
                    response_text = "## SQL Query Generated\n\n"
                    response_text += "```sql\n"
                    response_text += sql_query + "\n"
                    response_text += "```\n\n"
                    response_text += "### Error\n"
                    response_text += f"❌ **Error executing query:** {str(e)}"

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_timestamp = int(time.time())

            # Simulate streaming by chunking the response
            words = response_text.split()
            chunk_size = 3  # Words per chunk

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_content = " " + " ".join(chunk_words) if i > 0 else " ".join(chunk_words)

                stream_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk_content
                            },
                            "finish_reason": None
                        }
                    ]
                }

                yield stream_chunk

                # Small delay to simulate streaming
                await asyncio.sleep(0.05)

            # Send final chunk with finish reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }

            yield final_chunk

            logger.info(f"Completed streaming response with {len(words)} words")

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            # Send error in OpenAI format
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": "service_unavailable"
                }
            }
            yield error_chunk
