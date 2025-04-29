"""
Client for Google's Gemini API.
"""

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types as genai_types

from research.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GeminiClient:
    """Client for Google's Gemini API.

    top_p and top_k are set to Gemini API defaults (0.95 and 20) unless explicitly provided.
    """

    api_key: str
    model: str
    temperature: float
    max_tokens: int
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize the Gemini client and set defaults for top_p and top_k if not provided."""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.top_p is None:
            self.top_p = 0.95
        if self.top_k is None:
            self.top_k = 20
        self.client = genai.Client(api_key=self.api_key)
        logger.debug(f"Initialized Gemini client with model: {self.model}")

    def generate_content(self, system_instruction: str, user_prompt: str) -> str:
        """
        Generate content using the Gemini model.

        Args:
            system_instruction: System instruction for the model
            user_prompt: User prompt/question

        Returns:
            Generated text

        Raises:
            RuntimeError: If the API call fails
        """
        try:
            # Prepare generation config
            generation_config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_tokens,
            )

            # Check if model supports system instructions (Gemma models don't support this)
            if "gemma" not in self.model.lower():
                # Send the request with system instruction
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        max_output_tokens=self.max_tokens,
                    ),
                )
            else:
                # For Gemma models, include the system instruction as part of the user prompt
                combined_prompt = (
                    f"{system_instruction}\n\nUser question: {user_prompt}"
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=combined_prompt,
                    config=generation_config,
                )

            # Extract text from response, ensuring we always return a string
            if response and hasattr(response, "text"):
                response_text = response.text or ""  # Convert None to empty string
            else:
                response_text = ""
                logger.warning("Empty or invalid response from Gemini API")

            return response_text

        except Exception as e:
            logger.error(f"Error generating content: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate content: {e}") from e

    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        return bool(self.api_key) and bool(self.model)
