import os
import logging
import requests
from typing import ClassVar
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeHaikuLLM(LLM):
    """
    Custom LLM class to interface with Claude Haiku via API.
    """
    api_endpoint: ClassVar[str] = os.getenv("CLAUDE_HAIKU_API_ENDPOINT", "https://api.anthropic.com/v1/complete")
    api_key: ClassVar[str] = os.getenv("CLAUDE_HAIKU_API_KEY")
    max_tokens: int = 1024
    temperature: float = 0.7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            logger.error("Claude Haiku API key is missing. Please set the CLAUDE_HAIKU_API_KEY environment variable.")
            raise ValueError("API key is missing")

    @property
    def _llm_type(self):
        return "claude_haiku"

    def _call(self, prompt, stop=None):
        """
        Send a request to the Claude Haiku API and return the response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "claude-haiku",
            "prompt": prompt,
            "max_tokens_to_sample": self.max_tokens,
            "temperature": self.temperature,
        }

        if stop:
            payload["stop_sequences"] = stop

        try:
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            result = response.json()

            logger.debug(f"API Response: {result}")

            # Extract generated text from the response
            return result["completion"].strip()
        except requests.RequestException as e:
            logger.error(f"Error during API call: {e}")
            raise RuntimeError(f"Failed to call Claude Haiku API: {e}")

if __name__ == "__main__":
    try:
        llm = ClaudeHaikuLLM()
        prompt = "Generate a Python program to draw a line"
        response = llm.invoke(prompt)
        print(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error initializing ClaudeHaikuLLM or invoking prompt: {e}")
