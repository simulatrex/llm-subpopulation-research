import json
import os
import time
import requests
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum
import instructor
from openai import OpenAI
from dotenv import load_dotenv

# Default prompt that models will use as a base context.
DEFAULT_SYSTEM_PROMPT = """Please provide an objective response."""

# Load environment variables from a .env file.
load_dotenv()


class CognitiveModel(Enum):
    # List of different language models available for use.
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    LLAMA_2_70b_DEPRECATED = "llama2_70b"
    LLAMA_2_70b = "meta-llama/llama-2-70b-chat"
    MISTRAL_7b_INSTRUCT = "mistralai/mistral-7b-instruct"
    MIXTRAL_8x7B = "mistralai/mixtral-8x7b"
    PHI_2 = "microsoft/phi-2"
    CLAUDE_2 = "anthropic/claude-2"
    GEMINI_PRO = "google/gemini-pro"


# Abstract Base Class for language models.
class BaseLanguageModel(ABC):
    @abstractmethod
    def ask(
        self, prompt: str, context_prompt=DEFAULT_SYSTEM_PROMPT, temperature=1.0
    ) -> str:
        # Method to send a prompt to the model and get a response.
        pass

    @abstractmethod
    def generate_structured_output(
        self,
        prompt: str,
        response_model: BaseModel,
        context_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=1.0,
    ) -> dict:
        # Method to generate a structured output from the model.
        pass


# Wrapper class for the OpenAI API.
class OpenAILanguageModel(BaseLanguageModel):
    """
    This is a wrapper for the OpenAI API.
    """

    def __init__(self, model_id=CognitiveModel.GPT_4_TURBO):
        # Initialize with a specific cognitive model.
        self.model_id = model_id
        self.client = OpenAI()

        # Apply any necessary patches to the OpenAI client.
        instructor.patch(self.client)

    def ask(
        self, prompt: str, context_prompt=DEFAULT_SYSTEM_PROMPT, temperature=1.0
    ) -> str:
        # Ask the language model a question and return its response.
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model_id.value,
                temperature=temperature,
            )
            # Extracting the content of the response from the API call.
            response_message = chat_completion.choices[0].message.content

            if response_message == None:
                return Exception("Empty response from OpenAI API")

            return response_message
        except Exception as e:
            print(f"Error: {e}")
            return Exception("Error in OpenAI API call")

    def generate_structured_output(
        self,
        prompt: str,
        response_model: BaseModel,
        context_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=1.0,
    ):
        # Generate a structured output from the language model.
        try:
            structured_response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
                model=self.model_id.value,
                temperature=temperature,
            )

            return structured_response
        except Exception as e:
            print(f"Error: {e}")
            return Exception("Error in OpenAI API call")

    def prompt_image_response(self, prompt: str, image_url: str):
        # Ask the vision model a question and return its response.
        try:
            vision_completion = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
            return vision_completion
        except Exception as e:
            print(f"Error: {e}")
            return Exception("Error in OpenAI API call")


class OpenRouterProxyLanguageModel(BaseLanguageModel):
    """
    This class is a wrapper for the OpenRouter API.
    """

    def __init__(self, model_id=CognitiveModel.LLAMA_2_70b):
        access_token = os.environ.get("OPENROUTER_API_KEY")

        if access_token is None:
            return Exception(
                "No OpenRouter API key found. Please set OPENROUTER_API_KEY as environment variable."
            )

        self.access_token = access_token
        # Initialize with a specific cognitive model.
        self.model_id = model_id

    def ask(self, prompt: str) -> str:
        API_URL = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url=API_URL,
            headers=headers,
            json={
                "model": self.model_id.value,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

        response.raise_for_status()  # This will raise an exception if the HTTP request returned an unsuccessful status code
        while not response.content:
            time.sleep(0.1)  # Wait a bit for the response content to be received

        result = response.json()
        result_text = result["choices"][0]["message"]["content"]

        return result_text

    def generate_structured_output(
        self,
        prompt: str,
        response_model: BaseModel,
        context_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=1.0,
    ) -> dict:
        return response_model
