from google import genai
from google.genai import types
import os
import dotenv
import time
from .Base import BaseModel

dotenv.load_dotenv()


class Gemini(BaseModel):
    def __init__(
            self,
            model_name="gemini-3-flash-preview",
            temperature=0,
            input_price=0.5,
            output_price=3.0):
        self.client = genai.Client(api_key=os.getenv("Google_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.input_price = input_price
        self.output_price = output_price

    def prompt(self, processed_input):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=processed_input[0]['content'],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                candidate_count=1,
            )
        )

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.total_token_count - usage.prompt_token_count

        price = prompt_tokens * self.input_price + completion_tokens * self.output_price

        return response.text, prompt_tokens, completion_tokens, price


class GeminiPro(Gemini):
    def __init__(
            self,
            model_name="gemini-3-pro-preview",
            input_price=2.0,
            output_price=12.0,
            **kwargs):
        super().__init__(
            model_name=model_name,
            input_price=input_price,
            output_price=output_price,
            **kwargs)