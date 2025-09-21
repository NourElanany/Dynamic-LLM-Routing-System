import os
import time
import requests
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Map models to keys
MODEL_KEY_MAP = {
    "qwen-2.5-72b-instruct": "qwen-2.5-72b-instruct",
    "mistral-7b-instruct": "mistral-7b-instruct",
    "llama-3.3-8b-instruct": "llama-3.3-8b-instruct",
}

# Price per 1M tokens
PRICES = {
    "qwen-2.5-72b-instruct": {"input": 0.7, "output": 0.7},
    "mistral-7b-instruct": {"input": 0.25, "output": 0.25},
    "llama-3.3-8b-instruct": {"input": 0.15, "output": 0.15},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calc cost based on token usage"""
    if model_name not in PRICES:
        return 0.0
    price = PRICES[model_name]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


class FallbackChatGradientAI:
    """Try multiple models until success"""

    def __init__(self, models: list[str]):
        self.models = models
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def invoke(self, prompt: str, max_retries: int = 3):
        """Send prompt, fallback if model fails"""
        last_exception = None

        for model_name in self.models:
            api_key_name = MODEL_KEY_MAP.get(model_name)
            api_key = os.getenv(api_key_name)

            if not api_key:
                continue  # Skip if no key

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    response = requests.post(self.base_url, headers=headers, json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        choice = result["choices"][0]["message"]["content"]

                        usage = result.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                        cost = calculate_cost(model_name, input_tokens, output_tokens)
                        end_time = time.time()

                        return {
                            "model": model_name,
                            "response": choice,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost": cost,
                            "time_taken": end_time - start_time,
                        }

                    else:
                        last_exception = Exception(
                            f"Error {response.status_code}: {response.text}"
                        )

                except Exception as e:
                    last_exception = e

        raise Exception(f"All models failed. Last error: {str(last_exception)}")
