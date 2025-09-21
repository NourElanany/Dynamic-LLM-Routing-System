import pandas as pd
from semantic_cache import SemanticCache
from classifier import classify_text
from fallback import FallbackChatGradientAI

class TestSuite:
    def __init__(self):
        self.cache = SemanticCache()

        # Define models for different tiers

        self.models = {
            "tier1": [
                ["llama-3.3-8b-instruct", "meta-llama/llama-3.3-8b-instruct:free"],
                ["mistral-7b-instruct", "mistralai/mistral-7b-instruct:free"],
                ["qwen2.5-vl-32b-instruct", "qwen/qwen2.5-vl-32b-instruct:free"]
            ],
            "tier2": [
                ["qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free"],
                ["gpt-oss-20b", "openai/gpt-oss-20b:free"]
            ],
            "tier3": [
                ["llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free"],
                ["deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-llama-70b:free"]
            ]
        }


        self.results = []



    def _check_cache(self, query):
        """Check if query exists in cache and return cached response"""
        return self.cache.get(query)

    def _is_cache_hit(self, cached_response):
        """Check if we got a cache hit"""
        return cached_response is not None

    def _get_cache_result(self, cached_response):
        """Prepare cache hit result data"""
        return {
            "response": cached_response,
            "route": "cache",
            "used_model": "cache",
            "speed": 0.1,
            "accuracy": 1.0,
            "cost": 0.0
        }