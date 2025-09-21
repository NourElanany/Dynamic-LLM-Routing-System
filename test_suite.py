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

    def _classify_query(self, query):
        """Classify the query to determine routing"""
        classification = classify_text(query)
        if isinstance(classification, dict):
            return classification.get("route", "tier1")
        else:
            return str(classification) if classification else "tier1"

    def _get_models_for_route(self, route):
        """Get model identifiers for a specific route"""
        return [m[1] for m in self.models.get(route, [])]

    def _get_fallback_models(self):
        """Get all models flattened from tier1 -> tier2 -> tier3 for fallback"""
        models_for_route = []
        for tier in ("tier1", "tier2", "tier3"):
            models_for_route.extend([m[1] for m in self.models.get(tier, [])])
        return models_for_route

    def _invoke_model(self, query, models_list):
        """Invoke the model using fallback mechanism"""
        fallback = FallbackChatGradientAI(models=models_list)
        return fallback.invoke(query)

    def _parse_model_response(self, model_response, models_list, start_time):
        """Parse the response from model invocation"""
        if isinstance(model_response, dict):
            return {
                "response": model_response.get("response", ""),
                "used_model": model_response.get("model", ""),
                "speed": model_response.get("time", time.time() - start_time),
                "tokens": model_response.get("tokens", 0),
                "cost": model_response.get("cost", 0.0)
            }
        else:
            return {
                "response": str(model_response),
                "used_model": models_list[0] if models_list else "unknown",
                "speed": time.time() - start_time,
                "tokens": 0,
                "cost": 0.0
            }

    def _handle_model_error(self, error, start_time):
        """Handle errors during model invocation"""
        print(f"Error getting response: {str(error)}")
        return {
            "response": f"Error: {str(error)}",
            "used_model": "error",
            "speed": time.time() - start_time,
            "accuracy": 0.0,
            "cost": 0.0
        }


    def _invoke_model(self, query, models_list):
        """Invoke the model using fallback mechanism"""
        fallback = FallbackChatGradientAI(models=models_list)
        return fallback.invoke(query)

    def _parse_model_response(self, model_response, models_list, start_time):
        """Parse the response from model invocation"""
        if isinstance(model_response, dict):
            return {
                "response": model_response.get("response", ""),
                "used_model": model_response.get("model", ""),
                "speed": model_response.get("time", time.time() - start_time),
                "tokens": model_response.get("tokens", 0),
                "cost": model_response.get("cost", 0.0)
            }
        else:
            return {
                "response": str(model_response),
                "used_model": models_list[0] if models_list else "unknown",
                "speed": time.time() - start_time,
                "tokens": 0,
                "cost": 0.0
            }

    def _handle_model_error(self, error, start_time):
        """Handle errors during model invocation"""
        print(f"Error getting response: {str(error)}")
        return {
            "response": f"Error: {str(error)}",
            "used_model": "error",
            "speed": time.time() - start_time,
            "accuracy": 0.0,
            "cost": 0.0
        }

    def _cache_response(self, query, response):
        """Cache the response for future use"""
        self.cache.set(query, response)

    def _calculate_accuracy(self, query, response):
        """A very simple token-based overlap accuracy"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        if not query_words:
            return 0.0
        return len(query_words.intersection(response_words)) / len(query_words)


    def _create_result_entry(self, query, response_data, route, cache_hit):
        """Create a result entry dictionary"""
        return {
            "Query": query,
            "Response": response_data["response"],
            "Route": route,
            "UsedModel": response_data["used_model"],
            "Speed": round(response_data["speed"], 2),
            "Accuracy": round(response_data.get("accuracy", self._calculate_accuracy(query, response_data["response"])), 2),
            "Cost": f"${response_data['cost']:.6f}",
            "Cache": "Hit" if cache_hit else "Miss"
        }

    def _save_result(self, result_entry):
        """Save the result entry to results list"""
        self.results.append(result_entry)


    def _handle_cache_flow(self, query):
        """Handle the complete cache flow"""
        cached_response = self._check_cache(query)
        cache_hit = self._is_cache_hit(cached_response)

        if cache_hit:
            return self._get_cache_result(cached_response), cache_hit

        return None, cache_hit

    def _determine_models_to_use(self, query):
        """Determine which models to use based on query classification"""
        route = self._classify_query(query)
        models_for_route = self._get_models_for_route(route)

        # If no models found for the classified route, fall back to all models
        if not models_for_route:
            models_for_route = self._get_fallback_models()

        return route, models_for_route


if __name__ == "__main__":
    test_suite = TestSuite()