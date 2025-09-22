'''
Main application for dynamic LLM routing using LangGraph
'''
import asyncio
from typing import Dict, List, Any
from classifier import classify_text
from fallback import FallbackChatGradientAI
from semantic_cache import SemanticCache
from langgraph_router import Router, RouterState

# Model configuration
MODELS_CONFIG = {
    "tier1": [
        ["qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free"],
        ["llama-3.3-8b-instruct", "meta-llama/llama-3.3-8b-instruct:free"],
        # ["mistral-7b-instruct", "mistralai/mistral-7b-instruct:free"],
        ["qwen2.5-vl-32b-instruct", "qwen/qwen2.5-vl-32b-instruct:free"],
        
        ["gpt-oss-20b", "openai/gpt-oss-20b:free"],
        ["devstral-small-2505", "mistralai/devstral-small-2505:free"],
        ["qwq-32b", "qwen/qwq-32b:free"],
        ["qwen-2.5-coder-32b-instruct", "qwen/qwen-2.5-coder-32b-instruct:free"],
        ["deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-llama-70b:free"],
        ["llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free"],
    ],
    "tier2": [
        ["mistral-7b-instruct", "mistralai/mistral-7b-instruct:free"],
        ["qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free"],
        ["gpt-oss-20b", "openai/gpt-oss-20b:free"],
        ["devstral-small-2505", "mistralai/devstral-small-2505:free"],
        ["qwq-32b", "qwen/qwq-32b:free"],
        ["qwen-2.5-coder-32b-instruct", "qwen/qwen-2.5-coder-32b-instruct:free"],
        ["deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-llama-70b:free"],
        ["llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free"],
    ],
    "tier3": [
        ["qwen-2.5-coder-32b-instruct", "qwen/qwen-2.5-coder-32b-instruct:free"],
        ["deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-llama-70b:free"],
        ["llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free"],
        ["qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free"],
        ["gpt-oss-20b", "openai/gpt-oss-20b:free"],
        ["devstral-small-2505", "mistralai/devstral-small-2505:free"],
        ["qwq-32b", "qwen/qwq-32b:free"],
    ]
}

class Classifier:
    """Wrapper for the classifier to match the expected interface."""
    
    def __init__(self):
        self.classify_text = classify_text

class LLMClient:
    """Wrapper for the LLM client to match the expected interface."""
    
    def __init__(self, models_config: Dict[str, List[List[str]]]):
        self.models_config = models_config
        # Initialize fallback handlers for each tier
        self.fallback_handlers = {
            tier: FallbackChatGradientAI(models=models_list)
            for tier, models_list in models_config.items()
        }
    
    async def call(self, model: str, messages: List[Dict[str, str]], tier: str) -> str:
        """Call the LLM with the given messages and force correct tier fallback."""
        # Extract the query (assume one user message)
        query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

        # Pick the correct fallback handler (default = tier1 if not found)
        fallback = self.fallback_handlers.get(tier, self.fallback_handlers["tier1"])

        # Call the model
        response = fallback.invoke(query)

        # Return clean response
        return response[0] if isinstance(response, (list, tuple)) else response

async def process_query(query: str, router: Router) -> None:
    """Process a single query using the router."""
    print(f"\n{'='*50}")
    print(f"Processing query: {query}")
    
    # Route the query
    result = await router.route(query)
    
    # Process the result
    if result.get("cache_hit"):
        print("Cache hit! Using cached response.")
    else:
        print(f"Classification: {result.get('classification')}")
        print(f"Selected model: {result.get('selected_model')}")
    
    if result.get("error"):
        print(f"Error: {result['error']}")
    
    if result.get("llm_response"):
        print(f"\nResponse from {result.get('used_model', 'unknown model')}:")
        print("-" * 50)
        print(result["llm_response"])
        print("-" * 50)

async def main():
    # Initialize components
    cache = SemanticCache(default_ttl=600)  # 10 minute TTL
    classifier = Classifier()
    llm_client = LLMClient(MODELS_CONFIG)
    
    # Create the router
    router = Router(
        models_config={
            "tier1": [m[1] for m in MODELS_CONFIG["tier1"]],
            "tier2": [m[1] for m in MODELS_CONFIG["tier2"]],
            "tier3": [m[1] for m in MODELS_CONFIG["tier3"]],
        },
        cache=cache,
        classifier=classifier,
        llm_client=llm_client,
        max_retries=3
    )
    
    # Example queries
    queries = [
        "What is the capital of Ghana?",
        #"Explain quantum computing in simple terms.",
        #"Create code for a simple weather application that takes a city name and displays the current temperature by calling a weather API.",
        #"Develop a multi-step plan to reduce carbon emissions in a mid-sized city, considering economic, social, and political factors."
    ]
    
    # Process each query
    for query in queries:
        await process_query(query, router)

def testt():
    print("testttt")

if __name__ == "__main__":
    asyncio.run(main())
        