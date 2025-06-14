import os
import threading
import json
import functools # Used for @functools.lru_cache

# --- Global Configuration for LLM Providers ---
# Defines the base configurations for known providers.
# API keys are loaded from environment variables.
# LM Studio's api_base here is a general default; actual base is built from agent's specific port.
GLOBAL_PROVIDER_CONFIGS = {
    "lm_studio": {
        "display_name": "LM Studio (Local)", # User-friendly name
        "api_base": "http://localhost:1234/v1", # Default/fallback base for LM Studio
        "api_key": "lm-studio" # Dummy key for LM Studio
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "display_name": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    # Future support for other local model launchers like Ollama
    "ollama": {
        "display_name": "Ollama (Local)",
        "api_base": "http://localhost:11434/api", # Default Ollama API base
        # Ollama typically doesn't require an API key for local access
        # "api_key_env": "OLLAMA_API_KEY", # Uncomment if your Ollama setup needs it
    },
    # Future support for lmdeploy
    "lmdeploy": {
        "display_name": "LMDeploy (Local)",
        "api_base": "http://localhost:23333/v1", # Default lmdeploy API base
        # "api_key_env": "LMDEPLOY_API_KEY", # Uncomment if your lmdeploy setup needs it
    }
}

# --- Settings File Management ---
SETTINGS_FILE = 'model_settings.json'
_settings_lock = threading.Lock() # For thread-safe loading of settings

def _get_default_fallback_settings():
    """Provides a safe, minimal default settings structure if file loading fails."""
    return {
        "provider": "lm_studio", # Default global provider
        "overrides": {
            "default": { # Essential default override
                "provider": "lm_studio",
                "model": "Mistral-Nemo-Instruct-12B",
                "port": 1234
            }
        }
    }

@functools.lru_cache(maxsize=1) # Apply lru_cache to memoize the settings loading
def _load_settings():
    """
    Loads model settings from the JSON file into a global cache, with error handling.
    This function is memoized by lru_cache.
    """
    with _settings_lock: # Protects the initial file read and potential modification of settings
        try:
            if not os.path.exists(SETTINGS_FILE):
                print(f"WARNING: Settings file '{SETTINGS_FILE}' not found. Using default fallback settings.")
                return _get_default_fallback_settings()
            
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                
                # Validate basic structure for main keys
                if "provider" not in settings or "overrides" not in settings:
                    raise ValueError("Settings file is malformed: missing 'provider' or 'overrides' key.")
                
                # Validate 'default' override presence
                if "default" not in settings["overrides"]:
                    print(f"WARNING: No 'default' entry found in 'overrides' in {SETTINGS_FILE}. Using internal default for unassigned agents.")
                    # Add a default if missing in settings.json itself
                    settings["overrides"]["default"] = _get_default_fallback_settings()["overrides"]["default"]

                print(f"[Model Router] Loaded settings from {SETTINGS_FILE}")
                return settings

        except json.JSONDecodeError as e:
            print(f"ERROR: Malformed JSON in '{SETTINGS_FILE}': {e}. Using default fallback settings.")
            return _get_default_fallback_settings()
        except ValueError as e:
            print(f"ERROR: Invalid settings structure in '{SETTINGS_FILE}': {e}. Using default fallback settings.")
            return _get_default_fallback_settings()
        except Exception as e:
            print(f"ERROR: Unexpected error loading settings from '{SETTINGS_FILE}': {e}. Using default fallback settings.")
            return _get_default_fallback_settings()

# Initialize settings cache on module import to ensure it's ready
_load_settings()


# --- Main Model Resolution Logic ---

def get_model_details_for_agent(agent_name: str) -> dict:
    """
    Returns the API base URL, model name, API key, and provider name for a given agent
    based on the loaded settings, applying per-agent overrides.
    """
    settings = _load_settings() # This now correctly returns the cached settings dictionary
    
    # Determine the configuration for the specific agent
    overrides = settings.get("overrides", {})
    if not isinstance(overrides, dict):
        # This error should be caught by _load_settings's validation, but as a defensive check
        raise ValueError(f"Expected 'overrides' to be a dict but got {type(overrides)}: {overrides}")
    
    agent_config = overrides.get(agent_name)
    if agent_config is None:
        agent_config = overrides.get("default")
        if agent_config is None: # Should not happen if _load_settings is robust, but for safety
            print(f"CRITICAL ERROR: No specific or default override for '{agent_name}' and no internal default could be found. Using hardcoded fallback.")
            agent_config = {"provider": "lm_studio", "model": "Mistral-Nemo-Instruct-12B", "port": 1234}

    # Determine the effective provider for this agent
    effective_provider_name = agent_config.get("provider")
    if effective_provider_name is None:
        # If agent override doesn't specify provider, use the global default provider
        effective_provider_name = settings.get("provider", "lm_studio") 
        print(f"WARNING: Provider not specified for agent '{agent_name}' in settings. Using global default provider: '{effective_provider_name}'.")

    # Get base config for the chosen effective provider
    provider_base_config = GLOBAL_PROVIDER_CONFIGS.get(effective_provider_name)
    if not provider_base_config:
        print(f"ERROR: Configuration for provider '{effective_provider_name}' not found in GLOBAL_PROVIDER_CONFIGS. Falling back to LM Studio default.")
        provider_base_config = GLOBAL_PROVIDER_CONFIGS["lm_studio"]
        effective_provider_name = "lm_studio" # Revert provider name too

    api_base = provider_base_config["api_base"]
    api_key = provider_base_config.get("api_key") # For LM Studio dummy key, or directly from config
    
    model_name = agent_config.get("model")
    if model_name is None:
        print(f"WARNING: Model name not specified for agent '{agent_name}' under provider '{effective_provider_name}' in settings. Using generic fallback for display purposes.")
        model_name = "generic-model-fallback" # A placeholder for missing model names

    if effective_provider_name == "lm_studio":
        custom_port = agent_config.get("port")
        custom_host = agent_config.get("host", "localhost") # Get host, default to 'localhost'
        if custom_port:
            api_base = f"http://{custom_host}:{custom_port}/v1" # Use custom_host
        # If no custom_port, it defaults to GLOBAL_PROVIDER_CONFIGS["lm_studio"]["api_base"] (which implies localhost:1234)
        
    elif effective_provider_name in ["openrouter", "openai", "ollama", "lmdeploy"]: # Include new local providers
        api_key_env_var = provider_base_config.get("api_key_env")
        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)
            if not api_key and api_key_env_var: # Only warn if an env var was expected
                print(f"WARNING: API key for '{effective_provider_name}' not found in environment variable '{api_key_env_var}'. LLM calls may fail.")
        elif "api_key" in provider_base_config: # Direct key from config (e.g., LM Studio dummy)
            api_key = provider_base_config["api_key"]
        # If no api_key_env and no api_key, api_key remains None. This is fine for some local setups.


    return {
        "provider": effective_provider_name, # New: return the actual provider name
        "api_base": api_base,
        "model_name": model_name,
        "api_key": api_key
    }

def list_agent_model_assignments() -> dict:
    """
    Returns a summary of which agent is using which model and provider,
    based on the loaded settings. This is useful for UI display.
    """
    settings = _load_settings()
    assignments = {}

    # Get all agent names from the overrides, including 'default' for conceptual display
    all_agent_names_in_settings = list(settings["overrides"].keys())
    
    # We'll use a placeholder agent name to trigger the default fallback logic for display
    # This helps ensure the "Default Fallback" entry is consistent with get_model_details_for_agent
    
    for agent_name in sorted(all_agent_names_in_settings):
        # Skip "default" as a standalone agent for the first loop
        if agent_name == "default":
            continue

        details = get_model_details_for_agent(agent_name)
        
        # Use the actual provider name returned by get_model_details_for_agent
        display_provider_name = GLOBAL_PROVIDER_CONFIGS.get(details["provider"], {}).get("display_name", details["provider"])

        assignments[agent_name] = {
            "provider": display_provider_name,
            "model": details["model_name"],
            "api_base": details["api_base"]
        }
    
    # Add the "Default Fallback" assignment using the logic from get_model_details_for_agent
    default_details = get_model_details_for_agent("NON_EXISTENT_AGENT_TO_TRIGGER_DEFAULT_FALLBACK") # Use a dummy name
    display_default_provider_name = GLOBAL_PROVIDER_CONFIGS.get(default_details["provider"], {}).get("display_name", default_details["provider"])

    assignments["Default Fallback (Unassigned Agents)"] = {
        "provider": display_default_provider_name,
        "model": default_details["model_name"],
        "api_base": default_details["api_base"]
    }

    return assignments


# Example Usage (for testing purposes when run directly)
if __name__ == "__main__":
    print("--- Testing get_model_details_for_agent ---")

    # Set up dummy environment variables for testing API providers
    os.environ['OPENROUTER_API_KEY'] = 'test_openrouter_api_key_123'
    os.environ['OPENAI_API_KEY'] = 'test_openai_api_key_456'
    # os.environ['OLLAMA_API_KEY'] = 'test_ollama_key' # Uncomment if using Ollama API key

    print("\nDream Weaver details (LM Studio override):")
    dw_details = get_model_details_for_agent("Dream Weaver")
    print(dw_details)
    assert dw_details["provider"] == "lm_studio"
    assert dw_details["model_name"] == "mistralai/mistral-nemo-instruct-2407"
    assert dw_details["api_base"] == "http://localhost:1234/v1"
    assert dw_details["api_key"] == "lm-studio"

    print("\nCode Sage details (LM Studio override):")
    cs_details = get_model_details_for_agent("Code Sage")
    print(cs_details)
    assert cs_details["provider"] == "lm_studio"
    assert cs_details["model_name"] == "qwen/qwen2.5-coder-14b"
    assert cs_details["api_base"] == "http://10.0.0.31:1234/v1" # This will fail if model_settings.json doesn't specify host
    assert cs_details["api_key"] == "lm-studio"

    print("\nQuality Guardian details (OpenRouter override):")
    qg_details = get_model_details_for_agent("Quality Guardian")
    print(qg_details)
    assert qg_details["provider"] == "openrouter"
    assert qg_details["model_name"] == "deepseek-ai/deepseek-v3-0324"
    assert qg_details["api_base"] == "https://openrouter.ai/api/v1"
    assert qg_details["api_key"] == "test_openrouter_api_key_123"

    print("\nMaster Builder details (OpenRouter override):")
    mb_details = get_model_details_for_agent("Master Builder")
    print(mb_details)
    assert mb_details["provider"] == "openrouter"
    assert mb_details["model_name"] == "mistralai/mixtral-8x7b-instruct"
    assert mb_details["api_base"] == "https://openrouter.ai/api/v1"
    assert mb_details["api_key"] == "test_openrouter_api_key_123"

    print("\nNon-existent Agent details (should use default override):")
    ne_details = get_model_details_for_agent("NonExistentAgent")
    print(ne_details)
    assert ne_details["provider"] == "openrouter" # Based on model_settings.json default
    assert ne_details["model_name"] == "google/gemma-7b-it"
    assert ne_details["api_base"] == "https://openrouter.ai/api/v1"
    assert ne_details["api_key"] == "test_openrouter_api_key_123"

    # Test an agent with missing 'provider' in its override (should use global default from settings.json)
    # This portion will be affected by _load_settings() returning a normal dict and _load_settings.cache_clear() not being available.
    # We will remove this in main to avoid direct manipulation of global state.
    # The testing of this scenario will be more robustly handled by test_ai_app_builder.py's mocking of _load_settings.


    print("\n--- Testing list_agent_model_assignments ---")
    assignments_summary = list_agent_model_assignments()
    for agent, details in assignments_summary.items():
        print(f"{agent}: Provider='{details['provider']}', Model='{details['model']}', API Base='{details['api_base']}'")
    
    assert "Dream Weaver" in assignments_summary
    assert assignments_summary["Dream Weaver"]["model"] == "mistralai/mistral-nemo-instruct-2407"
    assert assignments_summary["Dream Weaver"]["provider"] == "LM Studio (Local)"

    assert "Code Sage" in assignments_summary
    assert assignments_summary["Code Sage"]["model"] == "qwen/qwen2.5-coder-14b"
    assert assignments_summary["Code Sage"]["provider"] == "LM Studio (Local)"

    assert "Quality Guardian" in assignments_summary
    assert assignments_summary["Quality Guardian"]["model"] == "deepseek-ai/deepseek-v3-0324"
    assert assignments_summary["Quality Guardian"]["provider"] == "OpenRouter"

    assert "Default Fallback (Unassigned Agents)" in assignments_summary
    assert assignments_summary["Default Fallback (Unassigned Agents)"]["model"] == "google/gemma-7b-it"
    assert assignments_summary["Default Fallback (Unassigned Agents)"]["provider"] == "OpenRouter"

    # Clean up dummy environment variables
    del os.environ['OPENROUTER_API_KEY']
    del os.environ['OPENAI_API_KEY']