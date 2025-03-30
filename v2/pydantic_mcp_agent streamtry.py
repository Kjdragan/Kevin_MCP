from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from rich.text import Text
from dotenv import load_dotenv
import asyncio
import pathlib
import sys
import os
import argparse

from pydantic_ai import Agent
from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers import Provider, infer_provider

import mcp_client
from mcp_client import LLMProvider, ProviderConfig

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

load_dotenv()

# Available models by provider
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini"
    ],
    "anthropic": [
        "claude-3-7-sonnet-20250219",
        "claude-3-haiku-20240307"
    ],
    "gemini": [
        "gemini-flash-2.0",
        "gemini-2.5-pro-exp-03-25"
    ],
    "ollama": [
        # No restrictions for Ollama as they're locally hosted
    ]
}

def validate_model(provider: str, model: str) -> bool:
    """Validate that the model is available for the given provider."""
    if provider not in AVAILABLE_MODELS:
        return False

    # For Ollama, we don't restrict models
    if provider == "ollama":
        return True

    return model in AVAILABLE_MODELS[provider]

def list_available_models():
    """Print available models for each provider."""
    print("Available models by provider:")
    for provider, models in AVAILABLE_MODELS.items():
        print(f"\n{provider.upper()}:")
        if models:
            for model in models:
                print(f"  - {model}")
        else:
            print("  - Any model (no restrictions)")

def parse_args():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(description="Pydantic AI MCP Chat")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini", "ollama"],
                      help="The LLM provider to use")
    parser.add_argument("--model", type=str, help="The specific model to use")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    return parser.parse_args()

def get_provider_config() -> ProviderConfig:
    """Get provider config from environment variables."""
    provider_name = os.getenv('PROVIDER', 'openai').lower()

    if provider_name == "openai":
        return ProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'gpt-4o'),
            base_url=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
        )
    elif provider_name == "anthropic":
        return ProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'claude-3-7-sonnet-20250219'),
            base_url=os.getenv('BASE_URL', 'https://api.anthropic.com/v1'),
            additional_params={
                "thinking": {"type": "enabled" if os.getenv('ENABLE_THINKING') else "disabled"}
            }
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            provider=LLMProvider.GEMINI,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'gemini-2.5-pro-exp-03-25'),
            base_url=os.getenv('BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai'),
        )
    elif provider_name == "ollama":
        return ProviderConfig(
            provider=LLMProvider.OLLAMA,
            api_key="ollama",  # Not used but required
            model=os.getenv('MODEL_CHOICE', 'llama3'),
            base_url=os.getenv('BASE_URL', 'http://localhost:11434/v1'),
        )
    else:
        print(f"Unknown provider: {provider_name}. Defaulting to OpenAI.")
        return ProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'gpt-4o'),
            base_url=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
        )

def get_model():
    """Create an OpenAI model with the chosen provider configuration."""
    config = get_provider_config()

    # Set environment variables for authentication
    os.environ['OPENAI_API_KEY'] = config.api_key

    provider_string = 'openai'
    if config.provider == LLMProvider.ANTHROPIC:
        provider_string = 'anthropic'
    elif config.provider == LLMProvider.GEMINI:
        provider_string = 'gemini'
    elif config.provider == LLMProvider.OLLAMA:
        provider_string = 'ollama'

    # Create a provider with the specified base URL
    provider = infer_provider(provider_string)

    # Set base_url on the provider directly
    if provider_string == 'openai' and config.get_base_url() != 'https://api.openai.com/v1':
        provider.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.get_base_url()
        )
    elif hasattr(provider, 'client'):
        # Make sure provider has client initialized with correct API key
        provider.client.api_key = config.api_key

    # Create the model with exactly the signature from the docs
    return OpenAIModel(
        model_name=config.model,
        provider=provider
    )

async def get_pydantic_ai_agent():
    """Initialize and return the MCP client and Pydantic AI agent."""
    client = mcp_client.MCPClient()
    client.load_servers(str(CONFIG_FILE))
    tools = await client.start()
    return client, Agent(model=get_model(), tools=tools)

async def reload_agent(client):
    """Recreate the agent with possibly new model settings."""
    # Get a new agent with updated model settings
    new_agent = Agent(model=get_model(), tools=client.tools)
    return new_agent

def handle_command(command: str, mcp_client_instance):
    """Handle special commands entered during chat."""
    if command.startswith("!MODEL="):
        new_model = command[7:].strip()
        provider = os.getenv('PROVIDER', 'openai')

        if validate_model(provider, new_model):
            os.environ['MODEL_CHOICE'] = new_model
            print(f"[System] Switched to model: {new_model}")
            # Note: The agent will be reloaded in the main loop
            return {"action": "reload_agent", "success": True}
        else:
            print(f"[System] Error: Model '{new_model}' is not available for provider '{provider}'.")
            return {"action": "none", "success": False}

    elif command.startswith("!PROVIDER="):
        new_provider = command[10:].strip().lower()
        if new_provider in ["openai", "anthropic", "gemini", "ollama"]:
            os.environ['PROVIDER'] = new_provider
            print(f"[System] Switched to provider: {new_provider}")
            # Reset to default model for provider
            if new_provider == "openai":
                os.environ['MODEL_CHOICE'] = "gpt-4o"
            elif new_provider == "anthropic":
                os.environ['MODEL_CHOICE'] = "claude-3-7-sonnet-20250219"
            elif new_provider == "gemini":
                os.environ['MODEL_CHOICE'] = "gemini-2.5-pro-exp-03-25"
            elif new_provider == "ollama":
                os.environ['MODEL_CHOICE'] = "llama3"
            return {"action": "reload_agent", "success": True}
        else:
            print(f"[System] Error: Provider '{new_provider}' is not supported.")
            return {"action": "none", "success": False}

    # List available models
    elif command == "!MODELS":
        list_available_models()
        return {"action": "none", "success": True}

    # Show help
    elif command == "!HELP":
        print("\nAvailable commands:")
        print("  !MODEL=<model>      - Switch to a different model")
        print("  !PROVIDER=<provider> - Switch to a different provider")
        print("  !MODELS             - List available models")
        print("  !HELP               - Show this help message")
        return {"action": "none", "success": True}

    return {"action": "none", "success": False}  # Command not recognized

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ Main Function with CLI Chat ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    # Parse command line arguments
    args = parse_args()

    # Handle --list-models flag
    if args.list_models:
        list_available_models()
        return

    # Set environment variables from command line arguments if provided
    if args.provider:
        os.environ['PROVIDER'] = args.provider
    if args.model:
        provider = os.getenv('PROVIDER', 'openai')
        if validate_model(provider, args.model):
            os.environ['MODEL_CHOICE'] = args.model
        else:
            print(f"Error: Model '{args.model}' is not available for provider '{provider}'.")
            list_available_models()
            return

    print("=== Pydantic AI MCP CLI Chat ===")
    print("Type 'exit' to quit the chat")
    print("Special commands: !MODEL=<model>, !PROVIDER=<provider>, !MODELS, !HELP")

    provider = os.getenv('PROVIDER', 'openai')
    model = os.getenv('MODEL_CHOICE', 'default')
    print(f"Using provider: {provider}")
    print(f"Using model: {model}")

    mcp_client_instance = None
    try:
        # Initialize the agent and message history
        mcp_client_instance, mcp_agent = await get_pydantic_ai_agent()
        console = Console()
        messages = []

        while True:
            # Get user input
            user_input = input("\n[You] ")

            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break

            # Check if this is a special command
            if user_input.startswith("!"):
                result = handle_command(user_input, mcp_client_instance)

                # If we need to reload the agent
                if result["action"] == "reload_agent" and result["success"]:
                    try:
                        print("[System] Reloading agent with new settings...")
                        mcp_agent = await reload_agent(mcp_client_instance)
                        provider = os.getenv('PROVIDER', 'openai')
                        model = os.getenv('MODEL_CHOICE', 'default')
                        print(f"[System] Now using: {provider} / {model}")
                    except Exception as e:
                        print(f"[System] Error reloading agent: {str(e)}")

                continue

            try:
                # Process the user input and output the response
                print("\n[Assistant]")

                # Modified streaming approach to fix the duplication issue
                full_response = ""
                console.print("", end="", highlight=False)

                async with mcp_agent.run_stream(
                    user_input, message_history=messages
                ) as result:
                    # Initialize these variables for the stream
                    last_displayed_length = 0

                    async for message in result.stream_text(delta=True):
                        # Add new content to the full response
                        full_response += message

                        # Print just the new content (delta)
                        # This avoids re-rendering the entire message each time
                        print(message, end="", flush=True)

                # Print a newline at the end for better readability
                print()

                # Add the new messages to the chat history
                messages.extend(result.all_messages())

            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
                if "out of memory" in str(e).lower():
                    print("This might be due to the message history growing too large. Try restarting the chat.")
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper cleanup of MCP client resources when exiting
        if mcp_client_instance:
            try:
                await mcp_client_instance.cleanup()
                print("MCP client resources cleaned up successfully.")
            except Exception as e:
                print(f"Warning during cleanup: {e}")
                # Cleanup errors are expected and non-critical

if __name__ == "__main__":
    asyncio.run(main())
