from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from dotenv import load_dotenv
import asyncio
import pathlib
import sys
import os

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

def get_provider_config() -> ProviderConfig:
    """Get provider config from environment variables."""
    provider_name = os.getenv('PROVIDER', 'openai').lower()

    if provider_name == "openai":
        return ProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'gpt-4o-mini'),
            base_url=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
        )
    elif provider_name == "anthropic":
        return ProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'claude-3-sonnet-20240229'),
            base_url=os.getenv('BASE_URL', 'https://api.anthropic.com/v1'),
            additional_params={
                "thinking": {"type": "enabled" if os.getenv('ENABLE_THINKING') else "disabled"}
            }
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            provider=LLMProvider.GEMINI,
            api_key=os.getenv('LLM_API_KEY', 'no-api-key-provided'),
            model=os.getenv('MODEL_CHOICE', 'gemini-1.5-pro'),
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
            model=os.getenv('MODEL_CHOICE', 'gpt-4o-mini'),
            base_url=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
        )

def get_model():
    """Create an OpenAI model with the chosen provider configuration.

    Following the exact documented signature from pydantic-ai docs:

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai', 'deepseek', 'azure'] | Provider[AsyncOpenAI] = 'openai',
        system_prompt_role: OpenAISystemPromptRole | None = None,
    )
    """
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ Main Function with CLI Chat ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    print("=== Pydantic AI MCP CLI Chat ===")
    print("Type 'exit' to quit the chat")
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

            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    async with mcp_agent.run_stream(
                        user_input, message_history=messages
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))

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
