import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler for better error reporting
install_rich_traceback()

import mcp_client
from mcp_client import LLMProvider, ProviderConfig
from pydantic_mcp_agent import get_pydantic_ai_agent, get_provider_config

# Load environment variables
load_dotenv()

async def demonstrate_provider(provider_name: str, query: str = None):
    """Demonstrate an agent using the specified provider."""
    console = Console()
    client = None

    # Set the provider in environment for get_provider_config to use
    os.environ['PROVIDER'] = provider_name

    console.print(f"\n[bold blue]=== Running with {provider_name.upper()} provider ===")
    console.print(f"[blue]Model: {os.getenv('MODEL_CHOICE', 'default')}")

    try:
        # Initialize with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Initializing {task.description}..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"{provider_name} agent", total=None)

            try:
                # Get the agent
                client, agent = await get_pydantic_ai_agent()
                progress.update(task, description="MCP tools loaded")
            except TypeError as e:
                if "base_url" in str(e):
                    console.print("\n[bold red]Error: Your version of pydantic-ai doesn't support the base_url parameter.")
                    console.print("Please update pydantic-ai to the latest version or check the implementation of OpenAIModel.")
                    return
                else:
                    raise

        # Use the provided query or ask for one
        if not query:
            query = Prompt.ask("[bold yellow]Enter your query",
                               default="What tools are available to help me?")
        else:
            console.print(f"\n[bold yellow]Query: {query}")

        console.print("\n[bold green]Response:")

        # Set a timeout for the query
        timeout = float(os.getenv('QUERY_TIMEOUT', '120'))  # Default 2 minutes

        try:
            with Live('', console=console, vertical_overflow='visible') as live:
                start_time = time.time()
                async with agent.run_stream(query) as result:
                    curr_message = ""
                    async for message in result.stream_text(delta=True):
                        curr_message += message
                        live.update(Markdown(curr_message))

                        # Check for timeout
                        if time.time() - start_time > timeout:
                            console.print(f"\n[bold yellow]Query timed out after {timeout} seconds")
                            break

            elapsed = time.time() - start_time
            console.print(f"\n[dim]Completed in {elapsed:.2f} seconds[/dim]")

        except Exception as e:
            console.print(f"\n[bold red]Error during query execution: {str(e)}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user")
    except Exception as e:
        console.print(f"\n[bold red]Error with {provider_name} provider: {type(e).__name__}: {str(e)}")
    finally:
        # Ensure cleanup happens
        if client:
            try:
                await client.cleanup()
            except Exception as e:
                # Suppress cleanup errors
                pass

async def interactive_demo():
    """Interactive demo allowing testing of multiple providers."""
    console = Console()
    console.print(Panel("[bold cyan]Pydantic AI MCP Multi-Provider Demo",
                       subtitle="Interactive Mode"))

    providers = {
        "1": "openai",
        "2": "anthropic",
        "3": "gemini",
        "4": "ollama"
    }

    while True:
        console.print("\n[bold cyan]Available Providers:")
        for key, provider in providers.items():
            model = os.getenv(f"{provider.upper()}_MODEL", os.getenv('MODEL_CHOICE', 'default'))
            console.print(f"  [bold]{key}[/bold]: {provider.capitalize()} - {model}")
        console.print("  [bold]5[/bold]: Test All Providers")
        console.print("  [bold]6[/bold]: Exit")
        console.print("  [bold]7[/bold]: Set Environment Variable")

        choice = Prompt.ask("\n[bold]Select an option", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "6":
            console.print("[bold]Exiting demo[/bold]")
            break

        if choice == "7":
            # Set an environment variable
            var_name = Prompt.ask("[bold yellow]Environment variable name",
                                  default="MODEL_CHOICE")
            var_value = Prompt.ask("[bold yellow]Value")
            os.environ[var_name] = var_value
            console.print(f"[green]Set {var_name}={var_value}")
            continue

        if choice == "5":
            query = Prompt.ask("[bold yellow]Enter your query for all providers",
                              default="What tools are available to help me?")

            # Test all providers sequentially
            for provider_name in providers.values():
                try:
                    await demonstrate_provider(provider_name, query)
                except Exception as e:
                    console.print(f"\n[bold red]Error with {provider_name}: {type(e).__name__}: {str(e)}")

                # Brief pause between providers
                await asyncio.sleep(1)
        else:
            try:
                await demonstrate_provider(providers[choice])
            except KeyboardInterrupt:
                console.print("[yellow]Interrupted!")
            except Exception as e:
                console.print(f"\n[bold red]Critical error: {type(e).__name__}: {str(e)}")

async def main():
    """Main function to demonstrate the multi-provider MCP client."""
    console = Console()

    try:
        console.print(Panel("[bold cyan]Pydantic AI MCP Multi-Provider Demo",
                       subtitle="Multi-Provider Integration"))

        # Process command line arguments
        if len(sys.argv) > 1:
            provider = sys.argv[1].lower()
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

            if provider == "interactive":
                await interactive_demo()
            elif provider == "all":
                # Demo all providers
                providers = ["openai", "anthropic", "gemini", "ollama"]
                for provider_name in providers:
                    try:
                        await demonstrate_provider(provider_name, query)
                    except Exception as e:
                        console.print(f"\n[bold red]Error with {provider_name}: {str(e)}")
                    # Brief pause between providers
                    await asyncio.sleep(1)
            else:
                # Demo just the specified provider
                await demonstrate_provider(provider, query)
        else:
            # No arguments, go to interactive mode
            await interactive_demo()

        console.print("\n[bold cyan]=== Demo Complete ===")
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted by user")
    except Exception as e:
        console.print(f"\n[bold red]Critical error: {type(e).__name__}: {str(e)}")
        console.print_exception()

if __name__ == "__main__":
    asyncio.run(main())
