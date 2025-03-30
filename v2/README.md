# Enhanced Pydantic AI MCP Agent with Multi-Provider Support

This project provides a unified interface for using MCP (Model Context Protocol) tools with multiple LLM providers through their OpenAI compatibility layers. It allows you to seamlessly switch between different AI providers while maintaining the same code structure.

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google Gemini, and Ollama
- **Unified Tool Interface**: Use the same tool code regardless of provider
- **OpenAI Compatibility**: Uses OpenAI compatibility layers across all providers
- **Rich MCP Server Support**: Works with a variety of MCP tool servers
- **Interactive Demo**: Test different providers with the same query
- **Runtime Model Switching**: Change models and providers without restarting
- **Command Line Parameters**: Configure provider and model via CLI arguments
- **Restricted Model Selection**: Limited to verified compatible models

## Supported LLM Providers and Models

| Provider  | Model Options                  | Default Model                | Description                             |
|-----------|-------------------------------|-----------------------------|-----------------------------------------|
| OpenAI    | gpt-4o, gpt-4o-mini          | gpt-4o                      | OpenAI's latest models with tool calling |
| Anthropic | claude-3-7-sonnet-20250219, <br>claude-3-haiku-20240307 | claude-3-7-sonnet-20250219  | Claude's most advanced models           |
| Gemini    | gemini-flash-2.0, <br>gemini-2.5-pro-exp-03-25 | gemini-2.5-pro-exp-03-25    | Google's Gemini models                 |
| Ollama    | *any locally hosted models*   | llama3                      | Self-hosted open-source models          |

## MCP Servers Included

This implementation includes configuration for these MCP servers:

- **Tavily**: Web search and information retrieval
- **Fetch**: Fetch and process web content
- **LangGraph Docs**: Documentation for LangGraph
- **Filesystem**: File system access and management
- **MCP Installer**: Install packages and tools
- **ArXiv**: Access to academic papers on ArXiv
- **Perplexity**: Use Perplexity's AI for complex queries
- **YouTube**: Work with YouTube content

## Prerequisites

- Python 3.9+
- Node.js (for MCP servers)
- Various API keys depending on which providers you want to use

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pydantic-ai-mcp-agent.git
   cd pydantic-ai-mcp-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and preferences

## Configuration

### Provider Configuration

You can set your preferred provider in the `.env` file:

```env
PROVIDER=openai
LLM_API_KEY=your-api-key-here
MODEL_CHOICE=gpt-4o-mini
```

### MCP Server Configuration

MCP servers are configured in `mcp_config.json`. You can add, remove, or modify server configurations as needed.

## Usage

### Command Line Parameters

The application supports several command line parameters:

```bash
# Start with a specific provider
python pydantic_mcp_agent.py --provider anthropic

# Start with a specific model
python pydantic_mcp_agent.py --model claude-3-7-sonnet-20250219

# Use both provider and model parameters
python pydantic_mcp_agent.py --provider gemini --model gemini-2.5-pro-exp-03-25

# List all available models
python pydantic_mcp_agent.py --list-models
```

### Runtime Commands

During chat sessions, you can use special commands:

- `!MODEL=<model>` - Switch to a different model without restarting
  ```
  !MODEL=gpt-4o
  ```

- `!PROVIDER=<provider>` - Switch to a different provider
  ```
  !PROVIDER=anthropic
  ```

- `!MODELS` - List all available models for all providers

- `!HELP` - Show available commands

- `exit`, `quit`, `bye`, `goodbye` - Exit the application

### Interactive Demo

The easiest way to use this project is with the interactive interface:

```bash
python main.py
```

This provides an interactive menu for selecting providers, models, and testing different queries.

### Direct API Usage

You can also integrate the agent into your own projects:

```python
import asyncio
from pydantic_mcp_agent import get_pydantic_ai_agent

async def run_query():
    client, agent = await get_pydantic_ai_agent()
    try:
        result = await agent.run("Search for the latest AI research papers")
        print(result.data)
    finally:
        await client.cleanup()

asyncio.run(run_query())
```

## Lessons Learned

During development, we encountered and addressed several important issues:

### 1. Text Streaming Challenges

**Problem**: The initial implementation caused text duplication and incomplete formatting during streaming responses. Each update would re-render the entire content with incremental additions, creating a stuttering and repetitive experience.

**Solution**: We replaced Rich's Live context with direct printing of delta content using Python's built-in `print()` function with `end=""` and `flush=True`. This approach only renders the new content rather than re-rendering the full response for every update.

**Lessons**:
- Markdown rendering libraries may struggle with incremental updates
- For streaming content, simpler approaches often work better
- Test streaming behavior thoroughly with long-form responses

### 2. Model Compatibility Issues

**Problem**: Not all models from each provider worked consistently with the tool-calling protocol, causing unexpected errors or silent failures.

**Solution**: We implemented a whitelist approach, restricting model selection to verified compatible models for each provider. This ensures users can only select models known to work with our MCP integration.

**Lessons**:
- Different LLM providers implement OpenAI compatibility layers differently
- Tool-calling capabilities vary significantly between models
- A whitelist approach improves reliability for end users

### 3. Async Resource Management

**Problem**: Improper cleanup of MCP client resources could lead to dangling processes and resource leaks.

**Solution**: We implemented robust async cleanup processes using AsyncExitStack and proper exception handling, ensuring resources are released even if errors occur.

**Lessons**:
- Always implement proper cleanup for async resources
- Use AsyncExitStack for managing multiple async context managers
- Handle cleanup exceptions gracefully to prevent cascading failures

### 4. Provider-Specific Configuration

**Problem**: Different providers required specific configuration settings that weren't obvious from their APIs.

**Solution**: We created a centralized ProviderConfig class that encapsulates all provider-specific details, making it easier to manage configurations across the application.

**Lessons**:
- Abstract provider-specific details behind a unified interface
- Use enumerated types for provider selection to prevent errors
- Provide sensible defaults that work across different environments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
