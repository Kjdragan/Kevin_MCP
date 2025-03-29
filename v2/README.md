# Enhanced Pydantic AI MCP Agent with Multi-Provider Support

This project provides a unified interface for using MCP (Model Context Protocol) tools with multiple LLM providers through their OpenAI compatibility layers. It allows you to seamlessly switch between different AI providers while maintaining the same code structure.

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google Gemini, and Ollama
- **Unified Tool Interface**: Use the same tool code regardless of provider
- **OpenAI Compatibility**: Uses OpenAI compatibility layers across all providers
- **Rich MCP Server Support**: Works with a variety of MCP tool servers
- **Interactive Demo**: Test different providers with the same query

## Supported LLM Providers

- **OpenAI**: GPT-4o, GPT-4o-mini, and other OpenAI models
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, and other Claude models
- **Google Gemini**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Ollama**: Run local open-source models like Llama, Mistral, and more

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

### Interactive Demo

The easiest way to use this project is with the interactive demo:

```bash
python
