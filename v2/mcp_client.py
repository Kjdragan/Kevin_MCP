from pydantic_ai import RunContext, Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
import logging
import shutil
import json
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_client")

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: LLMProvider
    api_key: str
    model: str
    base_url: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    def get_base_url(self) -> str:
        """Get the base URL for the provider, using defaults if not specified."""
        if self.base_url:
            return self.base_url

        # Default base URLs for each provider
        if self.provider == LLMProvider.ANTHROPIC:
            return "https://api.anthropic.com/v1"
        elif self.provider == LLMProvider.GEMINI:
            return "https://generativelanguage.googleapis.com/v1beta/openai"
        elif self.provider == LLMProvider.OLLAMA:
            return "http://localhost:11434/v1"
        else:  # OpenAI or default
            return "https://api.openai.com/v1"

    @property
    def max_context_length(self) -> int:
        """Get the approximate max context length for the model."""
        # This is a simplification - actual limits vary by model
        if self.provider == LLMProvider.OPENAI:
            return 128000  # GPT-4o approx
        elif self.provider == LLMProvider.ANTHROPIC:
            return 200000  # Claude 3.5 Sonnet approx
        elif self.provider == LLMProvider.GEMINI:
            return 1000000  # Gemini 1.5 Flash approx
        elif self.provider == LLMProvider.OLLAMA:
            return 32768  # Depends heavily on model and configuration
        return 8192  # Conservative default

class LLMResponse(BaseModel):
    """Standardized response from LLM providers."""
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

class MCPClient:
    """Manages connections to one or more MCP servers based on mcp_config.json"""

    def __init__(self) -> None:
        self.servers: List[MCPServer] = []
        self.config: Dict[str, Any] = {}
        self.tools: List[Any] = []
        self.openai_tools: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()

    def load_servers(self, config_path: str) -> None:
        """Load server configuration from a JSON file (typically mcp_config.json)
        and creates an instance of each server (no active connection until 'start' though).

        Args:
            config_path: Path to the JSON configuration file.
        """
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        self.servers = [
            MCPServer(name, config)
            for name, config in self.config["mcpServers"].items()
        ]

        logger.info(f"Loaded {len(self.servers)} MCP servers from configuration")

    async def start(self) -> List[PydanticTool]:
        """Starts each MCP server and returns the tools for each server formatted for Pydantic AI."""
        self.tools = []
        self.openai_tools = []

        for server in self.servers:
            try:
                await server.initialize()
                pydantic_tools, openai_tools = await server.create_tools()
                self.tools.extend(pydantic_tools)
                self.openai_tools.extend(openai_tools)
                logger.info(f"Initialized server '{server.name}' with {len(openai_tools)} tools")
            except Exception as e:
                logger.error(f"Failed to initialize server '{server.name}': {e}")
                await self.cleanup_servers()
                return []

        return self.tools

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        # Sequential cleanup to avoid race conditions and task group errors
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logger.warning(f"Warning during cleanup of server {server.name}: {e}")

        # Clear the servers list to prevent double cleanup attempts
        self.servers = []

    async def cleanup(self) -> None:
        """Clean up all resources including the exit stack."""
        try:
            # First clean up all servers
            await self.cleanup_servers()
            # Then close the exit stack
            try:
                await self.exit_stack.aclose()
            except RuntimeError as e:
                if "Attempted to exit cancel scope in a different task" in str(e):
                    logger.warning("Cancel scope exit error during cleanup (expected during program exit)")
                else:
                    logger.warning(f"Runtime error during exit stack cleanup: {e}")
            except Exception as e:
                logger.warning(f"Warning during exit stack cleanup: {e}")
        except Exception as e:
            logger.warning(f"Warning during final cleanup: {e}")

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Returns tools in OpenAI format for direct use with OpenAI compatible SDKs."""
        return self.openai_tools


class MCPServer:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=self.config["env"]
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logger.debug(f"Server '{self.name}' initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def create_tools(self) -> Tuple[List[PydanticTool], List[Dict[str, Any]]]:
        """Create tools in both Pydantic AI and OpenAI formats."""
        if not self.session:
            raise ValueError(f"Server '{self.name}' not initialized")

        tools_result = await self.session.list_tools()
        tools = tools_result.tools

        pydantic_tools = []
        openai_tools = []

        for tool in tools:
            # Create Pydantic AI tool
            pydantic_tool = self.create_pydantic_tool(tool)
            pydantic_tools.append(pydantic_tool)

            # Create OpenAI compatible tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            }
            openai_tools.append(openai_tool)

        return pydantic_tools, openai_tools

    def create_pydantic_tool(self, tool: MCPTool) -> PydanticTool:
        """Initialize a Pydantic AI Tool from an MCP Tool."""
        async def execute_tool(**kwargs: Any) -> Any:
            return await self.session.call_tool(tool.name, arguments=kwargs)

        async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            tool_def.parameters_json_schema = tool.inputSchema
            return tool_def

        return PydanticTool(
            execute_tool,
            name=tool.name,
            description=tool.description or "",
            takes_ctx=False,
            prepare=prepare_tool
        )

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments."""
        if not self.session:
            return None

        tools_result = await self.session.list_tools()
        tools = tools_result.tools
        for tool in tools:
            if tool.name == tool_name:
                logger.info(f"Executing tool '{tool_name}' on server '{self.name}'")
                try:
                    result = await self.session.call_tool(tool_name, arguments=arguments)
                    return result
                except Exception as e:
                    logger.error(f"Error executing tool '{tool_name}': {e}")
                    return {"error": str(e)}

        return None

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                try:
                    await self.exit_stack.aclose()
                except RuntimeError as e:
                    if "Attempted to exit cancel scope in a different task" in str(e):
                        logger.debug(f"Cancel scope exit error during cleanup of server {self.name} (expected during program exit)")
                    else:
                        logger.warning(f"Runtime error during cleanup of server {self.name}: {e}")
                except Exception as e:
                    logger.warning(f"Error during exit stack cleanup of server {self.name}: {e}")

                self.session = None
                self.stdio_context = None
                logger.debug(f"Server '{self.name}' cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {e}")
