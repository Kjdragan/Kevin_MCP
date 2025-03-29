# Converting MCP Tools to Pydantic AI Tools

Let me walk you through the detailed process of converting MCP (Model Context Protocol) tools to Pydantic AI tools. Understanding this conversion will help you adapt the pattern for other frameworks like the OpenAI Agents SDK.

## Core Conversion Process

The heart of the conversion happens in these two methods in the `MCPServer` class:

```python
async def create_pydantic_ai_tools(self) -> List[PydanticTool]:
    """Convert MCP tools to pydantic_ai Tools."""
    tools = (await self.session.list_tools()).tools
    return [self.create_tool_instance(tool) for tool in tools]

def create_tool_instance(self, tool: MCPTool) -> PydanticTool:
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
```

Let's break this down step by step to understand exactly what's happening:

## Step 1: Retrieving MCP Tools

```python
tools = (await self.session.list_tools()).tools
```

Here, the client calls the `list_tools()` method on the MCP session, which sends a request to the MCP server to list all available tools. The server responds with a list of tools, each containing:

- `name`: The name of the tool
- `description`: A description of what the tool does
- `inputSchema`: A JSON Schema that defines the parameters the tool accepts

This is a standard MCP protocol operation defined as part of the MCP specification.

## Step 2: Tool Conversion

For each MCP tool, the `create_tool_instance` method is called to convert it to a Pydantic AI tool:

### 2.1: Creating the Execution Function

```python
async def execute_tool(**kwargs: Any) -> Any:
    return await self.session.call_tool(tool.name, arguments=kwargs)
```

This inner function:
- Takes any keyword arguments (`**kwargs`) provided by the Pydantic AI agent
- Calls the MCP tool with those arguments using `session.call_tool()`
- Returns the result from the MCP tool

This is the critical adapter function that bridges between the Pydantic AI framework and the MCP tool. When the Pydantic AI agent calls the tool, it will actually invoke this function, which forwards the call to the MCP server.

### 2.2: Creating the Preparation Function

```python
async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
    tool_def.parameters_json_schema = tool.inputSchema
    return tool_def
```

This function:
- Takes a `RunContext` (provides execution context) and a `ToolDefinition` (metadata about the tool)
- Sets the `parameters_json_schema` of the tool definition to match the MCP tool's input schema
- Returns the updated tool definition

The purpose of this function is to configure the tool's parameter schema when the tool is first registered with the Pydantic AI agent. This ensures the agent knows what parameters the tool expects.

### 2.3: Creating the Pydantic AI Tool

```python
return PydanticTool(
    execute_tool,
    name=tool.name,
    description=tool.description or "",
    takes_ctx=False,
    prepare=prepare_tool
)
```

This creates a `PydanticTool` object with:
- The `execute_tool` function that will be called when the tool is invoked
- The name of the tool, carried over from the MCP tool
- The description of the tool, carried over from the MCP tool (or an empty string if None)
- `takes_ctx=False`, indicating that the tool function doesn't need the agent's context
- The `prepare_tool` function that configures the tool's parameter schema

## Detailed Examination of PydanticTool

To better understand the conversion, let's look at what a `PydanticTool` expects:

```python
class Tool:
    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool = False,
        prepare: Optional[Callable[[RunContext, ToolDefinition], Awaitable[ToolDefinition | None]]] = None,
        deprecation: Optional[ToolDeprecation] = None,
    ):
        # ...implementation...
```

- `func`: The function to call when the tool is invoked (our `execute_tool`)
- `name`: A name for the tool
- `description`: A description of what the tool does
- `takes_ctx`: Whether the function expects a context parameter
- `prepare`: A function that prepares the tool definition (our `prepare_tool`)
- `deprecation`: Optional deprecation information

Our conversion creates a `PydanticTool` that satisfies all these requirements, using information from the MCP tool.

## Behind the Scenes: ToolDefinition and JSON Schema

A critical part of the conversion is mapping the MCP tool's `inputSchema` to the Pydantic AI tool's parameter schema:

```python
tool_def.parameters_json_schema = tool.inputSchema
```

Both MCP and Pydantic AI use JSON Schema to define tool parameters. JSON Schema is a standard for describing the structure of JSON data, including:

- Types (string, number, object, array, etc.)
- Constraints (required, minimum, maximum, pattern, etc.)
- Descriptions and metadata

By directly assigning the MCP tool's input schema to the Pydantic AI tool's parameter schema, we ensure that the Pydantic AI agent has the same understanding of the tool's parameters as the MCP server.

## Example: Complete Conversion

Let's see an example of a complete conversion from an MCP tool to a Pydantic AI tool:

### MCP Tool (from server)

```python
{
    "name": "search_files",
    "description": "Search for files matching a pattern in a directory",
    "inputSchema": {
        "type": "object",
        "required": ["path", "pattern"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to search"
            },
            "pattern": {
                "type": "string",
                "description": "File pattern to match (e.g., '*.txt')"
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search recursively",
                "default": false
            }
        }
    }
}
```

### Converted Pydantic AI Tool

```python
PydanticTool(
    # Execution function
    async def execute_tool(path: str, pattern: str, recursive: bool = False) -> Any:
        return await session.call_tool("search_files", {
            "path": path,
            "pattern": pattern,
            "recursive": recursive
        }),

    # Tool metadata
    name="search_files",
    description="Search for files matching a pattern in a directory",
    takes_ctx=False,

    # Preparation function
    prepare=async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition:
        tool_def.parameters_json_schema = {
            "type": "object",
            "required": ["path", "pattern"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.txt')"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively",
                    "default": false
                }
            }
        }
        return tool_def
)
```

This example shows how the MCP tool's metadata (name, description, and input schema) becomes the Pydantic AI tool's metadata, while the execution function bridges between the Pydantic AI agent and the MCP server.

## Adapting the Pattern for OpenAI Agents SDK

Now that we understand the conversion process for Pydantic AI, let's adapt it for the OpenAI Agents SDK:

### OpenAI Agents Tool Definition

The OpenAI Agents SDK defines tools using a format like this:

```python
tool = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search for files matching a pattern in a directory",
        "parameters": {
            "type": "object",
            "required": ["path", "pattern"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.txt')"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively",
                    "default": False
                }
            }
        }
    }
}
```

### Converting MCP Tools to OpenAI Agents Tools

Here's how you might adapt the conversion for OpenAI Agents:

```python
def create_openai_agent_tool(server, tool: MCPTool) -> dict:
    """Convert an MCP tool to an OpenAI Agents tool."""

    # Create the tool definition
    openai_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
        }
    }

    # Create the execution function
    async def execute_tool(tool_name: str, tool_args: dict) -> Any:
        if tool_name == tool.name:
            return await server.session.call_tool(tool.name, arguments=tool_args)
        return None

    # Return both the tool definition and execution function
    return {
        "tool_definition": openai_tool,
        "execute": execute_tool
    }
```

### Using the Converted Tools with OpenAI Agents

```python
import json
from openai import OpenAI

async def run_agent_with_tools(client, tools, prompt):
    # Register the tools with OpenAI
    tool_definitions = [tool["tool_definition"] for tool in tools]

    # Create the assistant
    assistant = client.beta.assistants.create(
        name="MCP Tool Assistant",
        instructions="You are an assistant with access to external tools.",
        tools=tool_definitions,
        model="gpt-4"
    )

    # Create a thread
    thread = client.beta.threads.create()

    # Add user message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Process the run
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run.status == "completed":
            break

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Find the matching tool and execute it
                result = None
                for tool in tools:
                    if tool["tool_definition"]["function"]["name"] == function_name:
                        result = await tool["execute"](function_name, function_args)
                        break

                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result)
                })

            # Submit the tool outputs
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        await asyncio.sleep(1)

    # Get the messages
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    return messages.data
```

In this adaptation:

1. We convert MCP tools to OpenAI Agents tools with:
   - The same name and description
   - The same parameter schema (JSON Schema)
   - A new execution function that's called when the tool is invoked

2. We create an OpenAI Assistant with the tool definitions

3. We handle tool invocations by:
   - Checking which tool is being called
   - Calling the corresponding MCP tool
   - Submitting the result back to the OpenAI Assistant

## Key Differences Between Frameworks

While the core conversion pattern is similar across frameworks, there are some key differences:

### Pydantic AI vs. OpenAI Agents

**Pydantic AI:**
- Tools are Python objects (instances of `PydanticTool`)
- Tools are registered directly with the agent
- Tool execution is handled by the framework automatically

**OpenAI Agents:**
- Tools are defined as JSON objects
- Tool definitions are sent to the OpenAI API
- Tool execution must be handled by your code

### Common Patterns Across Frameworks

Despite the differences, the common patterns are:

1. **Metadata Mapping**: Map the MCP tool's metadata (name, description, parameter schema) to the target framework's format.

2. **Execution Bridging**: Create a function that bridges between the target framework and the MCP tool.

3. **Parameter Handling**: Ensure that the parameters expected by the target framework match those expected by the MCP tool.

## Advanced Considerations

When adapting the pattern for different frameworks, consider these advanced aspects:

### Error Handling

Both the MCP client and the target framework should handle errors appropriately:

```python
try:
    result = await session.call_tool(tool.name, arguments=kwargs)
    return result
except Exception as e:
    # Return a structured error format that the agent can understand
    return {"error": str(e)}
```

### Authentication and Security

If the MCP tools require authentication or have security considerations, these need to be managed in the bridging function:

```python
async def execute_tool(**kwargs: Any) -> Any:
    # Add authentication or security headers
    session.headers = {"Authorization": f"Bearer {get_token()}"}
    return await session.call_tool(tool.name, arguments=kwargs)
```

### Tool Lifecycle Management

Consider how tools are discovered, registered, and unregistered:

```python
async def update_tools(agent, client):
    # Get current tools
    current_tools = await client.start()

    # Update agent's tools
    agent.update_tools(current_tools)
```

### Tool Output Processing

The MCP tool's output format might need transformation for the target framework:

```python
async def execute_tool(**kwargs: Any) -> Any:
    result = await session.call_tool(tool.name, arguments=kwargs)

    # Transform result if needed
    if isinstance(result, bytes):
        return {"content_type": "binary", "data": result.hex()}
    return result
```

## Conclusion: Universal Adapter Pattern

The core pattern we've explored is a universal adapter pattern that can be applied to any agent framework:

1. **Get Tool Metadata**: Discover MCP tools and their metadata.

2. **Create Adapter Functions**: Create functions that bridge between the agent framework and MCP tools.

3. **Register Tools**: Register the adapted tools with the agent framework.

4. **Handle Execution**: Ensure tool calls are properly forwarded to MCP tools and results are returned to the agent.

This pattern allows you to integrate MCP tools with any agent framework, providing a consistent interface for AI agents to access external capabilities.

By understanding this conversion process in detail, you now have the knowledge to adapt MCP tools for any agent framework, whether it's Pydantic AI, OpenAI Agents, LangChain, AutoGen, or any other framework that supports tool usage.
