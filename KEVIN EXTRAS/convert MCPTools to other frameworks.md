###Langchain

from langchain.tools import BaseTool

def create_langchain_tool(server, tool: MCPTool) -> BaseTool:
    """Convert an MCP tool to a LangChain tool."""

    class MCPLangChainTool(BaseTool):
        name = tool.name
        description = tool.description or ""

        async def _arun(self, **kwargs):
            return await server.session.call_tool(tool.name, arguments=kwargs)

        def _run(self, **kwargs):
            raise NotImplementedError("This tool only supports async execution")

    return MCPLangChainTool()



