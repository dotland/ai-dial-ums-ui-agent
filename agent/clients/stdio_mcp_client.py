import logging
from typing import Optional, Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


class StdioMCPClient:
    """Connect to an MCP server exposed through a local stdio process."""

    def __init__(self, docker_image: str) -> None:
        self.docker_image = docker_image
        self.session: Optional[ClientSession] = None
        self._stdio_context = None
        self._session_context = None
        logger.debug("StdioMCPClient instance created", extra={"docker_image": docker_image})

    @classmethod
    async def create(cls, docker_image: str) -> 'StdioMCPClient':
        """Create a connected stdio MCP client instance."""
        client = cls(docker_image)
        await client.connect()
        return client

    async def connect(self):
        """Start Dockerized MCP server process and initialize session."""
        server_params = StdioServerParameters(
            command="docker",
            args=["run", "--rm", "-i", self.docker_image],
        )
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        init_result = await self.session.initialize()
        logger.info(
            "Connected to stdio MCP server",
            extra={"docker_image": self.docker_image, "init_result": str(init_result)},
        )

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return tools converted from MCP format to OpenAI-compatible format."""
        if self.session is None:
            raise RuntimeError("MCP client is not connected to MCP server")

        list_tools_result = await self.session.list_tools()
        tools: list[dict[str, Any]] = []
        for tool in list_tools_result.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )

        logger.info(
            "Loaded tools from stdio MCP server",
            extra={"docker_image": self.docker_image, "tool_count": len(tools), "tool_names": [t["function"]["name"] for t in tools]},
        )
        return tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Execute a stdio MCP tool and return text when available."""
        if self.session is None:
            raise RuntimeError("MCP client is not connected to MCP server")

        logger.info(
            "Calling stdio MCP tool",
            extra={"docker_image": self.docker_image, "tool_name": tool_name, "tool_args": tool_args},
        )
        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        content = result.content
        first_item = content[0] if content else None
        if isinstance(first_item, TextContent):
            return first_item.text
        return content
