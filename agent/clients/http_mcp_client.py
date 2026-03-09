import logging
from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


class HttpMCPClient:
    """Connect to an HTTP MCP server and execute its tools.

    This class wraps the MCP SDK session lifecycle and exposes a small,
    OpenAI/DIAL-friendly interface that the chat agent can use.
    """

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        logger.debug("HttpMCPClient instance created", extra={"server_url": mcp_server_url})

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'HttpMCPClient':
        """Create a connected HTTP MCP client instance."""
        client = cls(mcp_server_url)
        await client.connect()
        return client

    async def connect(self):
        """Open MCP streams, create a session, and initialize protocol state."""
        self._streams_context = streamable_http_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        init_result = await self.session.initialize()
        logger.info("Connected to HTTP MCP server", extra={"url": self.server_url, "init_result": str(init_result)})

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return tools in DIAL/OpenAI format.

        MCP list_tools response uses the Anthropic MCP shape. DIAL expects
        OpenAI-compatible tool definitions.
        """
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
            "Loaded tools from HTTP MCP server",
            extra={"url": self.server_url, "tool_count": len(tools), "tool_names": [t["function"]["name"] for t in tools]},
        )
        return tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Execute a tool and return text when possible."""
        if self.session is None:
            raise RuntimeError("MCP client is not connected to MCP server")

        logger.info(
            "Calling HTTP MCP tool",
            extra={"url": self.server_url, "tool_name": tool_name, "tool_args": tool_args},
        )
        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        content = result.content
        first_item = content[0] if content else None
        if isinstance(first_item, TextContent):
            return first_item.text
        return content
