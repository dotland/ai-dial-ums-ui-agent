import json
import logging
from collections import defaultdict
from typing import Any, AsyncGenerator

from openai import AsyncAzureOpenAI

from agent.clients.stdio_mcp_client import StdioMCPClient
from agent.models.message import Message, Role
from agent.clients.http_mcp_client import HttpMCPClient

logger = logging.getLogger(__name__)


class DialClient:
    """Handle model completions and delegate tool calls to MCP clients.

    The class supports both regular and streaming chat completion flows and
    recursively continues the conversation when the model requests tools.
    """

    def __init__(
            self,
            api_key: str,
            endpoint: str,
            model: str,
            tools: list[dict[str, Any]],
            tool_name_client_map: dict[str, HttpMCPClient | StdioMCPClient]
    ):
        """Initialize model client and tool routing.

        Args:
            api_key: DIAL API key.
            endpoint: DIAL-compatible Azure OpenAI endpoint.
            model: Model deployment name.
            tools: OpenAI-compatible tool list.
            tool_name_client_map: Maps tool names to concrete MCP client objects.
        """
        self.tools = tools
        self.tool_name_client_map = tool_name_client_map
        self.model = model
        self.async_openai = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="",
        )

    async def response(self, messages: list[Message]) -> Message:
        """Non-streaming completion with tool calling support"""
        response = await self.async_openai.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=self.tools,
            temperature=0.0,
            stream=False,
        )

        completion_message = response.choices[0].message
        ai_message = Message(
            role=Role.ASSISTANT,
            content=completion_message.content,
            tool_calls=[tool_call.model_dump() for tool_call in completion_message.tool_calls]
            if completion_message.tool_calls
            else None,
        )

        if ai_message.tool_calls:
            messages.append(ai_message)
            await self._call_tools(ai_message, messages)
            return await self.response(messages)

        return ai_message

    async def stream_response(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Streaming completion with tool calling support.
        Yields SSE-formatted chunks.
        """
        stream = await self.async_openai.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=self.tools,
            temperature=0.0,
            stream=True,
        )

        content_buffer = ""
        tool_deltas = []

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                chunk_data = {
                    "choices": [
                        {
                            "delta": {"content": delta.content},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                content_buffer += delta.content

            if delta.tool_calls:
                tool_deltas.extend(delta.tool_calls)

        if tool_deltas:
            tool_calls = self._collect_tool_calls(tool_deltas)
            ai_message = Message(
                role=Role.ASSISTANT,
                content=content_buffer or None,
                tool_calls=tool_calls,
            )
            messages.append(ai_message)
            await self._call_tools(ai_message, messages)

            async for chunk in self.stream_response(messages):
                yield chunk
            return

        messages.append(Message(role=Role.ASSISTANT, content=content_buffer))
        final_chunk = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def _collect_tool_calls(self, tool_deltas):
        """Convert streaming tool call deltas to complete tool calls"""
        tool_dict = defaultdict(lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None})

        for delta in tool_deltas:
            idx = delta.index
            if delta.id: tool_dict[idx]["id"] = delta.id
            if delta.function.name: tool_dict[idx]["function"]["name"] = delta.function.name
            if delta.function.arguments: tool_dict[idx]["function"]["arguments"] += delta.function.arguments
            if delta.type: tool_dict[idx]["type"] = delta.type

        collected_tools = list(tool_dict.values())
        logger.debug(
            "Collected tool calls from deltas",
            extra={"tool_count": len(collected_tools)}
        )
        return collected_tools

    async def _call_tools(self, ai_message: Message, messages: list[Message], silent: bool = False):
        """Execute tool calls using MCP client"""
        for tool_call in ai_message.tool_calls or []:
            function_block = tool_call.get("function", {})
            tool_name = function_block.get("name")
            raw_args = function_block.get("arguments") or "{}"

            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError:
                tool_args = {}

            mcp_client = self.tool_name_client_map.get(tool_name)
            if mcp_client is None:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=tool_call.get("id"),
                        name=tool_name,
                        content=f"Tool '{tool_name}' is not available.",
                    )
                )
                continue

            if not silent:
                logger.info("Executing tool", extra={"tool_name": tool_name, "tool_args": tool_args})
            tool_result = await mcp_client.call_tool(tool_name, tool_args)
            messages.append(
                Message(
                    role=Role.TOOL,
                    tool_call_id=tool_call.get("id"),
                    name=tool_name,
                    content=tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                )
            )
