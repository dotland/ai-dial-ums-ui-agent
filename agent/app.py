import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from agent.clients.dial_client import DialClient
from agent.clients.http_mcp_client import HttpMCPClient
from agent.clients.stdio_mcp_client import StdioMCPClient
from agent.conversation_manager import ConversationManager
from agent.models.message import Message

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

conversation_manager: Optional[ConversationManager] = None
SERVICE_NOT_INITIALIZED_ERROR = "Service not initialized"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP clients, Redis, and ConversationManager on startup"""
    global conversation_manager

    logger.info("Application startup initiated")

    tools: list[dict] = []
    tool_name_client_map: dict[str, HttpMCPClient | StdioMCPClient] = {}
    redis_client: Optional[redis.Redis] = None

    try:
        ums_client = await HttpMCPClient.create("http://localhost:8005/mcp")
        ums_tools = await ums_client.get_tools()
        for tool in ums_tools:
            tools.append(tool)
            tool_name_client_map[tool["function"]["name"]] = ums_client

        fetch_client = await HttpMCPClient.create("https://remote.mcpservers.org/fetch/mcp")
        fetch_tools = await fetch_client.get_tools()
        for tool in fetch_tools:
            tools.append(tool)
            tool_name_client_map[tool["function"]["name"]] = fetch_client

        duckduckgo_client = await StdioMCPClient.create("mcp/duckduckgo:latest")
        duckduckgo_tools = await duckduckgo_client.get_tools()
        for tool in duckduckgo_tools:
            tools.append(tool)
            tool_name_client_map[tool["function"]["name"]] = duckduckgo_client

        api_key = os.getenv("DIAL_API_KEY")
        if not api_key:
            raise RuntimeError("DIAL_API_KEY environment variable is not set")

        model = os.getenv("DIAL_MODEL", "gpt-4o")
        dial_client = DialClient(
            api_key=api_key,
            endpoint="https://ai-proxy.lab.epam.com",
            model=model,
            tools=tools,
            tool_name_client_map=tool_name_client_map,
        )

        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
        await redis_client.ping()

        conversation_manager = ConversationManager(dial_client=dial_client, redis_client=redis_client)
        logger.info(
            "Application startup complete",
            extra={"tool_count": len(tools), "tool_names": list(tool_name_client_map.keys()), "model": model},
        )
        yield
    finally:
        if redis_client is not None:
            await redis_client.close()
        logger.info("Application shutdown complete")


app = FastAPI(
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: Message
    stream: bool = True


class ChatResponse(BaseModel):
    content: str
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None

# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "conversation_manager_initialized": conversation_manager is not None
    }


@app.post("/conversations", responses={503: {"description": SERVICE_NOT_INITIALIZED_ERROR}})
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation"""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_INITIALIZED_ERROR)
    return await conversation_manager.create_conversation(request.title)


@app.get("/conversations", responses={503: {"description": SERVICE_NOT_INITIALIZED_ERROR}})
async def list_conversations():
    """List all conversations sorted by last update time"""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_INITIALIZED_ERROR)
    conversations = await conversation_manager.list_conversations()
    return [ConversationSummary(**conv_dict) for conv_dict in conversations]


@app.get(
    "/conversations/{conversation_id}",
    responses={
        404: {"description": "Conversation not found"},
        503: {"description": SERVICE_NOT_INITIALIZED_ERROR},
    },
)
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_INITIALIZED_ERROR)
    conversation = await conversation_manager.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete(
    "/conversations/{conversation_id}",
    responses={
        404: {"description": "Conversation not found"},
        503: {"description": SERVICE_NOT_INITIALIZED_ERROR},
    },
)
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_INITIALIZED_ERROR)
    deleted = await conversation_manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}


@app.post(
    "/conversations/{conversation_id}/chat",
    responses={
        404: {"description": "Conversation not found"},
        503: {"description": SERVICE_NOT_INITIALIZED_ERROR},
    },
)
async def chat(conversation_id: str, request: ChatRequest):
    """
    Chat endpoint that processes messages and returns assistant response.
    Supports both streaming and non-streaming modes.
    Automatically saves conversation state.
    """
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_INITIALIZED_ERROR)

    try:
        result = await conversation_manager.chat(
            user_message=request.message,
            conversation_id=conversation_id,
            stream=request.stream,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.stream:
        return StreamingResponse(result, media_type="text/event-stream")
    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting UMS Agent server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8011,
        log_level="debug",
    )