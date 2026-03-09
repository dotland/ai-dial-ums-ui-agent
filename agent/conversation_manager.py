import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator

import redis.asyncio as redis

from agent.clients.dial_client import DialClient
from agent.models.message import Message, Role
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CONVERSATION_PREFIX = "conversation:"
CONVERSATION_LIST_KEY = "conversations:list"


class ConversationManager:
    """Manage conversation lifecycle, persistence, and model interaction.

    The manager keeps all state in Redis so the API can be stateless and
    multiple requests can continue the same conversation by ID.
    """

    def __init__(self, dial_client: DialClient, redis_client: redis.Redis):
        self.dial_client = dial_client
        self.redis = redis_client
        logger.info("ConversationManager initialized")

    async def create_conversation(self, title: str) -> dict:
        """Create a new conversation and persist it in Redis."""
        conversation_id = str(uuid.uuid4())
        now_iso = datetime.now(UTC).isoformat()
        conversation_title = title or f"Conversation {now_iso}"
        conversation = {
            "id": conversation_id,
            "title": conversation_title,
            "messages": [],
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        await self.redis.set(f"{CONVERSATION_PREFIX}{conversation_id}", json.dumps(conversation))
        await self.redis.zadd(CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})
        logger.info("Conversation created", extra={"conversation_id": conversation_id, "title": conversation_title})
        return conversation

    async def list_conversations(self) -> list[dict]:
        """List all conversations sorted by last update time"""
        conversation_ids = await self.redis.zrevrange(CONVERSATION_LIST_KEY, 0, -1)
        conversations: list[dict] = []

        for conversation_id in conversation_ids:
            conversation_json = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
            if conversation_json:
                conv = json.loads(conversation_json)
                conversations.append(
                    {
                        "id": conv["id"],
                        "title": conv["title"],
                        "created_at": conv["created_at"],
                        "updated_at": conv["updated_at"],
                        "message_count": len(conv["messages"]),
                    }
                )

        return conversations

    async def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a specific conversation"""
        conversation_json = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
        if not conversation_json:
            return None
        return json.loads(conversation_json)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        deleted_count = await self.redis.delete(f"{CONVERSATION_PREFIX}{conversation_id}")
        if deleted_count:
            await self.redis.zrem(CONVERSATION_LIST_KEY, conversation_id)
            return True
        return False

    async def chat(
            self,
            user_message: Message,
            conversation_id: str,
            stream: bool = False
    ):
        """
        Process chat messages and return AI response.
        Automatically saves conversation state.
        """
        logger.info(
            "Chat request received",
            extra={"conversation_id": conversation_id, "stream": stream, "role": str(user_message.role.value)},
        )
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        messages = [Message(**msg_data) for msg_data in conversation.get("messages", [])]
        if not messages:
            messages.append(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))
        messages.append(user_message)

        if stream:
            return self._stream_chat(conversation_id=conversation_id, messages=messages)
        return await self._non_stream_chat(conversation_id=conversation_id, messages=messages)


    async def _stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat with automatic saving"""
        yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
        async for chunk in self.dial_client.stream_response(messages):
            yield chunk
        await self._save_conversation_messages(conversation_id, messages)

    async def _non_stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> dict:
        """Handle non-streaming chat"""
        ai_message = await self.dial_client.response(messages)
        messages.append(ai_message)
        await self._save_conversation_messages(conversation_id, messages)
        return {
            "content": ai_message.content or "",
            "conversation_id": conversation_id,
        }

    async def _save_conversation_messages(
            self,
            conversation_id: str,
            messages: list[Message]
    ):
        """Save or update conversation messages"""
        conversation_json = await self.redis.get(f"{CONVERSATION_PREFIX}{conversation_id}")
        if conversation_json is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        conversation = json.loads(conversation_json)
        conversation["messages"] = [message.model_dump(mode="json") for message in messages]
        conversation["updated_at"] = datetime.now(UTC).isoformat()
        await self._save_conversation(conversation)

    async def _save_conversation(self, conversation: dict):
        """Internal method to persist conversation to Redis"""
        conversation_id = conversation["id"]
        await self.redis.set(f"{CONVERSATION_PREFIX}{conversation_id}", json.dumps(conversation))
        await self.redis.zadd(CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})
