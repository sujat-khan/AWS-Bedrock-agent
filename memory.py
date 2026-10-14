"""
Amazon Bedrock AgentCore Memory Integration.

What this module does:
1. Connects to AWS Bedrock AgentCore Memory (data plane) using Boto3.
2. Reads past conversational history from AWS cloud for a specific session (`sessionId`).
3. Translates AWS AgentCore events into LangChain message objects (HumanMessage, AIMessage).
4. Saves new conversation turns to AWS AgentCore Memory so the agent remembers context across questions.

Key Concepts:
- memoryId: The unique identifier of your AWS Memory resource (e.g., 'memory_aslbw-86x9vs8AFD').
- sessionId: Groups conversation turns together (e.g., 'session-sujat-01').
- actorId: The identifier of the user/caller (e.g., 'sujat').
- event: An immutable record stored in AWS containing conversational turns (USER and ASSISTANT).
"""

import os
from datetime import datetime, timezone
from typing import List
import boto3
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Load environment configuration (.env)
load_dotenv()


def get_agentcore_memory_client(region_name: str = None):
    """
    Initialize the Boto3 client for Amazon Bedrock AgentCore data plane.
    
    The 'bedrock-agentcore' service client is used for:
    - create_event: Writing conversational turns to memory.
    - list_events: Reading conversation turns from memory.
    - retrieve_memory_records: Semantic search in long-term memory.
    """
    resolved_region = region_name or os.getenv("AWS_REGION", "ap-south-1")
    return boto3.client("bedrock-agentcore", region_name=resolved_region)


def load_session_messages(
    session_id: str,
    actor_id: str = "sujat",
    memory_id: str = None,
    region_name: str = None
) -> List[BaseMessage]:
    """
    Retrieve past conversation turns from AWS Bedrock AgentCore Memory and convert
    them into LangChain message objects (HumanMessage, AIMessage).

    Args:
        session_id: The session identifier (e.g. 'session-sujat-01')
        actor_id: Identifier for the user (default: 'sujat')
        memory_id: AWS Memory ID (defaults to AGENTCORE_MEMORY_ID from .env)
        region_name: AWS region name (defaults to AWS_REGION from .env)

    Returns:
        List of LangChain messages ready to be fed into LangGraph state.
    """
    resolved_memory_id = memory_id or os.getenv("AGENTCORE_MEMORY_ID")
    if not resolved_memory_id:
        print("[Memory Notice] AGENTCORE_MEMORY_ID not set in .env. Memory retrieval skipped.")
        return []

    client = get_agentcore_memory_client(region_name)

    try:
        # Call AWS Bedrock AgentCore Data Plane API
        response = client.list_events(
            memoryId=resolved_memory_id,
            sessionId=session_id,
            actorId=actor_id,
        )

        events = response.get("events", [])
        messages: List[BaseMessage] = []

        # Sort events by timestamp so conversation is chronological (oldest to newest)
        sorted_events = sorted(events, key=lambda e: e.get("eventTimestamp", 0))

        for event in sorted_events:
            for item in event.get("payload", []):
                conv = item.get("conversational")
                if not conv:
                    continue

                role = conv.get("role")
                text = conv.get("content", {}).get("text", "")

                if not text:
                    continue

                # Map AWS roles to LangChain message classes
                if role == "USER":
                    messages.append(HumanMessage(content=text))
                elif role == "ASSISTANT":
                    messages.append(AIMessage(content=text))

        return messages

    except Exception as e:
        print(f"[Memory Warning] Could not retrieve session history from AgentCore: {e}")
        return []


def save_conversation_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    actor_id: str = "sujat",
    memory_id: str = None,
    region_name: str = None
) -> str:
    """
    Record a completed user question and assistant answer to AWS Bedrock AgentCore Memory.

    Args:
        session_id: The active session identifier.
        user_text: The prompt submitted by the user.
        assistant_text: The final answer produced by the agent.
        actor_id: Identifier for the user.
        memory_id: AWS Memory ID.
        region_name: AWS region name.

    Returns:
        The created AWS Event ID, or empty string on failure.
    """
    resolved_memory_id = memory_id or os.getenv("AGENTCORE_MEMORY_ID")
    if not resolved_memory_id:
        return ""

    client = get_agentcore_memory_client(region_name)

    try:
        now_utc = datetime.now(timezone.utc)

        # Structure payload matching AWS Bedrock AgentCore specification
        payload = [
            {
                "conversational": {
                    "role": "USER",
                    "content": {"text": user_text}
                }
            },
            {
                "conversational": {
                    "role": "ASSISTANT",
                    "content": {"text": assistant_text}
                }
            }
        ]

        response = client.create_event(
            memoryId=resolved_memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=now_utc,
            payload=payload
        )

        event_id = response.get("event", {}).get("eventId", "")
        return event_id

    except Exception as e:
        print(f"[Memory Warning] Failed to save conversation turn to AgentCore: {e}")
        return ""
