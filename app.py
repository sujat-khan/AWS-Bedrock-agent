"""
Amazon Bedrock AgentCore Application Entrypoint.

This file wraps the LangGraph agent inside AWS Bedrock AgentCore's runtime
using `BedrockAgentCoreApp`. It enables the agent to be deployed as a managed
serverless agent on AWS Bedrock AgentCore or tested locally.

Memory Integration:
- Uses Amazon Bedrock AgentCore Memory (data plane) to load and persist conversation turns.
- Sessions are tracked via `sessionId` and `actorId`.
"""

import sys
import os
from dotenv import load_dotenv
from bedrock_agentcore import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage

from graph import create_agent_graph
from memory import load_session_messages, save_conversation_turn

# Load configuration (.env)
load_dotenv()

# Initialize the Bedrock AgentCore application wrapper
app = BedrockAgentCoreApp()

# Compile the LangGraph agent graph
print("Compiling LangGraph agent graph...")
agent_graph = create_agent_graph()
print("LangGraph agent compiled and ready!")


def _extract_text(content) -> str:
    """Helper to extract clean text from string or message blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()
    return str(content)


@app.entrypoint
def invoke(request: dict) -> dict:
    """
    Main entrypoint called by the Amazon Bedrock AgentCore Runtime.

    Args:
        request: Invocation payload. Supports:
                 - {"prompt": "user question", "sessionId": "...", "actorId": "..."}
                 - {"query": "user question"}
                 - or raw string

    Returns:
        Structured JSON response containing the agent answer and execution details.
    """
    # 1. Parse user input and session parameters
    session_id = "default_session"
    actor_id = "default_user"

    if isinstance(request, dict):
        user_input = request.get("prompt") or request.get("query") or request.get("input") or ""
        session_id = request.get("sessionId") or request.get("session_id") or session_id
        actor_id = request.get("actorId") or request.get("actor_id") or actor_id
    elif isinstance(request, str):
        user_input = request
    else:
        user_input = str(request)

    if not user_input.strip():
        return {
            "status": "error",
            "message": "Missing 'prompt' or 'query' in invocation payload."
        }

    # 2. Retrieve past conversation history from AWS Bedrock AgentCore Memory
    history = load_session_messages(session_id=session_id, actor_id=actor_id)

    # 3. Invoke the LangGraph agent with complete context
    initial_state = {"messages": history + [HumanMessage(content=user_input)]}
    graph_result = agent_graph.invoke(initial_state)

    # 4. Extract final response and inspect tool activity
    messages = graph_result.get("messages", [])
    final_message = messages[-1] if messages else None
    response_text = _extract_text(final_message.content) if final_message else "No response generated."

    # Inspect tool calls made during the reasoning graph
    tools_executed = []
    for msg in messages[len(history):]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_executed.append({
                    "name": tc.get("name"),
                    "args": tc.get("args")
                })

    # 5. Persist this turn to AWS Bedrock AgentCore Memory
    event_id = save_conversation_turn(
        session_id=session_id,
        user_text=user_input,
        assistant_text=response_text,
        actor_id=actor_id
    )

    # 6. Return structured response for Bedrock AgentCore
    return {
        "status": "success",
        "sessionId": session_id,
        "actorId": actor_id,
        "eventId": event_id,
        "prompt": user_input,
        "response": response_text,
        "tools_executed": tools_executed,
        "prior_messages_loaded": len(history),
        "total_messages": len(messages),
        "bedrock_region": os.getenv("AWS_REGION", "ap-south-1"),
        "bedrock_model": os.getenv("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0"),
    }


if __name__ == "__main__":
    # If '--serve' argument is passed, launch the Bedrock AgentCore HTTP server (default port 8080)
    if "--serve" in sys.argv:
        port = int(os.getenv("PORT", "8080"))
        print(f"\nStarting Amazon Bedrock AgentCore server on port {port}...")
        print("Endpoints available:")
        print(f"  - POST http://localhost:{port}/invocations")
        print(f"  - GET  http://localhost:{port}/ping")
        app.run(host="0.0.0.0", port=port)
    else:
        # Default: Run a local simulation test
        print("\n==========================================")
        print("  Amazon Bedrock AgentCore Local Runner")
        print("==========================================")
        test_payload = {
            "prompt": "Hello! Introduce yourself, tell me what time it is, and what 85 * 14 is.",
            "sessionId": "demo-session-local",
            "actorId": "sujat"
        }
        print(f"\nSending test payload:\n{test_payload}")
        result = invoke(test_payload)
        
        print("\n=== AgentCore Output ===")
        print(f"Status: {result.get('status')}")
        print(f"Session ID: {result.get('sessionId')}")
        print(f"Event ID Saved: {result.get('eventId')}")
        print(f"Tools Executed: {result.get('tools_executed')}")
        print("\nResponse:")
        print(result.get("response"))
        print("\n[Tip] Run with '--serve' to launch the production Bedrock AgentCore HTTP server.")
