"""
Amazon Bedrock AgentCore Application Entrypoint.

This file wraps the LangGraph agent inside AWS Bedrock AgentCore's runtime
using `BedrockAgentCoreApp`. It enables the agent to be deployed as a managed
serverless agent on AWS Bedrock AgentCore or tested locally.
"""

import sys
import os
from dotenv import load_dotenv
from bedrock_agentcore import BedrockAgentCoreApp
from graph import create_agent_graph

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
                 - {"prompt": "user question"}
                 - {"query": "user question"}
                 - {"input": "user question"}
                 - or a string directly

    Returns:
        Structured JSON response containing the agent answer and execution details.
    """
    # 1. Parse user input from request
    if isinstance(request, dict):
        user_input = request.get("prompt") or request.get("query") or request.get("input") or ""
    elif isinstance(request, str):
        user_input = request
    else:
        user_input = str(request)

    if not user_input.strip():
        return {
            "status": "error",
            "message": "Missing 'prompt' or 'query' in invocation payload."
        }

    # 2. Invoke the LangGraph agent
    initial_state = {"messages": [("user", user_input)]}
    graph_result = agent_graph.invoke(initial_state)

    # 3. Extract final response and inspect tool activity
    messages = graph_result.get("messages", [])
    final_message = messages[-1] if messages else None
    response_text = _extract_text(final_message.content) if final_message else "No response generated."

    # Inspect tool calls made during the reasoning graph
    tools_executed = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_executed.append({
                    "name": tc.get("name"),
                    "args": tc.get("args")
                })

    # 4. Return structured response for Bedrock AgentCore
    return {
        "status": "success",
        "prompt": user_input,
        "response": response_text,
        "tools_executed": tools_executed,
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
            "prompt": "Hello! Introduce yourself, tell me what time it is, and what 85 * 14 is."
        }
        print(f"\nSending test payload:\n{test_payload}")
        result = invoke(test_payload)
        
        print("\n=== AgentCore Output ===")
        print(f"Status: {result.get('status')}")
        print(f"Tools Executed: {result.get('tools_executed')}")
        print("\nResponse:")
        print(result.get("response"))
        print("\n[Tip] Run with '--serve' to launch the production Bedrock AgentCore HTTP server.")
