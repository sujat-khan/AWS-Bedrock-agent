"""
Interactive CLI Tester with Amazon Bedrock AgentCore Memory.

Features:
1. Multi-turn conversation: Remembers your name, questions, and context across turns.
2. AWS AgentCore Memory integration: Reads and writes conversation events to AWS.
3. Step-by-step reasoning inspector: Shows tool calls, arguments, and live responses.
4. Session commands:
   - Type '/history' to see all past turns stored in AWS.
   - Type '/new' to start a fresh conversation session.
   - Type 'quit' or 'exit' to exit.
"""

import sys
import uuid
from graph import create_agent_graph
from memory import load_session_messages, save_conversation_turn
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def format_response(content) -> str:
    """Format string or multi-block LLM response content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                elif block.get("type") == "reasoning_content":
                    rc = block.get("reasoning_content", {})
                    if isinstance(rc, dict) and "text" in rc:
                        out.append(f"*(Thinking: {rc['text']})*")
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out).strip()
    return str(content)


def run_query(graph, query: str, session_id: str, actor_id: str = "sujat"):
    """
    Run a query through the LangGraph agent with AWS Bedrock AgentCore Memory.

    Workflow:
    1. Fetch prior conversation turns from AWS AgentCore Memory for this session.
    2. Inject prior messages + new query into LangGraph state.
    3. Run the reasoning graph (Bedrock Converse + tools).
    4. Save the completed turn (user query + assistant answer) back to AWS.
    """
    print(f"\nUser: {query}")
    print("-" * 60)

    # 1. Load conversation history from AWS Bedrock AgentCore Memory
    history = load_session_messages(session_id=session_id, actor_id=actor_id)
    if history:
        print(f"  [AgentCore Memory]: Loaded {len(history)} prior message(s) from AWS.")

    print("Agent is reasoning...")

    # 2. Build initial state with complete history + current user message
    new_message = HumanMessage(content=query)
    initial_state = {"messages": history + [new_message]}

    # 3. Invoke LangGraph
    result = graph.invoke(initial_state)
    messages = result.get("messages", [])

    # Print intermediate tool calls and tool outputs
    for msg in messages[len(history):]:  # Only print new turns
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"\n  [Tool Triggered]: {tc['name']}")
                print(f"  [Tool Arguments]: {tc['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"  [Tool Output]   : {msg.content[:200]}..." if len(str(msg.content)) > 200 else f"  [Tool Output]   : {msg.content}")

    # Print final agent answer
    final_message = messages[-1] if messages else None
    final_text = format_response(final_message.content) if final_message else "No response generated."

    print("\nAgent Response:")
    print("=" * 60)
    print(final_text)
    print("=" * 60)

    # 4. Save new turn to AWS Bedrock AgentCore Memory
    event_id = save_conversation_turn(
        session_id=session_id,
        user_text=query,
        assistant_text=final_text,
        actor_id=actor_id
    )
    if event_id:
        print(f"  [AgentCore Memory]: Turn saved to AWS (Event ID: {event_id})")


def show_history(session_id: str, actor_id: str = "sujat"):
    """Display all events stored in AWS Bedrock AgentCore Memory for this session."""
    history = load_session_messages(session_id=session_id, actor_id=actor_id)
    print("\n" + "=" * 60)
    print(f"  AWS AgentCore Memory History (Session: {session_id})")
    print("=" * 60)
    if not history:
        print("  (No conversation history found for this session yet).")
    else:
        for idx, msg in enumerate(history, 1):
            sender = "User" if isinstance(msg, HumanMessage) else "Agent"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"\n[{idx}] {sender}: {content.strip()}")
    print("=" * 60 + "\n")


def main():
    print("==========================================================")
    print("  LangGraph + Bedrock AgentCore Memory Interactive Tester ")
    print("==========================================================")
    
    print("\nInitializing compiled LangGraph agent...")
    graph = create_agent_graph()
    print("Ready!\n")

    # Default session identifier
    session_id = "session-sujat-01"
    actor_id = "sujat"

    print(f"Current Session ID : {session_id}")
    print(f"Current Actor ID   : {actor_id}")
    print("Commands:")
    print("  /history -> View full conversation stored in AWS AgentCore Memory")
    print("  /new     -> Start a new clean session")
    print("  quit     -> Exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting. Happy learning!")
                break

            if user_input.lower() == "/history":
                show_history(session_id, actor_id)
                continue

            if user_input.lower() == "/new":
                session_id = f"session-sujat-{uuid.uuid4().hex[:6]}"
                print(f"\n[Session Switched] Now using new session: {session_id}")
                continue

            run_query(graph, user_input, session_id=session_id, actor_id=actor_id)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Happy learning!")
            break


if __name__ == "__main__":
    main()
