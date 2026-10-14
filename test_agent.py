"""
Interactive CLI Tester for LangGraph + AWS Bedrock + Bedrock AgentCore.

Run this script to test the agent interactively in your terminal:
    python test_agent.py

Or run automated demo scenarios:
    python test_agent.py --demo
"""

import sys
from graph import create_agent_graph
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


def run_query(graph, query: str):
    """Run a query through the LangGraph agent and print step-by-step reasoning."""
    print(f"\nUser: {query}")
    print("-" * 60)
    print("Agent is reasoning...")

    initial_state = {"messages": [HumanMessage(content=query)]}
    result = graph.invoke(initial_state)

    messages = result.get("messages", [])

    # Print intermediate tool calls and tool outputs
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"\n  [Tool Triggered]: {tc['name']}")
                print(f"  [Tool Arguments]: {tc['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"  [Tool Output]   : {msg.content}")

    # Print final agent answer
    final_message = messages[-1] if messages else None
    if final_message:
        print("\nAgent Response:")
        print("=" * 60)
        print(format_response(final_message.content))
        print("=" * 60)


def run_demo(graph):
    """Run predefined demo questions demonstrating tools and cloud reasoning."""
    print("\nRunning Automated Demo Queries...")
    
    demo_queries = [
        "What is Amazon Bedrock AgentCore and how does it compare to raw Bedrock?",
        "Can you calculate the square root of 625 plus 175?",
        "What is the current UTC time, and what is Amazon S3 used for?",
    ]

    for q in demo_queries:
        run_query(graph, q)
        print("\n" + "#" * 60 + "\n")


def main():
    print("==========================================================")
    print("  LangGraph + Amazon Bedrock + Bedrock AgentCore Tester   ")
    print("==========================================================")
    
    print("\nInitializing compiled LangGraph agent...")
    graph = create_agent_graph()
    print("Ready!\n")

    if "--demo" in sys.argv:
        run_demo(graph)
        return

    print("Type your questions below (or type 'quit' / 'exit' to stop).")
    print("Try asking about AWS services, math calculations, or the current time!")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting. Happy learning!")
                break

            run_query(graph, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Happy learning!")
            break


if __name__ == "__main__":
    main()
