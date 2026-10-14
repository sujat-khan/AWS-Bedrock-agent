"""
LangGraph Agent Definition powered by AWS Bedrock.

This module defines:
1. AgentState: Schema for the state passed between graph nodes.
2. Nodes:
   - 'agent': Invokes the Bedrock Converse LLM with tools bound.
   - 'tools': Executes whichever tool the LLM decided to invoke.
3. Edges:
   - START -> 'agent'
   - Conditional edge: If tool called -> 'tools', else -> END
   - 'tools' -> 'agent' (Loop back to formulate the final answer)
"""

import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

# Load environment configuration (.env)
load_dotenv()

# System prompt defining agent persona and instructions
SYSTEM_PROMPT = (
    "You are an expert AI Cloud Assistant powered by Amazon Bedrock and LangGraph. "
    "You help users learn AWS AI services, architecture, and cloud concepts. "
    "When a user asks questions requiring live web information, current events, "
    "calculation, AWS service details, or current time, use your available tools. "
    "Be concise, friendly, and structured in your answers."
)



# 1. State Definition
class AgentState(TypedDict):
    """
    The state of our agent graph.
    The `add_messages` annotation ensures new messages from nodes are appended to
    the list rather than overwriting existing history.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_agent_graph(
    model_id: str = None,
    region_name: str = None,
    temperature: float = 0.1
):
    """
    Build and compile the LangGraph agent graph.

    Args:
        model_id: AWS Bedrock model or inference profile ID (defaults to .env)
        region_name: AWS region name (defaults to .env or ap-south-1)
        temperature: Sampling temperature for the model
    """
    # Resolve configuration from environment if not explicitly provided
    resolved_region = region_name or os.getenv("AWS_REGION", "ap-south-1")
    resolved_model = model_id or os.getenv("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")

    # Initialize Bedrock Converse model
    # ChatBedrockConverse uses the modern AWS Bedrock Converse API with native tool calling
    llm = ChatBedrockConverse(
        model=resolved_model,
        region_name=resolved_region,
        temperature=temperature,
    )

    # Bind tools to the model
    # This teaches the LLM the JSON schema of each tool so it can trigger them
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Node 1: Agent reasoning node
    def call_model(state: AgentState):
        messages = list(state["messages"])
        
        # Ensure system message is included at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        # Return state update: append the model's response
        return {"messages": [response]}

    # Node 2: Tool execution node
    tool_node = ToolNode(ALL_TOOLS)

    # 2. Graph Construction
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Add the edges defining control flow
    workflow.add_edge(START, "agent")

    # Conditional edge: If LLM generated tool_calls -> route to 'tools', otherwise route to END
    workflow.add_conditional_edges("agent", tools_condition)

    # After tools are executed, route back to the agent so it can read tool outputs
    workflow.add_edge("tools", "agent")

    # Compile the graph into an executable Runnable
    compiled_graph = workflow.compile()
    return compiled_graph


if __name__ == "__main__":
    # Quick standalone test of the graph
    print("Initializing LangGraph agent...")
    graph = create_agent_graph()
    
    test_query = "What is Bedrock AgentCore and what is 45 * 12?"
    print(f"\nUser Query: {test_query}\nRunning graph...")
    
    initial_state = {"messages": [("user", test_query)]}
    result = graph.invoke(initial_state)
    
    print("\n--- Final Agent Response ---")
    final_message = result["messages"][-1]
    
    # Handle response content (can be string or list of blocks)
    if isinstance(final_message.content, list):
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text"))
            elif isinstance(block, str):
                print(block)
    else:
        print(final_message.content)
