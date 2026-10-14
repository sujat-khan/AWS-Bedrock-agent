"""
Custom tools for our LangGraph Agent.
Tools give foundation models (LLMs) the ability to take actions,
query databases, perform exact calculations, and retrieve real-time facts.
"""

import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Load environment variables from .env
load_dotenv()


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Useful for math calculations, percentages, and formulas.
    Example inputs: '25 * 4', 'sqrt(144) + 10', '100 / 4'
    """
    try:
        # Allowed safe math names
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }
        # Evaluate safely without builtins
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


@tool
def aws_service_advisor(service_name: str) -> str:
    """
    Look up architecture details, key concepts, and use cases for AWS services,
    especially Bedrock, Bedrock AgentCore, Lambda, S3, and OpenSearch.
    Example inputs: 'bedrock', 'agentcore', 's3', 'lambda'
    """
    service = service_name.strip().lower()
    
    knowledge_base = {
        "bedrock": (
            "Amazon Bedrock is a fully managed AWS service that offers access to top foundation models "
            "(Anthropic Claude, Amazon Nova, Meta Llama, Mistral, etc.) via a unified API. "
            "Key features: Converse API, Guardrails for Bedrock, Knowledge Bases (RAG), and Model Evaluation."
        ),
        "agentcore": (
            "Amazon Bedrock AgentCore is an enterprise serverless platform for deploying, hosting, "
            "and scaling AI agents in production. It is framework-agnostic (works with LangGraph, CrewAI, "
            "Strands, etc.), providing managed serverless runtime (`BedrockAgentCoreApp`), persistent memory across "
            "sessions, security/workload identity, MCP gateways, and CloudWatch observability."
        ),
        "s3": (
            "Amazon S3 (Simple Storage Service) is an object storage service offering industry-leading scalability, "
            "data availability, and security. Frequently used in AI architectures for raw documents, vector store "
            "snapshots, and model artifacts."
        ),
        "lambda": (
            "AWS Lambda is a serverless, event-driven compute service that lets you run code without provisioning "
            "or managing servers. In agentic workflows, Lambda functions often act as tool execution targets."
        ),
        "opensearch": (
            "Amazon OpenSearch Service provides managed search, analytics, and vector database capabilities. "
            "It is commonly paired with Amazon Bedrock Knowledge Bases to perform hybrid keyword + semantic vector search."
        ),
    }

    # Match key substring
    for key, description in knowledge_base.items():
        if key in service:
            return f"AWS Service Knowledge [{key.upper()}]:\n{description}"
            
    return (
        f"AWS Service '{service_name}': General AWS cloud service. "
        "Can be integrated into agentic architectures via AWS SDK (Boto3) or AgentCore Gateways."
    )


@tool
def get_system_time() -> str:
    """
    Get the current Indian Standard Time (IST) date and time.
    Use this tool whenever the user asks for the current time,
    today's date, or time-sensitive calculations.
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return f"Current Indian Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')}"

# Tavily web search tool
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
    include_answer=True,
)

# List of all available tools to bind to our agent
ALL_TOOLS = [calculate, aws_service_advisor, get_system_time, tavily_search]
