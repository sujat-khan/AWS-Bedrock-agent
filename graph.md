flowchart TD
    START([START]) --> AGENT["Agent Node<br/>call_model()"]

    AGENT --> MODEL["Amazon Bedrock Converse LLM<br/>ChatBedrockConverse<br/>+ Bound Tools"]

    MODEL --> CONDITION{"tools_condition<br/>Tool calls present?"}

    CONDITION -->|Yes| TOOLS["Tools Node<br/>ToolNode(ALL_TOOLS)"]
    CONDITION -->|No| END_NODE([END])

    TOOLS --> RESULTS["Tool Execution Results<br/>ToolMessage"]

    RESULTS --> AGENT

    style START fill:#d1fae5,stroke:#059669,color:#065f46
    style END_NODE fill:#d1fae5,stroke:#059669,color:#065f46
    style AGENT fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style MODEL fill:#fef3c7,stroke:#d97706,color:#78350f
    style CONDITION fill:#f3e8ff,stroke:#9333ea,color:#581c87
    style TOOLS fill:#ffedd5,stroke:#ea580c,color:#9a3412
    style RESULTS fill:#f3f4f6,stroke:#6b7280,color:#1f2937