# 2-Layer Greeting Detection Diagram 👋

```mermaid
graph TD
    %% Define Nodes
    In([Input: Reformulated Query]) --> L1_Regex{Layer 1: <br/>Regex Lookup}

    %% Layer 1 Flow
    L1_Regex -- "Match (Fast Track)" --> LLM_Greet[LLM: Generate <br/>Friendly Response]
    L1_Regex -- "No Match" --> L2_Classifier[Layer 2: <br/>LLM Intent Classifier]

    %% Layer 2 Flow
    subgraph L2_Gate [Intent Gate]
        L2_Classifier --> L2_Prompt[Analyze Query Type:<br/>Is this Small Talk or Factual?]
        L2_Prompt --> L2_Decision{Classification}
    end

    L2_Decision -- "GREETING" --> LLM_Greet
    L2_Decision -- "FACTUAL" --> RAG_Fork[[Trigger Dual RAG Pipeline:<br/>Self-RAG + Co-RAG]]

    %% Termination
    LLM_Greet --> Out([Output: Greeting to UI])
    RAG_Fork --> End((Continue to Search Engine))

    %% Styling
    style L1_Regex fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style L2_Gate fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style RAG_Fork fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style LLM_Greet fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
```

---

## Description

- Input: reformulated query (from contextual query reformulation).

**Layer 1 – Regex Lookup:**

- Lookup the reformulated query against a robust, multi-language greeting regex pattern list.
- If the query matches any pattern → greeting detected at Layer 1 → call the LLM to generate a friendly greeting response using its general knowledge and the chat history → return the response to the user and end the pipeline.
- If no pattern matches → no greeting detected at Layer 1 → proceed to Layer 2.

**Layer 2 – LLM Intent Classifier:**

- Call the LLM to classify the reformulated query as either a GREETING (small talk) or a FACTUAL question.
- If the LLM classifies as GREETING → call the LLM to generate a friendly greeting response using its general knowledge and the chat history → return the response to the user and end the pipeline.
- If the LLM classifies as FACTUAL → the query is a real question → continue to the Dual RAG Pipeline (Self-RAG + Co-RAG).
