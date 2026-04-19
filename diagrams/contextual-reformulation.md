# Contextual Reformulation Diagram 💬

```mermaid
graph TD
    A[User Input: Raw Query] --> B{History Exists?}
    B -- No --> C[Use Original Query]
    B -- Yes --> D[Fetch max_msg_history from DB]

    D --> E[Construct Reformulation Prompt]
    subgraph LLM_Processor [Reformulation Engine]
        E --> F[Inject Context: History + Persona]
        F --> G[LLM Rewrite: De-contextualize]
    end

    G --> H[Output: Context-Aware Query]
    C --> H

    H --> I[Proceed to 2-Layer Greeting Detection Engine]

    %% Styling
    style LLM_Processor fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
    style H fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
```

---

## Description

- Take the user's raw query along with the latest `max_msg_history` messages from the chat history (both user and assistant turns).
- If no chat history exists → use the original query as-is and skip the reformulation step entirely.
- If chat history exists → pass the original query and the history to the LLM reformulation engine.
  - The LLM rewrites the query to be fully standalone and context-aware (e.g., resolves pronoun references such as "tell me more about it" into explicit standalone questions).
  - Output: a context-aware standalone reformulated query.
- Proceed with the reformulated query (or the original if no history) to the 2-Layer Greeting Detection.
