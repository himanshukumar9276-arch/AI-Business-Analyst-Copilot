# AI Business Analyst Copilot

A multi-agent LLM system that answers natural-language business questions by routing them to the right specialized agent — SQL, document retrieval (RAG), live web search, or general reasoning — instead of relying on a single monolithic prompt.

> One question, four agents, zero manual SQL.

## What it does

Ask a question in plain English. A supervisor agent classifies the question and routes it to the right specialist:

| Question type | Routed to | Example |
|---|---|---|
| Structured business data | **SQL Agent** | "Which genre generated the most revenue last year?" |
| Internal company knowledge | **RAG Agent** | "What's our customer return policy?" |
| Current events / external info | **Web Agent** | "What are the latest trends in music streaming?" |
| General reasoning | **LLM Agent** | "Explain what a foreign key is." |

The system maintains conversational memory across turns, so follow-up questions ("what about the quarter before that?") work without re-stating context.

## Architecture

```
                     ┌─────────────┐
                     │  Supervisor │  (LLM + structured output)
                     └──────┬──────┘
                            │
        ┌───────────┬───────┴───────┬────────────┐
        ▼           ▼               ▼            ▼
   ┌─────────┐ ┌─────────┐    ┌───────────┐ ┌──────────┐
   │   SQL   │ │   RAG   │    │    Web    │ │   LLM    │
   │  Agent  │ │  Agent  │    │   Agent   │ │  Agent   │
   └─────────┘ └─────────┘    └───────────┘ └──────────┘
```

Built with **LangGraph** as a stateful graph: a supervisor node routes to one of four leaf nodes via conditional edges, with `MemorySaver` checkpointing for persistent multi-turn conversation state (per session thread ID).

### SQL Agent
- Generates SQL from natural language against the schema (no hardcoded queries)
- Validates every query with `sqlparse` at the AST level — rejects anything that isn't a `SELECT` statement, rather than relying on keyword string-matching
- Runs `EXPLAIN QUERY PLAN` before execution as an extra safety check
- On failure, feeds the exact database error back into a correction prompt and retries (up to 3 attempts) before failing gracefully

### RAG Agent
- Retrieves from a **FAISS** vector store built over 9 internal company documents (return policy, employee handbook, vendor agreements, etc.)
- Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Strictly grounded: if the answer isn't in the retrieved context, it says so explicitly instead of guessing

### Web Agent
- Uses the **Tavily Search API** for real-time information beyond the LLM's training cutoff

### Supervisor
- A structured-output LLM call (Pydantic schema) picks one of `sql | rag | web | llm` — avoids the reliability issues of parsing a raw text response for routing decisions

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | Groq API (Llama 3.3 70B) |
| Orchestration | LangGraph, LangChain |
| Vector search | FAISS + Sentence-Transformers |
| Database | SQLite (Chinook — 11 tables) |
| SQL safety | sqlparse (AST-level validation) |
| Web search | Tavily API |
| Frontend | Streamlit |
| Observability | LangSmith |
| Config/validation | python-dotenv, Pydantic |

## Project Structure

```
project/
├── agents/
│   ├── model.py              # Shared LLM client (Groq)
│   ├── cust_sql_agent.py     # SQL generation, validation, retry loop
│   ├── rag_agent.py          # Runtime RAG retrieval + answering
│   ├── rag_agent_setup.py    # One-time script: builds the FAISS index
│   └── web_agent.py          # Tavily-backed web search agent
├── graph/
│   ├── state.py              # LangGraph state schema
│   ├── supervisior.py        # Routing logic (structured output)
│   ├── nodes.py               # Node functions wrapping each agent
│   └── graph_builder.py      # Graph wiring + conditional routing + memory
├── data/
│   ├── Chinook.sqlite        # Business database
│   ├── docs/                 # Source PDFs for RAG
│   └── faiss_index/          # Prebuilt vector index
├── main.py                   # CLI entry point
├── app.py                    # Streamlit chat UI
└── config.py                 # Environment/config loading
```

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone <your-repo-url>
   cd project
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** — create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   LANGSMITH_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```

4. **(Optional) Rebuild the RAG index** — a prebuilt FAISS index is already included under `data/faiss_index/`. Only run this if you change the source documents in `data/docs/`:
   ```bash
   python agents/rag_agent_setup.py
   ```

5. **Run it**

   CLI:
   ```bash
   python main.py
   ```

   Streamlit UI:
   ```bash
   streamlit run app.py
   ```

## Design Notes

A few deliberate scoping decisions, and why:

- **Chinook over a more complex e-commerce dataset** — a clean, well-understood 11-table relational schema keeps the focus on the agent architecture and routing logic rather than data modeling.
- **A supervisor-router over a longer fixed pipeline** — each agent is independently testable and swappable, rather than a multi-stage pipeline where a failure in one stage silently breaks the next.
- **A custom SQL agent over LangChain's `create_sql_agent`** — the built-in agent bundles query generation and execution into one opaque loop. Building it manually keeps the generate → validate → execute → retry cycle fully visible and debuggable.

## Known Limitations

- SQL validation blocks non-`SELECT` statements but is not a full production-grade security layer (no row-level access control, no query cost limits).
- The Chinook schema has no cost/margin fields, so profitability-style questions aren't answerable from the current dataset.
- Conversational memory is scoped to a single session thread; there's no persistent user history across app restarts.
