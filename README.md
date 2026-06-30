# 🧠 Agentic Codebase Intelligence Platform (ACIP)

A production-grade multi-agent system for deep codebase understanding using RAG, AST parsing, dependency graphs, and LLM synthesis.

---

## Architecture

```
User Query
  → Orchestrator
    → Planner Agent         (decides execution steps)
    → Memory Agent          (retrieves similar past queries)
    → Retriever Agent       (semantic search via FAISS)
    → Parser Agent          (AST metadata extraction)
    → Graph Agent           (dependency reasoning via NetworkX)
    → Tool Agent            (file access, keyword search)
    → Reasoning Agent       (LLM synthesis of all context)
    → Evaluator Agent       (LLM-as-judge quality scoring)
    → Memory Agent          (stores interaction)
  → QueryResponse
```

## Project Structure

```
codebase-agent/
├── main.py                   # CLI entry point
├── config.py                 # All configuration
├── requirements.txt
│
├── ingestion/
│   ├── loader.py             # Recursive file loader
│   ├── parser.py             # Python AST parser
│   ├── summarizer.py         # LLM-based file summariser
│   ├── chunker.py            # Token-aware chunker
│   └── indexer.py            # FAISS vector index
│
├── agents/
│   ├── orchestrator.py       # Central controller ← START HERE
│   ├── planner.py            # Execution plan generator
│   ├── retriever.py          # Semantic search agent
│   ├── parser_agent.py       # On-demand AST agent
│   ├── graph_agent.py        # Dependency graph agent
│   ├── tool_agent.py         # File/search tools
│   ├── reasoning_agent.py    # LLM synthesis agent
│   ├── evaluator.py          # Answer quality agent
│   └── memory_agent.py       # Persistent memory
│
├── utils/
│   ├── llm.py                # OpenAI client wrapper
│   ├── embeddings.py         # Embedding generation
│   └── prompts.py            # All prompt templates
│
├── api/
│   └── server.py             # FastAPI REST server
│
├── memory/
│   ├── vector_store/         # FAISS index files
│   └── chat_history.json     # Persisted interactions
│
├── graph/
│   └── dependency_graph.pkl  # NetworkX graph
│
└── data/
    └── repo/                 # ← PUT YOUR CODEBASE HERE
```

---

## Quick start (web UI)

The easiest way to use this project is through the built-in web interface.

### 1. Install dependencies

```powershell
cd "c:\path\to\codebase-agent-v3"
py -m pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```powershell
$env:OPENAI_API_KEY="sk-..."
```

### 3. Start the app

```powershell
py main.py serve
```

Then open:

```text
http://127.0.0.1:8000/
```

Paste a GitHub repository URL into the input box, optionally add a branch, and click “Analyse Repository”. The app will clone the repository, index it, and let you ask questions about the codebase.

### 4. Optional: run from the command line

```powershell
py main.py ingest --repo "C:\path\to\your\project"
py main.py query --repo "C:\path\to\your\project"
```

---

## Setup

### 1. Install dependencies

```bash
cd codebase-agent
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# Option A: .env file (recommended)
echo "OPENAI_API_KEY=sk-..." > .env

# Option B: environment variable
export OPENAI_API_KEY=sk-...
```

### 3. Add your codebase

```bash
# Copy or clone the repo you want to analyse
cp -r /path/to/your/project data/repo/

# Or clone directly
git clone https://github.com/your/repo data/repo/
```

---

## Usage

### Step 1 — Ingest

Parses, summarises, chunks, and indexes the codebase:

```bash
python main.py ingest
```

This creates:
- `memory/vector_store/faiss.index` — semantic search index
- `graph/dependency_graph.pkl` — module dependency graph

### Step 2 — Query (Interactive REPL)

```bash
python main.py query
```

Built-in commands inside the REPL:
- `/history` — show recent queries
- `/files` — list indexed files
- `/quit` — exit

### Step 3 — One-Shot Query

```bash
python main.py query --ask "Explain the architecture of this repository"
python main.py query --ask "Where is model training implemented?"
python main.py query --ask "What happens if I modify data_utils.py?"
python main.py query --ask "Suggest refactoring improvements for models.py"
python main.py query --ask "Which files depend on the DataLoader class?"
```

### Step 4 — REST API

```bash
python main.py serve
# or with hot-reload:
python main.py serve --reload
```

API docs auto-generated at: `http://localhost:8000/docs`

#### API Examples

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the Trainer class do?"}'

# Ingest
curl -X POST http://localhost:8000/ingest

# Status
curl http://localhost:8000/status

# File list
curl http://localhost:8000/files

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"keyword": "fit_transform", "max_results": 10}'

# History
curl http://localhost:8000/history?limit=5
```

---

## Configuration

All settings in `config.py` (or override via `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o-mini` | Main LLM |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K_RETRIEVAL` | `5` | Chunks returned per query |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `EVALUATION_ENABLED` | `true` | Toggle LLM-as-judge |
| `MIN_ACCEPTABLE_SCORE` | `0.6` | Evaluator pass threshold |
| `API_PORT` | `8000` | FastAPI port |

---

## Demo Queries

The system is designed to handle:

| Query Type | Example |
|---|---|
| Architecture | "Explain the overall architecture of this repo" |
| Location | "Where is model training implemented?" |
| Dependency | "Which files depend on data_utils?" |
| Impact | "What happens if I modify the StandardScaler class?" |
| Refactor | "Suggest refactoring improvements for trainer.py" |
| Symbol lookup | "What does the run_experiment function do?" |

---

## Performance Metrics

Every query returns:
- **Total latency** (ms)
- **Retrieval latency** (ms)
- **Reasoning latency** (ms)
- **Chunks searched** (count)
- **Evaluation scores** (correctness, completeness, clarity)

---

## Extending the System

### Add a new agent
1. Create `agents/my_agent.py` with a class
2. Register it in `agents/orchestrator.py` `__init__` and the execution loop
3. Add it to the valid agent set in `agents/planner.py`

### Support new file types
Add the extension to `config.SUPPORTED_EXTENSIONS` and implement a parser
in `ingestion/parser.py` (the chunker and indexer are language-agnostic).

### Swap the vector store
Replace `ingestion/indexer.py` with a Chroma or Pinecone backend while
keeping the same `VectorIndex` interface (`build`, `search`, `save`, `load`).
