# YouTube-to-Graph Pipeline — Source Design

> **Provenance.** Pasted by the user on 2026-11-05 as the design for a fully local YouTube-to-Graph summarization pipeline in the `ytb` repo (https://github.com/dangnguyenducanh/ytb). Reproduced verbatim below (formatting normalised) as the source of truth for `docs/superpowers/plans/2026-11-05-ytb-youtube-to-graph.md`.
>
> **User directive (2026-11-05, verbatim):** "https://huggingface.co/Qwen/Qwen3-0.6B use this model". How this applies to this pipeline's local LLM is recorded in the plan's *Decisions* section. Anything else the plan deviates from or completes is recorded there too.

---

Building a fully local YouTube-to-Graph summarization pipeline with LangGraph works best by separating deterministic ingestion (fetching the transcript) from the probabilistic LLM steps (summarizing and extracting entities/edges), then routing the structured output to a graph renderer.

### 1. System Architecture & Tech Stack

* **Orchestration:** `LangGraph` (manages state transitions, parallel execution, and validation retries).
* **Local LLM Runtime:** `Ollama` or `vLLM` running **Llama 3.1 (8B)** or **Qwen 2.5 (7B/14B)**. Both models handle structured JSON extraction (`with_structured_output`) reliably on consumer GPUs (8GB–16GB VRAM).
* **Ingestion Layer:** `youtube-transcript-api` (fast caption scraping) + `yt-dlp` and `faster-whisper` (local audio fallback if the video has no captions).
* **Graph Visualization & Storage:** `NetworkX` + `PyVis` (generates interactive HTML knowledge graphs locally), `Mermaid.js` (for flowcharts/mindmaps), or `Neo4j` (if you want persistent multi-video graph queries).
* **UI Layer:** `Streamlit` or `Chainlit` (native support for rendering HTML graphs and streaming LangGraph steps).

### 2. LangGraph State & Workflow Design

The pipeline uses a shared `TypedDict` state that passes the raw text into two parallel or sequential branches: **Summarization** and **Knowledge Graph Extraction**.

```
[Input URL]
     │
     ▼
[1. Ingest Node] ──(No Captions?)──► [Whisper Fallback]
     │
     ▼
[2. Chunker & Cleaner]
     │
     ├─────────────────────────┐
     ▼                         ▼
[3a. Map-Reduce Summary]  [3b. Entity & Relation Extractor]
     │                         │
     │                         ▼
     │                    [4. Graph Deduplicator & Validator]
     │                         │
     └────────────┬────────────┘
                  ▼
       [5. Graph Renderer (PyVis / Mermaid)]
```

**State Definition:**

```python
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

class Node(BaseModel):
    id: str = Field(description="Unique entity name, normalized to Title Case")
    type: str = Field(description="Category: Concept, Person, Tool, Event, etc.")

class Edge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relationship: str = Field(description="Short active verb phrase, e.g., 'depends_on', 'created_by'")

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class GraphState(TypedDict):
    youtube_url: str
    transcript: str
    chunks: List[str]
    summary: str
    raw_extractions: Annotated[List[KnowledgeGraph], operator.add]
    final_graph: KnowledgeGraph
    html_graph_path: str
```

### 3. Step-by-Step Node Breakdown

**Node 1: Ingestion (`ingest_transcript`)**
Extract the video ID from the URL and fetch the transcript. If `youtube-transcript-api` raises `TranscriptsDisabled`, trigger `yt-dlp` to download a low-bitrate `.mp3` and pass it through `faster-whisper` (`base` or `small` model) locally.

**Node 2: Semantic Chunking (`chunk_text`)**
Local 8B models lose accuracy on entity extraction when context exceeds 4,000–6,000 tokens ("lost in the middle" phenomenon). Split the transcript into 2,000-token chunks with a 200-token overlap using `RecursiveCharacterTextSplitter`.

**Node 3a: Summarization (`summarize_node`)**

* *Short videos (<8k tokens):* Single-pass structured prompt generating an executive summary, key takeaways, and timestamped topics.
* *Long videos (>8k tokens):* Use LangGraph's `Send` API to run a Map-Reduce loop—summarize each chunk in parallel, then combine them in a final reducer prompt.

**Node 3b: Graph Extraction (`extract_graph_node`)**
Bind the `KnowledgeGraph` Pydantic schema to your local Ollama model:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
structured_llm = llm.with_structured_output(KnowledgeGraph)
```

Prompt the model to extract the core concepts and how they interact across each chunk.

**Node 4: Entity Resolution & Deduplication (`resolve_graph_node`)**
Local LLMs often create duplicate nodes across chunks (e.g., `"LangGraph"`, `"Langgraph"`, and `"LangGraph Framework"`).

* *Rule-based pass:* Lowercase and strip whitespace/punctuation to merge exact matches.
* *Embedding/LLM pass:* Use local embeddings (`nomic-embed-text` via Ollama) to compute cosine similarity between node names, or run a final LLM cleanup prompt that takes the list of raw nodes and outputs a canonical mapping dictionary before assembling the final edge list. Drop any "dangling edges" where the source or target node doesn't exist.

**Node 5: Graph Drawing (`render_graph_node`)**
Choose your visualization style based on what you want the graph to represent:

* **Interactive Concept Network (`PyVis` + `NetworkX`):** Best for exploring entities and relationships. Color-code nodes by `type` and scale node size by degree centrality (number of connections).
* **Hierarchical Mindmap / Flowchart (`Mermaid.js`):** Have the LLM output valid Mermaid syntax (`graph TD; A-->B;`) if you want a top-down structural breakdown of the video's argument rather than an entity network.

```python
import networkx as nx
from pyvis.network import Network

def render_pyvis(kg: KnowledgeGraph, output_path="graph.html"):
    G = nx.DiGraph()
    for node in kg.nodes:
        G.add_node(node.id, label=node.id, title=node.type, group=node.type)
    for edge in kg.edges:
        if G.has_node(edge.source) and G.has_node(edge.target):
            G.add_edge(edge.source, edge.target, label=edge.relationship)

    net = Network(height="600px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=150, spring_length=200)
    net.save_graph(output_path)
    return output_path
```

### 4. Practical Bottlenecks & Edge Cases to Handle

* **JSON Hallucinations on Local Models:** Smaller models occasionally fail strict Pydantic validation. Wrap the extraction node in a LangGraph conditional retry loop (up to 2 retries) that feeds the validation error back to the LLM if parsing fails.
* **Hairball Graphs:** A 1-hour podcast will generate hundreds of noisy nodes. Restrict your extraction prompt to *"Extract at most 10 core entities and 15 primary relationships per chunk"* and filter out low-degree nodes (nodes with only 1 connection) in `NetworkX` before rendering.
* **Transcript Noise:** Auto-generated YouTube captions lack punctuation and contain filler words. Include a lightweight system prompt instruction during chunking or extraction to ignore sponsor reads, intros, and conversational filler.
