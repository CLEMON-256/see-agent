 Multi-Framework Agentic Orchestration Subsystem (Uganda Market Intelligence)

An experimental integration blueprint combining five modern AI agent frameworks into a resilient, fault-tolerant data pipeline powered by **Google Gemini 2.5-Flash** and local vector embeddings.

## 🏗️ Architectural Topology

1. **User Interface Gateway (AutoGen / AG2)**: Manages open-ended, bidirectional conversation loops with human operators without blocking execution threads.
2. **State Machine Fabric (LangGraph)**: Eliminates stochastic agent wandering by confining operations into a deterministic, acyclic execution graph.
3. **Enterprise Data Fabric (LlamaIndex + ChromaDB)**: Provides semantic search query matching against real, on-disk persistent vector tables.
4. **Autonomous Scraping Node (CrewAI + LangChain)**: Deploys isolated worker agents backed by resilience policies (`tenacity`) to ingest real-time external network trends.

## ⚠️ Architectural Evaluation & Over-Engineering Disclosures (Read Before Deploying)

While highly functional as a framework exploration workspace, deploying this full library stack into an enterprise production environment introduces several performance trade-offs:

* **Dependency Overhead**: Importing five core agent libraries introduces massive dependency graphing conflicts, framework deprecation issues, and system weight.
* **Redundant Capabilities**: `LangGraph` alone contains state patterns capable of handling tool routing (replacing AutoGen) and role-based workflows (replacing CrewAI). In a strict production system, this codebase would be refactored down to a lean **LangGraph + Native Google GenAI SDK** architecture to save memory footprints and network latencies.
* **Token Bloat**: Passing contextual states between independent library frameworks increases prompt token overhead. 