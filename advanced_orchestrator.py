import os
import sys
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv

# --- Framework Blueprint & Core Dependents ---
from autogen import ConversableAgent, UserProxyAgent
from crewai import Agent as CrewAgent, Crew, Process, Task, LLM as CrewLLM

# LlamaIndex Production Import Topology
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini as LlamaGemini

# Extensions & State Machines
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

# Load local operational configurations
load_dotenv()

# Integrity barrier validation for target system key vectors
if not os.environ.get("GEMINI_API_KEY"):
    print("[FATAL] Environment security variable 'GEMINI_API_KEY' is not loaded. Process terminated.")
    sys.exit(1)

# =====================================================================
# 1. COMPUTE FABRIC INJECTION (Google Gemini Platform Matrix)
# =====================================================================
autogen_llm_config = {
    "config_list": [{
        "model": "gemini-2.5-flash",
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "api_type": "google"
    }]
}

crew_compute_engine = CrewLLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# =====================================================================
# 2. INTERNAL ARCHIVE FABRIC (LlamaIndex Enterprise Multi-Source Engine)
# =====================================================================
print("[System Init] Instantiating Corporate Knowledge Fabric...")

# SENIOR FIX: Swapped out the deprecated text-embedding-004 model
# for the globally active, production-supported gemini-embedding-001 endpoint
Settings.embed_model = GeminiEmbedding(
    model_name="models/gemini-embedding-001", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

Settings.llm = LlamaGemini(
    model_name="models/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Immutable data matrices representing core institutional parameters
ugandan_registry_payload = [
    Document(text="URSB Company Registration Fee in Uganda is 50,000 UGX. Stamp duty is 10,000 UGX."),
    Document(text="Fintech and Mobile Money compliance in Uganda is strictly governed by the National Payment Systems (NPS) Act 2020 managed by Bank of Uganda."),
    Document(text="Agricultural tax exemptions apply to processing equipment imported into Uganda under the East African Community Customs Management Act.")
]

# This line invokes the embedding generation process safely over the active endpoint
db_vector_index = VectorStoreIndex.from_documents(ugandan_registry_payload)
db_query_engine = db_vector_index.as_query_engine()

# =====================================================================
# 3. INTERNET SCRAPER SUITE (LangChain Network Connectors)
# =====================================================================
live_network_scraper = DuckDuckGoSearchRun()

# =====================================================================
# 4. MICRO-AGENT CLUSTER EXECUTION (CrewAI Structural Worker)
# =====================================================================
def execute_background_research_crew(target_business_model: str) -> str:
    """Encapsulates a focused research crew using LangChain tools to mine live data."""
    print(f"\n[CrewAI Cluster] Allocating compute node for operational crawling: '{target_business_model}'")
    
    research_agent = CrewAgent(
        role="Ugandan Competitive Intelligence Specialist",
        goal=f"Scrape real-time, live pricing, operational variables, and trends for: {target_business_model} inside Uganda.",
        backstory="An expert market crawler based in Kampala who analyzes macro-economic adjustments and local business footprints.",
        llm=crew_compute_engine,
        verbose=True
    )
    
    execution_task = Task(
        description=f"Query open web networks to find active market signals, competitor vulnerabilities, and costs for: {target_business_model}",
        expected_output="An analytical market log containing verified numbers, active hurdles, and operational variables.",
        agent=research_agent
    )
    
    operational_crew = Crew(
        agents=[research_agent],
        tasks=[execution_task],
        process=Process.sequential
    )
    
    raw_execution_result = operational_crew.kickoff()
    return str(raw_execution_result)

# =====================================================================
# 5. STATE TRANSITION GRAPH (LangGraph State Machine Orchestration)
# =====================================================================
class GlobalOrchestratorState(TypedDict):
    query: str
    db_results: str
    web_results: str
    final_synthesis: str

def node_query_internal_database(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph Engine] Routing Execution State -> LlamaIndex Retrieval Node...")
    query_string = state["query"]
    retrieved_data = db_query_engine.query(query_string)
    return {"db_results": str(retrieved_data)}

def node_query_external_networks(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph Engine] Routing Execution State -> CrewAI Research Cluster Node...")
    query_string = state["query"]
    scraped_data = execute_background_research_crew(query_string)
    return {"web_results": scraped_data}

def node_synthesize_intelligence_streams(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph Engine] Routing Execution State -> Context Synthesis Consolidation Node...")
    unified_market_profile = (
        f"=====================================================================\n"
        f"🕵️‍♂️ ENTERPRISE BUSINESS INTEL PROFILE FOR UGANDA\n"
        f"=====================================================================\n\n"
        f"📥 [LAYER 1 - INTERNAL RETRIEVAL / LLAMAINDEX DATABASE ENDPOINTS]:\n"
        f"{state['db_results']}\n\n"
        f"🌐 [LAYER 2 - LIVE WEB SEARCH / CREWAI + LANGCHAIN DATA MINING]:\n"
        f"{state['web_results']}\n\n"
        f"=====================================================================\n"
    )
    return {"final_synthesis": unified_market_profile}

# Construct Declarative Graph
state_fabric = StateGraph(GlobalOrchestratorState)

# Connect Nodes to State Topology
state_fabric.add_node("search_database", node_query_internal_database)
state_fabric.add_node("search_web", node_query_external_networks)
state_fabric.add_node("synthesize_data", node_synthesize_intelligence_streams)

# Enforce State Edge Boundaries
state_fabric.add_edge(START, "search_database")
state_fabric.add_edge("search_database", "search_web")
state_fabric.add_edge("search_web", "synthesize_data")
state_fabric.add_edge("synthesize_data", END)

# Compile Runtime Engine
compiled_runtime_graph = state_fabric.compile()

# =====================================================================
# 6. EXTERNAL CLIENT GATEWAY (AutoGen Agent Chat Interface)
# =====================================================================
def master_workflow_bridge(business_query: str) -> str:
    """The master tool executed by AutoGen to activate LangGraph's processing topology."""
    print(f"\n[AutoGen Broker] Payload intercepted. Passing to backend graph runtime for: '{business_query}'")
    
    runtime_initial_state = {
        "query": business_query,
        "db_results": "",
        "web_results": "",
        "final_synthesis": ""
    }
    
    graph_execution_output = compiled_runtime_graph.invoke(runtime_initial_state)
    return graph_execution_output["final_synthesis"]

# Initialize human interface agent
user_interface_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS", 
    code_execution_config=False
)

# Initialize master advisory runtime agent
corporate_advisor_agent = ConversableAgent( 
    name="Advanced_Business_Consultant",
    llm_config=autogen_llm_config,
    system_message="You are an elite enterprise Business Advisor specializing in East African market entry. "
                   "Engage the client conversationally to evaluate their strategic goals. "
                   "Once they detail a business model, immediately route the execution payload via the 'master_workflow_bridge' tool. "
                   "Deliver the returned multi-layer compilation with high architectural value."
)

# Structural binding linking AutoGen orchestration layer to the execution pipeline
corporate_advisor_agent.register_for_llm(name="master_workflow_bridge", description="Launches the backend enterprise data and scraping pipeline")(master_workflow_bridge)
user_interface_proxy.register_for_execution(name="master_workflow_bridge")(master_workflow_bridge)

# Unified Thread Entrypoint
if __name__ == "__main__":
    print("\n[System Online] Core Sub-Graph Bridges Operational.")
    user_interface_proxy.initiate_chat(
        corporate_advisor_agent, 
        message="Hello! I am your advanced multi-agent business advisor. Ask me any complex business model or regulatory question regarding Uganda."
    )
