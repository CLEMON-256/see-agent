import os
import sys
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv

# --- Production Clean Stack ---
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb

# Modern Enterprise Tools & State Machine
from tavily import TavilyClient  # Direct, clean enterprise search connection
from langgraph.graph import StateGraph, START, END

# Initialize System Environment
load_dotenv()

# Guard block checking critical system credentials
if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    print("[FATAL] Required system variables 'GEMINI_API_KEY' or 'TAVILY_API_KEY' are missing.")
    sys.exit(1)

# =====================================================================
# 1. DATA FABRIC LAYER (Persistent ChromaDB Vector Store Engine)
# =====================================================================
print("[System Init] Connecting to Local Persistent ChromaDB Vector Store...")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

Settings.llm = GoogleGenAI(
    model_name="models/gemini-2.5-flash", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Initialize on-disk persistent vector database folder
db_storage_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_storage_path)
chroma_collection = chroma_client.get_or_create_collection("ugandan_business_registry")

# Bind Chroma to LlamaIndex Storage Context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Industry Static Context Data Matrix
ugandan_registry_payload = [
    Document(text="URSB Company Registration Fee in Uganda is 50,000 UGX. Stamp duty is 10,000 UGX. Legal representation is standard for local incorporation."),
    Document(text="Fintech and Mobile Money compliance in Uganda is strictly governed by the National Payment Systems (NPS) Act 2020 managed by Bank of Uganda. Minimum paid up capital for Electronic Money Issuers is 250,000,000 UGX."),
    Document(text="Agricultural tax exemptions apply to processing equipment imported into Uganda under the East African Community Customs Management Act."),
    Document(text="Under the National Environment Act, developers planning industrial processing, manufacturing plants, or large-scale commercial structures must complete an Environmental Impact Assessment (EIA) for NEMA clearance.")
]

if chroma_collection.count() == 0:
    print("[ChromaDB] Embedding and inserting enterprise documents on-disk...")
    db_vector_index = VectorStoreIndex.from_documents(ugandan_registry_payload, storage_context=storage_context)
else:
    print(f"[ChromaDB] Existing index detected with {chroma_collection.count()} vectors. Loading indices...")
    db_vector_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

db_query_engine = db_vector_index.as_query_engine()

# =====================================================================
# 2. ENTERPRISE TAVILY CLIENT
# =====================================================================
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# =====================================================================
# 3. STATE MACHINE GRAPH (LangGraph Implementation)
# =====================================================================
class GlobalOrchestratorState(TypedDict):
    query: str
    db_results: str
    web_results: str
    final_synthesis: str

def node_query_internal_database(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph] Node 1: Extracting Compliance Data via ChromaDB...")
    response = db_query_engine.query(state["query"])
    return {"db_results": str(response)}

def node_query_external_networks(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph] Node 2: Executing Live Internet Search via Tavily API...")
    try:
        # Clean execution call straight to Tavily API
        search_result = tavily_client.search(query=state["query"], search_depth="advanced")
        
        # Format the search results cleanly
        context_string = "\n".join([f"- {res['title']}: {res['content']} ({res['url']})" for res in search_result['results']])
        return {"web_results": context_string}
    except Exception as e:
        print(f"[Warning] Web search failed. Falling back to database assets. Details: {e}")
        return {"web_results": "Live web search unavailable due to network boundaries."}

def node_synthesize_intelligence_streams(state: GlobalOrchestratorState) -> Dict[str, Any]:
    print("\n[LangGraph] Node 3: Synthesizing Contextual Data Streams via Gemini...")
    
    synthesis_prompt = (
        f"You are a Senior Business Consultant for East African market entry. "
        f"Synthesize the following data fields into a professional, comprehensive executive business proposal report for Uganda.\n\n"
        f"User Query: {state['query']}\n\n"
        f"Internal Registry & Legal Database Context:\n{state['db_results']}\n\n"
        f"Live Web Search Market Signals:\n{state['web_results']}\n"
    )
    
    # Use standard LlamaIndex LLM wrapper to structure the response text safely
    ai_response = Settings.llm.complete(synthesis_prompt)
    unified_market_profile = str(ai_response)
    
    try:
        with open("uganda_market_report.md", "w") as f:
            f.write(unified_market_profile)
        print("[System Info] Final report exported successfully to 'uganda_market_report.md'")
    except Exception as e:
        print(f"[System Error] Failed to export report file: {e}")

    return {"final_synthesis": unified_market_profile}

# Build Out the State Machine Structure
state_fabric = StateGraph(GlobalOrchestratorState)
state_fabric.add_node("search_database", node_query_internal_database)
state_fabric.add_node("search_web", node_query_external_networks)
state_fabric.add_node("synthesize_data", node_synthesize_intelligence_streams)

state_fabric.add_edge(START, "search_database")
state_fabric.add_edge("search_database", "search_web")
state_fabric.add_edge("search_web", "synthesize_data")
state_fabric.add_edge("synthesize_data", END)

compiled_runtime_graph = state_fabric.compile()

# =====================================================================
# 4. CRASH-PROOF INTERACTION INTERFACE
# =====================================================================
if __name__ == "__main__":
    print("\n[System Online] Enterprise Multi-Agent Graph Engine Initialized.")
    print("================================================================")
    print("Hello! I am your advanced multi-agent business advisor.")
    
    while True:
        print("\n" + "="*64)
        user_query = input("Enter your business query for Uganda (or type 'exit' to quit):\n> ")
        
        if user_query.strip().lower() == 'exit':
            print("[System Info] Closing agent fabric session. Goodbye.")
            break
            
        if user_query.strip():
            print(f"\n[System Core] Launching LangGraph pipeline execution...")
            
            runtime_initial_state = {
                "query": user_query,
                "db_results": "",
                "web_results": "",
                "final_synthesis": ""
            }
            
            graph_execution_output = compiled_runtime_graph.invoke(runtime_initial_state)
            print("\n" + graph_execution_output["final_synthesis"])
            print("\n[Process Complete] Review the file 'uganda_market_report.md' in your sidebar.")
        else:
            print("[System Warning] Please enter a valid question.")
