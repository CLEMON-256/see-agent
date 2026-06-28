import os
from dotenv import load_dotenv
from autogen import ConversableAgent, UserProxyAgent 
from marketing_crew import run_marketing_crew

# Load the environment variables from your .env file
load_dotenv()

# CRITICAL FIX: Updated retired model to gemini-2.5-flash
config_list = [{
    "model": "gemini-2.5-flash",
    "api_key": os.environ.get("GEMINI_API_KEY"),
    "api_type": "google"
}]

# Define the bridge tool that AutoGen uses to execute CrewAI
def trigger_marketing_pipeline(product_name: str, target_audience: str) -> str:
    """Triggers the background CrewAI team to generate a full marketing campaign."""
    print(f"\n[AutoGen] Activating CrewAI for {product_name}...")
    result = run_marketing_crew(product_name, target_audience)
    return str(result)

# Initialize the User Proxy to handle terminal input and tool execution
user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="ALWAYS", 
    code_execution_config=False
)

# Initialize the Orchestration Agent to manage the user conversation
orchestrator_agent = ConversableAgent( 
    name="Campaign_Orchestrator",
    llm_config={"config_list": config_list},
    system_message="You are the project manager. Ask the user for their product name and audience. "
                   "Once you have both details, execute the 'trigger_marketing_pipeline' tool. "
                   "Review the final output from the tool and ask the user if they want revisions."
)

# Register the tool so AutoGen knows how to call it
orchestrator_agent.register_for_llm(name="trigger_marketing_pipeline", description="Runs CrewAI engine")(trigger_marketing_pipeline)
user_proxy.register_for_execution(name="trigger_marketing_pipeline")(trigger_marketing_pipeline)

# Start the application loop
if __name__ == "__main__":
    user_proxy.initiate_chat(
        orchestrator_agent, 
        message="Hello! I want to plan a product launch campaign."
    )
