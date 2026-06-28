import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM

# Load the environment variables from the .env file
load_dotenv()

def run_marketing_crew(product_name, target_audience):
    # CRITICAL FIX: Updated retired model to gemini/gemini-2.5-flash
    gemini_llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY")
    )

    # Define the execution agents and pass the LLM
    researcher = Agent(
        role="Market Researcher",
        goal=f"Analyze top competitors for {product_name}.",
        backstory="Expert market analyst specializing in SaaS products.",
        llm=gemini_llm,
        verbose=True
    )
    
    writer = Agent(
        role="Content Copywriter",
        goal=f"Create a high-converting launch email sequence for {target_audience}.",
        backstory="Conversion copywriter with a track record of viral launches.",
        llm=gemini_llm,
        verbose=True
    )

    # Define sequential execution tasks
    task_research = Task(
        description=f"Identify 3 unique selling points for {product_name}.",
        expected_output="A bulleted list of 3 USPs.",
        agent=researcher
    )

    task_write = Task(
        description=f"Write a 3-part email sequence based on the researched USPs.",
        expected_output="Markdown formatted email templates.",
        agent=writer
    )

    # Assemble and run the CrewAI workflow
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task_research, task_write],
        process=Process.sequential
    )

    return crew.kickoff()
