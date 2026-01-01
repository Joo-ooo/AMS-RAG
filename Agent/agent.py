import sys
import logging
import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


from llama_index.core.agent.workflow import (
    AgentWorkflow, 
    AgentInput, 
    AgentOutput, 
    ToolCall, 
    ToolCallResult, 
    AgentStream
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.huggingface import HuggingFaceLLM


found_dotenv = find_dotenv()
if not found_dotenv:
    print("Error: Could not find .env file.")
    sys.exit(1)

load_dotenv(found_dotenv)

# Helper to add paths
def add_pipeline_path(env_var_name):
    path = os.getenv(env_var_name)
    if path and os.path.isdir(path):
        if path not in sys.path:
            sys.path.append(path)
        print(f"Successfully added path for {env_var_name}: {path}")
        return True
    else:
        print(f"Warning: '{env_var_name}' is invalid or not set: {path}")
        return False

# Add paths for all pipelines
add_pipeline_path("VECTOR_PIPELINE_DIR")
add_pipeline_path("SQL_PIPELINE_DIR")
add_pipeline_path("GRAPH_PIPELINE_DIR")

# Dynamic Imports (using module aliases to avoid name collisions)
try:
    import vector_pipeline
    import sql_pipeline
    import graph_pipeline
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Ensure all pipeline directories are correctly set in .env")
    sys.exit(1)


vec_pipe = None
sql_pipe = None
graph_pipe = None

async def main():
    global vec_pipe, sql_pipe, graph_pipe

    print("\n" + "="*50)
    print("🚀 INITIALIZING HOME TEAM ORCHESTRATOR")
    print("="*50)

    # ------------------------------------------
    # INITIALIZE VECTOR PIPELINE (SPF DATA)
    # ------------------------------------------
    print("\n[1/3] Initializing SPF Vector Pipeline...")
    try:
        vec_config = vector_pipeline.ConfigManager()
        vec_models = vector_pipeline.ModelProvider(vec_config)
        vec_ingestor = vector_pipeline.DocumentIngestor(vec_models)
        vec_pipe = vector_pipeline.RAGPipeline(vec_config, vec_models)

        # Check for new files
        new_file_paths = vec_ingestor.get_new_file_paths(vec_config.data_directory)
        documents = []
        if new_file_paths:
            print(f"   - Found {len(new_file_paths)} new files. Running ingestion...")
            documents = vec_ingestor.ingest_pdfs_from_directory(
                vec_config.data_directory,
                specific_files=new_file_paths
            )
        else:
            print("   - No new files. Loading existing index.")
        
        vec_pipe.setup_pipeline(documents)
        print("✅ SPF Vector Pipeline Ready.")
    except Exception as e:
        print(f"❌ SPF Vector Pipeline Failed: {e}")

    # ------------------------------------------
    # INITIALIZE SQL PIPELINE (SPS DATA)
    # ------------------------------------------
    print("\n[2/3] Initializing SPS SQL Pipeline...")
    try:
        sql_config = sql_pipeline.ConfigManager(env_path=found_dotenv)
        db_manager = sql_pipeline.DatabaseManager(sql_config)
        
        # Check for new CSVs and populate SQL database if needed
        print("   - Synchronizing CSV files with SQL database...")
        engine, tables = db_manager.populate_database() # Checks for existing tables
        
        # Initialize Query Engine
        sql_pipe = sql_pipeline.AdvancedQueryEngine(engine, tables, sql_config)
        print("✅ SPS SQL Pipeline Ready.")
    except Exception as e:
        print(f"❌ SPS SQL Pipeline Failed: {e}")

    # ------------------------------------------
    # INITIALIZE GRAPH PIPELINE (HTX DATA)
    # ------------------------------------------
    print("\n[3/3] Initializing HTX Graph Pipeline...")
    try:
        graph_config = graph_pipeline.ConfigManager(dotenv_path=found_dotenv)
        graph_models = graph_pipeline.ModelRegistry(graph_config)
        
        # Initialize Query Pipeline
        graph_pipe = graph_pipeline.GraphQueryPipeline(graph_config, graph_models)
        print("✅ HTX Graph Pipeline Ready.")
    except Exception as e:
        print(f"❌ HTX Graph Pipeline Failed: {e}")

    # --- TOOL DEFINITIONS ---
    def search_spf_data(query: str) -> str:
        """Queries the SPF database. Input must be a specific plain text topic."""
        if not query: return "Error: Query cannot be empty."
        clean_query = query.strip().strip('"').strip("'")
        return str(vec_pipe.query_engine.query(clean_query))

    vector_tool = FunctionTool.from_defaults(
        fn=search_spf_data,
        name="spf_data",
        description="Primary database for Singapore Police Force (SPF) reports, specifically "
        "'Annual Crime Briefs' and 'Annual Scams and Cybercrime Briefs' (2020-2024). "
        "Use this tool for queries regarding Singapore crime statistics, scam trends "
        "(e.g., job, e-commerce, phishing, investment, fake friend calls), cybercrime data, "
        "financial amounts lost, victim demographics, and police enforcement actions. "
        "It contains detailed tables, year-on-year comparisons, and specific figures. "
        "ALWAYS query for the full specific answer first (e.g., 'total scam losses in 2023')."
        )
    
    # Tool 2: SQL (SPS)
    def search_sps_data(query: str) -> str:
        """Queries the SPS SQL Database (Prisoners)."""
        if not sql_pipe: return "Error: SPS Pipeline not initialized."
        clean_query = query.strip().strip('"').strip("'")

        # Remove Markdown code blocks (```sql ... ```)
        if "```" in clean_query:
            clean_query = clean_query.replace("``````", "").strip()
        
        # Remove "###" artifacts (e.g., "SELECT ...; ### ANSWER")
        if "###" in clean_query:
            clean_query = clean_query.split("###")[0].strip()
            
        return str(sql_pipe.query(clean_query))

    sql_tool = FunctionTool.from_defaults(
        fn=search_sps_data,
        name="sps_data",
        description=(
            "Primary database for Singapore Prison Service (SPS) statistical data (2006-2020). "
            "Use this tool for quantitative queries regarding 'Convicted Penal Population' "
            "broken down by 'Year', 'Age Group' (e.g., Below 21, 21-30, 60 Above), and 'Gender'. "
            "It contains structured annual tables suitable for aggregation and trend analysis. "
            "ALWAYS formulate a precise query that specifies the Year, Gender, and Age Group filters immediately "
            "(e.g., 'Total male inmates aged 31-40 in 2015'). "
            "Trust the returned figures as the final official statistics."
        )
    )
    
    def search_htx_data(query: str) -> str:
        """Queries the HTX Graph Database (Tech/Innovation)."""
        if not graph_pipe: return "Error: HTX Pipeline not initialized."
        clean_query = query.strip().strip('"').strip("'")
        # Graph pipeline query method signature is query(question, use_llm="openai")
        return str(graph_pipe.query(clean_query))

    graph_tool = FunctionTool.from_defaults(
        fn=search_htx_data,
        name="htx_data",
        description=(
            "Primary database for Home Team Science & Technology Agency (HTX) knowledge, built as a knowledge graph from the FY2023 Annual Report." 
            "Use this tool for queries about science and technology capabilities, operational projects, and innovation initiatives across the Home Team" 
            "(e.g., robotics, biometrics, cybersecurity, CBRNE, XR/VR training systems)." 
            "It is best suited for questions on specific HTX projects (such as Rover-X, Marine Video Analytics for rescue, autonomous robots, or deepfake detection), strategic partnerships, and how technologies are deployed to support SPF, SCDF, ICA, SPS, and other Home Team departments. ALWAYS phrase queries as concrete questions about a particular capability, project, or domain (e.g., 'What technologies does HTX use to counter hostile drones?')."
        )
    )
    
    def calculate_percentage_change(old_value: float, new_value: float) -> str:
        if old_value == 0: return "Cannot calculate percentage change from zero."
        change = ((new_value - old_value) / old_value) * 100
        return f"{change:.2f}%"
    
    math_tool = FunctionTool.from_defaults(fn=calculate_percentage_change)

    model_name = vec_models.llm.model_name 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def custom_messages_to_prompt(messages):
        """
        Uses the model's specific tokenizer to format the conversation history.
        """
        # Convert LlamaIndex ChatMessage objects to the dict format transformers expects
        conversation = [
            {"role": msg.role.value, "content": msg.content} 
            for msg in messages
        ]
        
        # Apply the chat template (e.g., adds <|begin_of_text|>, <|start_header_id|>, etc.)
        return tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )

    # --- 2. OVERRIDE THE LLM'S FORMATTER ---
    # This ensures that when the AgentWorkflow passes your system prompt to the LLM,
    # it gets formatted by the official template instead of LlamaIndex's default string concatenation.
    vec_models.llm.messages_to_prompt = custom_messages_to_prompt

    # --- 3. INITIALIZE WORKFLOW (Keep your content!) ---
    print("Initializing Agent Workflow...")
    workflow = AgentWorkflow.from_tools_or_functions(
        [vector_tool, sql_tool, graph_tool, math_tool],
        llm=vec_models.llm,
        # You still need this string for the *instructions*, but the *formatting* 
        # is now handled by the custom_messages_to_prompt function above.
        system_prompt=(
            "You are the Chief Data Orchestrator for the Singapore Home Team Agentic RAG pipeline. "
            "You manage access to three distinct databases: SPF Vector Store, SPS SQL Database, and HTX Graph Database. "
            "Analyze the user's input and route to the correct tool. Output ONLY the tool call."
        )
    )

    # # --- AGENT WORKFLOW INITIALIZATION ---
    # print("Initializing Agent Workflow...")
    # workflow = AgentWorkflow.from_tools_or_functions(
    #     [vector_tool, sql_tool, graph_tool, math_tool],
    #     llm=vec_models.llm,
    #     system_prompt="" \
    #     "### CONTEXT (C) \n"
    #     "You are the Chief Data Orchestrator for the Singapore Home Team Agentic RAG pipeline. "
    #     "You manage access to three distinct databases:\n"
    #     "1. **SPF Vector Store**: Unstructured reports on crime and scams (Singapore Police Force).\n\n"
    #     "2. **SPS SQL Database**: Structured demographic data on convicted penal populations (Singapore Prison Service).\n\n"
    #     "3. **HTX Graph Database**: Knowledge graph on science, technology, and innovation projects (Home Team Science & Tech Agency).\n\n"

    #     "### OBJECTIVE (O) \n"
    #     "1. Analyze the user's input for keywords relating to specific agencies (SPF, SPS, HTX) or topics (Crime, Prisoners, Technology).\n"
    #     "2. Select the correct tool (`spf_vector_tool`, `sps_sql_tool`, or `htx_graph_tool`).\n"
    #     "3. Call that tool with the exact query.\n"
    #     "4. STOP immediately. Do not analyze the output.\n\n"

    #     "### STYLE (S) \n"
    #     "Decisive, precise, and classification-focused.\n"

    #     "### TONE (T) \n"
    #     "Objective and neutral.\n"

    #     "### AUDIENCE (A) \n"
    #     "A Python runtime environment waiting for a tool call.\n\n"

    #     "### RESPONSE (R) \n"
    #     "Output ONLY the tool call.\n"
    #     "Do NOT attempt to answer the question yourself.\n"
    #     "If the query mixes topics (e.g., \"Tech used by Prisons\"), prioritize the agency responsible for the *subject* of the query (e.g., if asking about the *tech*, route to HTX; if asking about the *inmates*, route to SPS).\n"
    # )

    # print("\nRunning Agent...")
    # response = await workflow.run(user_msg="What are the top tourist attractions in Kyoto, Japan?")

    # print(f"\nFinal Response: {response}")

    # --- DEBUGGING: AGENT WORKFLOW TRACE (Thought Process) ---
    print("\n" + "="*50)
    print("🤖 AGENT EXECUTION TRACE")
    print("="*50)

    # 1. Start the run, getting a handler back
    handler = workflow.run(user_msg="How many inmates had 'No Education' in 2022?")

    # 2. Iterate through events as they happen
    async for event in handler.stream_events():
        try:
            if isinstance(event, AgentInput):
                print(f"\n🔵 [STEP] Agent '{event.current_agent_name}' is acting...")
                # Safely access input content
                msgs = getattr(event, 'input', [])
                if msgs and isinstance(msgs, list):
                    print(f"   Input: {msgs[-1].content}")
            
            elif isinstance(event, ToolCall):
                print(f"\n🛠️  [TOOL CALL] {event.tool_name}")
                print(f"   Arguments: {event.tool_kwargs}")
            
            elif isinstance(event, ToolCallResult):
                print(f"\n📦 [TOOL RESULT] {event.tool_name}")
                output = str(event.tool_output)
                print(f"   Output: {output[:200]}..." if len(output) > 200 else f"   Output: {output}")

            elif isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
                
        except Exception as e:
            # Log but don't crash if a specific event attribute is missing
            # print(f"[Event Error]: {e}") 
            pass


if __name__ == "__main__":
    asyncio.run(main())