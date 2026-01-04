import streamlit as st
import asyncio
import sys
import os
import re
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index.core.agent.workflow import (
    AgentWorkflow, 
    AgentInput, 
    ToolCall, 
    ToolCallResult, 
    AgentStream
)
from llama_index.core.tools import FunctionTool

# Import agent module for helper functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import add_pipeline_path

# Load environment variables
found_dotenv = find_dotenv()
if found_dotenv:
    load_dotenv(found_dotenv)

# Add pipeline paths
add_pipeline_path("VECTOR_PIPELINE_DIR")
add_pipeline_path("SQL_PIPELINE_DIR")
add_pipeline_path("GRAPH_PIPELINE_DIR")

# Import pipeline modules (after paths are added)
import vector_pipeline
import sql_pipeline
import graph_pipeline

# Initialize session state for caching pipelines
if 'pipelines_initialized' not in st.session_state:
    st.session_state.pipelines_initialized = False
    st.session_state.vec_pipe = None
    st.session_state.sql_pipe = None
    st.session_state.graph_pipe = None
    st.session_state.workflow = None
    st.session_state.vec_models = None

def format_thinking(text: str) -> str:
    """Format thinking output - clean up Thoughts, remove Actions, and HIDE the Answer."""
    if not text:
        return ""

    # STRICTLY CUT OFF at "Answer:" 
    # This ensures the answer never appears in the thinking block
    if "Answer:" in text:
        text = text.split("Answer:")[0]

    # Remove Action and Action Input lines
    # Matches "Action: ..." up to a newline or end of string
    text = re.sub(r'Action:.*?(\n|$)', '', text)
    text = re.sub(r'Action Input:.*?(\n|$)', '', text)
    
    # Format Thoughts to start on new lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Handle multiple thoughts or thoughts embedded in lines
        if 'Thought:' in line:
            parts = line.split('Thought:')
            for i, part in enumerate(parts):
                clean_part = part.strip()
                if clean_part:
                    # If i > 0, it came after a "Thought:" delimiter, so add the prefix back
                    # If i == 0 and the line didn't start with Thought, it's pre-thought text
                    if i > 0:
                        lines.append(f"Thought: {clean_part}")
                    elif not line.startswith("Thought:"):
                         lines.append(clean_part)
        else:
            lines.append(line)
    
    return '\n\n'.join(lines)

def extract_answer(text: str) -> str:
    """Extract only the final answer after 'Answer:' marker."""
    if not text:
        return ""
    
    # Find the last occurrence of "Answer:"
    if "Answer:" in text:
        answer_part = text.split("Answer:")[-1].strip()
        return answer_part
    
    # If no Answer: marker, return empty (nothing to show)
    return ""

async def initialize_pipelines():
    """Initialize all pipelines and return the workflow."""
    if st.session_state.pipelines_initialized:
        return st.session_state.workflow

    with st.spinner("Initializing pipelines..."):
        # Initialize Vector Pipeline
        try:
            vec_config = vector_pipeline.ConfigManager()
            st.session_state.vec_models = vector_pipeline.ModelProvider(vec_config)
            vec_ingestor = vector_pipeline.DocumentIngestor(st.session_state.vec_models)
            st.session_state.vec_pipe = vector_pipeline.RAGPipeline(vec_config, st.session_state.vec_models)
            
            new_file_paths = vec_ingestor.get_new_file_paths(vec_config.data_directory)
            documents = []
            if new_file_paths:
                documents = vec_ingestor.ingest_pdfs_from_directory(
                    vec_config.data_directory,
                    specific_files=new_file_paths
                )
            st.session_state.vec_pipe.setup_pipeline(documents)
        except Exception as e:
            st.error(f"Vector Pipeline Failed: {e}")
            return None

        # Initialize SQL Pipeline
        try:
            sql_config = sql_pipeline.ConfigManager(env_path=found_dotenv)
            db_manager = sql_pipeline.DatabaseManager(sql_config)
            engine, tables = db_manager.populate_database()
            st.session_state.sql_pipe = sql_pipeline.AdvancedQueryEngine(engine, tables, sql_config)
        except Exception as e:
            st.error(f"SQL Pipeline Failed: {e}")
            return None

        # Initialize Graph Pipeline
        try:
            graph_config = graph_pipeline.ConfigManager(dotenv_path=found_dotenv)
            graph_models = graph_pipeline.ModelRegistry(graph_config)
            st.session_state.graph_pipe = graph_pipeline.GraphQueryPipeline(graph_config, graph_models)
        except Exception as e:
            st.error(f"Graph Pipeline Failed: {e}")
            return None

        # --- TOOLS DEFINITION ---
        
        # Capture objects in local variables for safety
        vec_pipe = st.session_state.vec_pipe
        sql_pipe = st.session_state.sql_pipe
        graph_pipe = st.session_state.graph_pipe

        # Vector Tool (SPF)
        def search_spf_data(query: str) -> str:
            if not query: return "Error: Query cannot be empty."
            clean_query = query.strip().strip('"').strip("'")
            return str(vec_pipe.query_engine.query(clean_query))

        # SQL Tool (SPS)
        def search_sps_data(query: str) -> str:
            # Direct access to sql_pipe, no st.session_state needed
            clean_query = query.strip().strip('"').strip("'")
            if "```" in clean_query:
                clean_query = clean_query.replace("```", "")
            if "###" in clean_query:
                clean_query = clean_query.split("###").strip()
            return str(sql_pipe.query(clean_query))

        # Graph Tool (HTX)
        def search_htx_data(query: str = "") -> str:  # Add default value = ""
            if not query: return "Error: Query was empty."
            clean_query = query.strip().strip('"').strip("'")
            return str(graph_pipe.query(clean_query))

        # Create Tool Objects
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

        sql_tool = FunctionTool.from_defaults(
            fn=search_sps_data,
            name="sps_data",
            description="Primary database for Singapore Prison Service (SPS) statistical data (2006-2020). "
            "Use this tool for quantitative queries regarding 'Convicted Penal Population' "
            "broken down by 'Year', 'Age Group' (e.g., Below 21, 21-30, 60 Above), and 'Gender'. "
            "It contains structured annual tables suitable for aggregation and trend analysis. "
            "ALWAYS formulate a precise query that specifies the Year, Gender, and Age Group filters immediately "
            "(e.g., 'Total male inmates aged 31-40 in 2015'). "
            "Trust the returned figures as the final official statistics."
            )

        graph_tool = FunctionTool.from_defaults(
            fn=search_htx_data,
            name="htx_data",
            description="Primary database for Home Team Science & Technology Agency (HTX) knowledge, built as a knowledge graph from the FY2023 Annual Report." 
            "Use this tool for queries about science and technology capabilities, operational projects, and innovation initiatives across the Home Team" 
            "(e.g., robotics, biometrics, cybersecurity, CBRNE, XR/VR training systems)." 
            "It is best suited for questions on specific HTX projects (such as Rover-X, Marine Video Analytics for rescue, autonomous robots, or deepfake detection), strategic partnerships, and how technologies are deployed to support SPF, SCDF, ICA, SPS, and other Home Team departments. ALWAYS phrase queries as concrete questions about a particular capability, project, or domain (e.g., 'What technologies does HTX use to counter hostile drones?')."
            )


        # Setup custom tokenizer
        model_name = st.session_state.vec_models.llm.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def custom_messages_to_prompt(messages):
            conversation = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

        st.session_state.vec_models.llm.messages_to_prompt = custom_messages_to_prompt

        # Initialize workflow with local tools
        workflow = AgentWorkflow.from_tools_or_functions(
            [vector_tool, sql_tool, graph_tool],
            llm=st.session_state.vec_models.llm,
            system_prompt=(
                "You are the Chief Data Orchestrator for the Singapore Home Team Agentic RAG pipeline. "
                "You manage access to three distinct databases: SPF Vector Store, SPS SQL Database, and HTX Graph Database. "
                "Analyze the user's input and route to the correct tool. Output ONLY the tool call."
            )
        )

        st.session_state.workflow = workflow
        st.session_state.pipelines_initialized = True
        
        return st.session_state.workflow


async def run_agent_with_streaming(query: str, placeholder):
    """Run the agent and stream updates to the placeholder."""
    workflow = await initialize_pipelines()
    if not workflow:
        placeholder.error("Error: Failed to initialize pipelines.")
        return "Error: Failed to initialize pipelines."
    
    # Start the agent run
    handler = workflow.run(user_msg=query)
    
    final_response = ""
    thinking_log = []
    
    # Stream events
    async for event in handler.stream_events():
        try:
            if isinstance(event, AgentInput):
                msg = f"🔵 [STEP] Agent '{event.current_agent_name}' is acting..."
                thinking_log.append(msg)
                msgs = getattr(event, 'input', [])
                if msgs and isinstance(msgs, list):
                    input_msg = f"   Input: {msgs[-1].content}"
                    thinking_log.append(input_msg)
                    placeholder.markdown("\n".join(thinking_log))
            
            elif isinstance(event, ToolCall):
                msg = f"\n🛠️  [TOOL CALL] {event.tool_name}"
                thinking_log.append(msg)
                args_msg = f"   Arguments: {event.tool_kwargs}"
                thinking_log.append(args_msg)
                placeholder.markdown("\n".join(thinking_log))
            
            elif isinstance(event, ToolCallResult):
                msg = f"\n📦 [TOOL RESULT] {event.tool_name}"
                thinking_log.append(msg)
                output = str(event.tool_output)
                output_msg = f"   Output: {output[:200]}..." if len(output) > 200 else f"   Output: {output}"
                thinking_log.append(output_msg)
                placeholder.markdown("\n".join(thinking_log))
            
            elif isinstance(event, AgentStream):
                final_response += event.delta
                formatted_thinking = format_thinking(final_response)
                # Only show formatted chain-of-thought in the status box
                full_content = "\n".join(thinking_log)
                if formatted_thinking:
                    full_content += "\n\n📝 **Agent Reasoning:**\n" + formatted_thinking
                
                placeholder.markdown(full_content)
                            
        except Exception as e:
            pass
    
    return final_response

def main():
    st.set_page_config(
        page_title="Home Team Agentic RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Home Team Agentic RAG Pipeline")
    st.markdown("Query the Singapore Home Team databases: SPF, SPS, and HTX")

    if 'pipelines_initialized' not in st.session_state or not st.session_state.pipelines_initialized:
        with st.status("🚀 System Startup: Loading Models & Ingesting Data...", expanded=True) as status:
            st.write("Initializing Vector, SQL, and Graph pipelines...")
            asyncio.run(initialize_pipelines())
            status.update(label="✅ System Ready!", state="complete", expanded=False)
    
    # Text input for user query
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., How many inmates had 'No Education' in 2022?",
        key="user_query"
    )
    
    # Run button
    if st.button("Run", type="primary"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            # Create status container for agent thinking
            with st.status("Agent Thinking...", expanded=True) as status:
                # Create a placeholder inside the status for streaming updates
                thinking_placeholder = st.empty()
                
                # Run the agent query with streaming
                result = asyncio.run(run_agent_with_streaming(user_query, thinking_placeholder))
            
            # Display final result in a markdown box
            st.markdown("### Final Result")
            answer_only = extract_answer(result)

            if answer_only:
                st.markdown(answer_only)
            else:
                st.info("No result returned.")

if __name__ == "__main__":
    main()

