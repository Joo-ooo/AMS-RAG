import streamlit as st
import asyncio
import sys
import os
import re
import hashlib
import psycopg2
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer

from llama_index.core.agent.workflow import (
    AgentWorkflow, 
    AgentInput, 
    ToolCall, 
    ToolCallResult, 
    AgentStream,
    AgentOutput
)
from llama_index.core.tools import FunctionTool

# Load .env file
found_dotenv = find_dotenv()
if found_dotenv:
    load_dotenv(found_dotenv)
else:
    st.warning("No .env file found. Please check your configuration.")

# Function to add paths from environment variables
def add_path_from_env(env_var_name):
    path = os.getenv(env_var_name)
    if path and os.path.exists(path):
        if path not in sys.path:
            sys.path.append(path)
            print(f"Added to path: {path}")
    else:
        st.error(f"Warning: Path for {env_var_name} not found or invalid: {path}")

# Add the directories BEFORE importing the modules
add_path_from_env("VECTOR_PIPELINE_DIR")
add_path_from_env("SQL_PIPELINE_DIR")
add_path_from_env("GRAPH_PIPELINE_DIR")

# Import pipeline modules
import vector_pipeline
import sql_pipeline
import graph_pipeline

# Authentication & Database Connection
def get_auth_connection():
    """Connects to the separate AUTH database."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_AUTH_DB")
    )

def verify_login(email, password):
    """Verifies credentials against the auth_db."""
    try:
        conn = get_auth_connection()
        cur = conn.cursor()
        
        # Query user
        cur.execute("SELECT password_hash, rank, full_name FROM users WHERE email = %s", (email,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            stored_hash, rank, name = result
            # Hash input password to compare (SHA256)
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            if input_hash == stored_hash:
                return {"email": email, "rank": rank, "name": name}
    except Exception as e:
        st.error(f"Login Database Error: {e}")
        return None
    return None

# RBAC Logic
class RBACManager:
    def __init__(self):
        # Define which domains can access which tools
        self.domain_permissions = {
            "spf_data": ["spf.gov.sg", "mha.gov.sg"],
            "sps_data": ["sps.gov.sg", "mha.gov.sg"],
            "htx_data": ["htx.gov.sg", "mha.gov.sg"]
        }
        # Ranks that bypass all restrictions
        self.super_ranks = ["Director", "System Admin", "Commander", "CEO", "CTO", "Senior Officer"]

    def get_allowed_tools(self, email: str, rank: str, all_available_tools: list) -> list:
        """Filters tools based on user identity."""
        # 1. Super User Bypass
        if rank in self.super_ranks:
            return all_available_tools

        allowed_tools = []
        user_domain = email.split('@')[-1].lower() if '@' in email else ""

        for tool in all_available_tools:
            tool_name = tool.metadata.name

            # Domain Check
            allowed_domains = self.domain_permissions.get(tool_name, [])
            if user_domain in allowed_domains:
                allowed_tools.append(tool)
        
        return allowed_tools
    
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
    
    # Return empty if no answer found
    return ""

@st.cache_resource(show_spinner="🚀 System Boot: Loading AI Models...")
def load_global_resources():
    """
    Loads models ONCE for the entire server process.
    Returns a dictionary of loaded resources.
    """
    resources = {}

    # found_dotenv = find_dotenv() 

    try:
        # Vector Pipeline
        vec_config = vector_pipeline.ConfigManager(env_path=found_dotenv)
        vec_models = vector_pipeline.ModelProvider(vec_config)
        vec_ingestor = vector_pipeline.DocumentIngestor(vec_models)
        vec_pipe = vector_pipeline.RAGPipeline(vec_config, vec_models)
        
        # Ingest/Load Index
        new_files = vec_ingestor.get_new_file_paths(vec_config.data_directory)
        docs = vec_ingestor.ingest_pdfs_from_directory(vec_config.data_directory, specific_files=new_files) if new_files else []
        vec_pipe.setup_pipeline(docs)
        
        resources['vec_pipe'] = vec_pipe
        resources['vec_models'] = vec_models

        # SQL Pipeline
        sql_config = sql_pipeline.ConfigManager(env_path=found_dotenv)
        db_manager = sql_pipeline.DatabaseManager(sql_config)
        engine, tables = db_manager.populate_database()
        resources['sql_pipe'] = sql_pipeline.AdvancedQueryEngine(engine, tables, sql_config)

        # Graph Pipeline
        graph_config = graph_pipeline.ConfigManager(dotenv_path=found_dotenv)
        graph_models = graph_pipeline.ModelRegistry(graph_config)
        resources['graph_pipe'] = graph_pipeline.GraphQueryPipeline(graph_config, graph_models)

        return resources
        
    except Exception as e:
        st.error(f"Critical System Error: {e}")
        return None
    
# Tool Wrappers
def get_all_tools():
    """Define tools using the initialized pipelines."""
    vec_pipe = st.session_state.vec_pipe
    sql_pipe = st.session_state.sql_pipe
    graph_pipe = st.session_state.graph_pipe

    def search_spf_data(query: str) -> str:
        """Queries SPF Vector Store (Crime Stats)."""
        if not query: return "Error: Empty query."
        return str(vec_pipe.query_engine.query(query))

    def search_sps_data(query: str) -> str:
        """Queries SPS SQL Database (Prisoner Stats)."""
        clean_query = query.replace("```", "").replace("sql", "").strip()
        return str(sql_pipe.query(clean_query))

    def search_htx_data(query: str) -> str:
        """Queries HTX Graph Database (Tech & Innovation)."""
        return str(graph_pipe.query(query))

    def calculate_percentage_change(old_value: float, new_value: float) -> str:
        if old_value == 0: return "Cannot calculate percentage change from zero."
        change = ((new_value - old_value) / old_value) * 100
        return f"{change:.2f}%"

    return [
        FunctionTool.from_defaults(
            fn=search_spf_data, 
            name="spf_data", 
            description="Primary database for Singapore Police Force (SPF) reports, specifically "
            "'Annual Crime Briefs' and 'Annual Scams and Cybercrime Briefs' (2020-2024). "
            "Use this tool for queries regarding Singapore crime statistics, scam trends "
            "(e.g., job, e-commerce, phishing, investment, fake friend calls), cybercrime data, "
            "financial amounts lost, victim demographics, and police enforcement actions. "
            "It contains detailed tables, year-on-year comparisons, and specific figures. "
            "ALWAYS query for the full specific answer first (e.g., 'total scam losses in 2023')."),
        FunctionTool.from_defaults(
            fn=search_sps_data, 
            name="sps_data", 
            description="Primary database for Singapore Prison Service (SPS) statistical data (2006-2020). "
            "Use this tool for quantitative queries regarding 'Convicted Penal Population' "
            "broken down by 'Year', 'Age Group' (e.g., Below 21, 21-30, 60 Above), and 'Gender'. "
            "It contains structured annual tables suitable for aggregation and trend analysis. "
            "ALWAYS formulate a precise query that specifies the Year, Gender, and Age Group filters immediately "
            "(e.g., 'Total male inmates aged 31-40 in 2015'). "
            "Trust the returned figures as the final official statistics."),
        FunctionTool.from_defaults(
            fn=search_htx_data, 
            name="htx_data", 
            description="Primary database for Home Team Science & Technology Agency (HTX) knowledge, built as a knowledge graph from the FY2023 Annual Report."
            "Use this tool for queries about science and technology capabilities, operational projects, and innovation initiatives across the Home Team"
            "(e.g., robotics, biometrics, cybersecurity, CBRNE, XR/VR training systems)."
            "It is best suited for questions on specific HTX projects (such as Rover-X, Marine Video Analytics for rescue, autonomous robots, or deepfake detection), strategic partnerships, and how technologies are deployed to support SPF, SCDF, ICA, SPS, and other Home Team departments. ALWAYS phrase queries as concrete questions about a particular capability, project, or domain (e.g., 'What technologies does HTX use to counter hostile drones?')."
            )
    ]

async def run_agent_with_rbac(user_query, user_email, user_rank, placeholder):
    """Run agent with tools filtered by user credentials."""
    
    # Get All Tools
    all_tools = get_all_tools()
    
    # Filter Tools (RBAC)
    rbac = RBACManager()
    user_tools = rbac.get_allowed_tools(user_email, user_rank, all_tools)
    
    # Debug: Show active tools in UI (Optional, good for demo)
    active_tool_names = [t.metadata.name for t in user_tools]
    st.toast(f"🔓 Active Tools for {user_rank}: {active_tool_names}")

    if not user_tools:
        return "⛔ Access Denied: You do not have permission to access any available datastores."

    # Create Workflow
    workflow = AgentWorkflow.from_tools_or_functions(
        user_tools,
        llm=st.session_state.vec_models.llm,
        system_prompt=(
            f"You are the Home Team Assistant for a {user_rank}. "
            f"Your access is limited to: {active_tool_names}. "
            "If asked for data you cannot access, state that you lack permission."
        )
    )

    # Run & Stream
    handler = workflow.run(user_msg=user_query)
    final_response = ""
    thinking_log = []

    placeholder = st.status("🧠 Agent Thinking...", expanded=True)

    try:
        async for event in handler.stream_events():
            try:
                if isinstance(event, AgentInput):
                    thinking_log.append("🔵 [Thinking] Agent is processing...")
                    placeholder.markdown("\n\n".join(thinking_log))
                
                elif isinstance(event, ToolCall):
                    thinking_log.append(f"🛠️ [Tool Call] {event.tool_name} (Args: {event.tool_kwargs})")
                    placeholder.markdown("\n\n".join(thinking_log))
                
                elif isinstance(event, ToolCallResult):
                    output = str(event.tool_output)
                    preview = output[:100] + "..." if len(output) > 100 else output
                    thinking_log.append(f"📦 [Tool Result] {preview}")
                    placeholder.markdown("\n\n".join(thinking_log))
                    
                elif isinstance(event, AgentStream):
                    final_response += event.delta
                    
                elif isinstance(event, AgentOutput):
                    # Safely extract text from the LlamaIndex AgentOutput object
                    if hasattr(event, "response") and hasattr(event.response, "content"):
                        final_response = str(event.response.content)
                    else:
                        final_response = str(event)
                        
            except Exception as e:
                thinking_log.append(f"⚠️ [Parse Error] {str(e)}")
                placeholder.markdown("\n\n".join(thinking_log))

    except Exception as stream_err:
        thinking_log.append(f"❌ [Stream Error] {str(stream_err)}")
        placeholder.markdown("\n\n".join(thinking_log))

    if not final_response:
        try:
            result = await handler
            if hasattr(result, "response") and hasattr(result.response, "content"):
                final_response = str(result.response.content)
            else:
                final_response = str(result)
        except Exception as e:
            thinking_log.append(f"❌ [Await Error] {str(e)}")
            placeholder.markdown("\n\n".join(thinking_log))

    if not final_response.strip():
        final_response = "⚠️ No output generated by the LLM. Please check vLLM logs."

    
    # Clean Up Final Output
    
    # Split by "Answer:" if it exists
    if "Answer:" in final_response:
        parts = final_response.split("Answer:")
        # Take the part BEFORE "Answer:" for the thoughts section
        hidden_thoughts = parts[0].strip()
        if hidden_thoughts:
            thinking_log.append(f"💭 **Reasoning:** {hidden_thoughts}")
            placeholder.markdown("\n\n".join(thinking_log))

        # Take the part after "Answer:"
        clean_answer = parts[-1].strip()
    else:
        # Fallback: if no "Answer:" tag found, remove "Thought:" lines manually 
        clean_answer = re.sub(r'Thought:.*?(?=\n|$)', '', final_response).strip()
        clean_answer = re.sub(r'Action:.*?(?=\n|$)', '', clean_answer).strip()

    placeholder.update(label="✅ Processing Complete", state="complete", expanded=False)

    return clean_answer


def main():
    st.set_page_config(page_title="Home Team Agent", layout="wide")

    global_resources = load_global_resources()

    if not global_resources:
        st.stop() # Stop if pipelines failed to load

    # Unpack resources for easy access in your tool definitions
    st.session_state.vec_pipe = global_resources['vec_pipe']
    st.session_state.vec_models = global_resources['vec_models']
    st.session_state.sql_pipe = global_resources['sql_pipe']
    st.session_state.graph_pipe = global_resources['graph_pipe']

    # --- AUTHENTICATION ---
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        # Login UI
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 Home Team Data Agent")
            st.info("System is ready. Please log in.")
            
            with st.form("login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = verify_login(email, password)
                    if user:
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
    else:
        # Main App
        user = st.session_state['user']

        # Sidebar
        with st.sidebar:
            st.success(f"👤 **{user['name']}**")
            st.info(f"🏷️ Rank: **{user['rank']}**")
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                # Uncomment the below line to not reset the pipelines so next login is fast
                # st.session_state['pipelines_initialized'] = False 
                st.rerun()

        # Chat Interface
        st.title("🤖 Home Team Unified Agent")
        st.caption(f"Welcome, {user['name']}. Your access level determines which databases (SPF, SPS, HTX) I can search.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User Input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Agent Response
            with st.chat_message("assistant"):
                status_container = st.status("Thinking...", expanded=True)
                with status_container:
                    placeholder = st.empty()
                    response_text = asyncio.run(run_agent_with_rbac(
                        prompt, 
                        user['email'], 
                        user['rank'], 
                        placeholder
                    ))
                    status_container.update(label="Response Ready", state="complete", expanded=False)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()

