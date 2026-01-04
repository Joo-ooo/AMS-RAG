import os
import re
import pandas as pd
import torch
from typing import Any, Dict, List
import hashlib
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, inspect
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList


from llama_index.core import (
    VectorStoreIndex,
    SQLDatabase,
    Settings,
    PromptTemplate,
    StorageContext,
)
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode



class ConfigManager:
    """Manages environment variables and file paths."""
    def __init__(self, env_path: str = None):
        dotenv_path = env_path if env_path else find_dotenv()
        if not dotenv_path:
            raise FileNotFoundError("Could not find a .env file.")
        load_dotenv(dotenv_path=dotenv_path)
        
        self.project_root = os.path.dirname(dotenv_path)
        self.data_directory = self._get_path('SQL_DATASET_DIR')
        self.benchmark_filepath = self._get_path('SQL_BENCHMARK_DATASET_DIR')

        self.model_path = self._get_path("LLAMA_3.1_8B_INSTRUCT_DIR")
        self.embedding_model_path = self._get_path("MULTILINGUAL_E5_LARGE_DIR")
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # PostgreSQL Settings
        self.pg_user = os.getenv("POSTGRES_USER")
        self.pg_password = os.getenv("POSTGRES_PASSWORD")
        self.pg_host = os.getenv("POSTGRES_HOST")
        self.pg_port = os.getenv("POSTGRES_PORT")
        self.pg_db = os.getenv("POSTGRES_DB")

        self._verify_paths_and_keys()

    # Construct the database connection URL
    def get_database_url(self):
        return f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

    def _load_environment_variables(self):
        if not os.path.exists(self.dotenv_path):
            raise FileNotFoundError(f"CRITICAL ERROR: .env file NOT FOUND at: {self.dotenv_path}")
        load_dotenv(dotenv_path=self.dotenv_path)
        print(f"✅ .env file loaded from: {self.dotenv_path}")

    def _get_path(self, env_var):
        relative_path = os.getenv(env_var)
        if not relative_path:
            raise ValueError(f"ERROR: '{env_var}' not found in your .env file.")
        return os.path.join(self.project_root, relative_path)

    def _verify_paths_and_keys(self):
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory not found at: {self.data_directory}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Llama 3.1 model not found at: {self.model_path}")
        print(f"✅ Project root set to: {self.project_root}")
        print(f"📁 Data directory set to: {self.data_directory}")
        print(f"🤖 Model path set to: {self.model_path}")

class DatabaseManager:
    """
    Handles the creation and population of the in-memory SQLite database.
    """
    def __init__(self, config):
        self.config = config
        self.data_directory = config.data_directory
        # Create PostgreSQL engine
        self.engine = create_engine(self.config.get_database_url())

    def sanitize_table_name(self, filename):
        name = os.path.splitext(filename)[0].lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)

        # Check if filename exceeds PostgreSQL limit (63 characters)
        if len(name) > 63:
            hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
            prefix = name[:54]
            name = f"{prefix}_{hash_suffix}"
        return name
    

    def populate_database(self):
        print(f"\nSearching for CSV files in '{self.data_directory}'...")

        # Get list of existing tables in the database
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        print(f"Found {len(existing_tables)} existing tables in database.")

        all_table_names = []
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".csv"):
                table_name = self.sanitize_table_name(filename)
                all_table_names.append(table_name)

                # Check if table already exists
                if table_name in existing_tables:
                    print(f"  - Table '{table_name}' already exists. Skipping ingestion of '{filename}'.")
                    continue

                file_path = os.path.join(self.data_directory, filename)
                try:
                    df = pd.read_csv(file_path)
                    df.to_sql(table_name, self.engine, index=False, if_exists="fail")
                    print(f"  - Ingested '{filename}' into table '{table_name}'")
                except Exception as e:
                    print(f"  - Failed to ingest '{filename}'. Error: {e}")
                    
        print(f"Postgre SQL database populated with {len(all_table_names)} table(s).")
        return self.engine, all_table_names

class AdvancedQueryEngine:
    """
    Sets up and manages the LlamaIndex SQL query engine with table retrieval only.
    """
    def __init__(self, engine, all_table_names, config):
        self.engine = engine
        self.all_table_names = all_table_names
        self.config = config
        self.sql_database = SQLDatabase(self.engine, include_tables=self.all_table_names)
        self._setup_settings()
        
        # Create only table retrieval engine
        self.query_engine = self._create_table_retrieval_engine()
    
    def _setup_settings(self):
        """Configure LLM and embedding models."""
        print("\nSetting up LLM and Embedding models...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        stop_token_ids = []
    
        # Iterate through vocab to find semicolon variants (runs once at startup)
        vocab = tokenizer.get_vocab()
        for token, id in vocab.items():
            # Decode handles special chars like Ġ in some tokenizers, or raw bytes in Llama 3
            if ";" in tokenizer.decode([id]):
                stop_token_ids.append(id)
                
        # Ensure the standard EOS token is also included
        stop_token_ids.append(tokenizer.eos_token_id)
        stop_token_ids = list(set(stop_token_ids)) # Remove duplicates
        
        print(f"Found {len(stop_token_ids)} stop tokens containing ';'")

        Settings.llm = HuggingFaceLLM(
            max_new_tokens=256,
            tokenizer_name=self.config.model_path,
            model_name=self.config.model_path,
            device_map="auto",
            model_kwargs={"token": self.config.hf_token, "dtype": torch.bfloat16},
            
            stopping_ids=stop_token_ids, 
            
            generate_kwargs={
                "temperature": 0.1,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id
            }
        )
        
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config.embedding_model_path)
        print("✅ Models loaded.")
    
    def _create_table_retrieval_engine(self):
        """
        Creates a query engine for table-level retrieval only.
        """
        print("\n✅ Creating table retrieval engine...")

        CUSTOM_TEXT_TO_SQL_PROMPT = PromptTemplate(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "### CONTEXT (C)\n"
        "You are a specialist data analyst working with a prison inmate database. "
        "You have access to a specific database schema and your job is to extract accurate data based on user queries.\n\n"

        "### OBJECTIVE (O)\n"
        "Generate a single, syntactically correct {dialect} SQL query to answer the user's question. "
        "CRITICAL: Your output MUST STOP immediately after the SQL query ends with a semicolon (;).\n"
        "Do NOT generate 'SQLResult', 'Answer', or any hypothetical text.\n\n"

        "### CRITICAL RULES\n"
        "1. FILTERING: You must accurately filter data based on the user's specific criteria (Year, Gender, etc.).\n"
        "2. GENDER: If the user asks for 'male', filter by population_by_gender = 'Male'. If 'female', filter by 'Female'.\n"
        "3. YEAR: Always use the EXACT year specified in the 'USER INPUT' question below. Do NOT copy years from the examples.\n"
        "4. NO HALLUCINATION: Do not invent table names. Use ONLY the schema provided.\n\n"

        "### EXAMPLES (Syntax Reference Only)\n"
        "Question: How many male inmates were there in 2015?\n"
        "SQLQuery: SELECT number_of_population FROM convicted_penal_population_by_gender WHERE year = 2015 AND population_by_gender = 'Male';\n\n"

        "Question: Count the number of female prisoners in 2019?\n"
        "SQLQuery: SELECT number_of_population FROM convicted_penal_population_by_gender WHERE year = 2019 AND population_by_gender = 'Female';\n\n"

        "### STYLE (S)\n"
        "Write efficient, standard SQL. Select only necessary columns.\n\n"

        "### AUDIENCE (A)\n"
        "A SQL Database Engine. It expects RAW SQL code only.\n\n"

        "### RESPONSE (R)\n"
        "Output ONLY the SQL query string starting with SELECT and ending with a semicolon (;).\n"
        "Do NOT include 'SQLResult:' or 'Answer:'.\n\n"

        "### DATA SCHEMA\n"
        "Only use the tables and columns listed here:\n"
        "{schema}\n"
        "<|eot_id|>\n"
        
        # ---------------- USER MESSAGE (DYNAMIC QUERY) ----------------
        "<|start_header_id|>user<|end_header_id|>\n"
        "### USER INPUT\n"
        "Question: {query_str}\n"
        "<|eot_id|>\n"
        
        # ---------------- ASSISTANT MESSAGE (PRIMING) ----------------
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "SQLQuery: "
    )
        
        # Create table schema retriever
        table_node_mapping = SQLTableNodeMapping(self.sql_database)
        table_schema_objs = [SQLTableSchema(table_name=t) for t in self.all_table_names]
        obj_index = ObjectIndex.from_objects(
            table_schema_objs, 
            table_node_mapping, 
            VectorStoreIndex
        )
        obj_retriever = obj_index.as_retriever(similarity_top_k=3)
        
        # Create query engine
        return SQLTableRetrieverQueryEngine(
            self.sql_database,
            obj_retriever,
            text_to_sql_prompt=CUSTOM_TEXT_TO_SQL_PROMPT,
            return_direct=True
        )
    
    def query(self, query_str):
        """
        Execute a query using table retrieval.
        
        Args:
            query_str: The natural language query
        
        Returns:
            Query response with metadata
        """
        return self.query_engine.query(query_str)