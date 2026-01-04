from __future__ import annotations

import os
import gc
import re
from typing import List, Optional, Any, Dict

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase

from llama_index.core import Document, PropertyGraphIndex, PromptTemplate, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, Node, QueryBundle
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from transformers import AutoTokenizer

# ---------------------------
# Configuration Manager
# ---------------------------

class ConfigManager:
    """Manages configuration and initializes the Neo4j graph store connection."""
    def __init__(self, dotenv_path: str):
        load_dotenv(dotenv_path=dotenv_path)
        self.project_root = os.path.dirname(dotenv_path)
        self.data_directory = os.path.join(self.project_root, os.getenv("GRAPH_DATASET_DIR", ""))
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_model_dir = os.getenv("LLAMA_3.1_8B_INSTRUCT_DIR", "")
        self.reranker_model_dir = os.getenv("BGE_RERANKER_V2_M_DIR", "")
        self.neo4j_uri = os.getenv("NEO4J_URI", "")
        self.neo4j_user = os.getenv("NEO4J_DATABASE", "")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        self.openai_model = "gpt-4o"
        self.rerank_top_n = 5
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials not found in .env file")
        self._graph_store = Neo4jPropertyGraphStore(
            username=self.neo4j_user, password=self.neo4j_password, url=self.neo4j_uri,
        )
        self._driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

    def get_store(self) -> Neo4jPropertyGraphStore:
        return self._graph_store

    def query_graph(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Run an arbitrary Cypher query against the graph."""
        with self._driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

# --------------
# Model Registry
# --------------

class ModelRegistry:
    def __init__(self, config: ConfigManager):
        self.config = config
        self._hf_llm: Optional[HuggingFaceLLM] = None
        self._openai_llm: Optional[OpenAI] = None
        self._reranker: Optional[SentenceTransformerRerank] = None

    def load_hf_llm(self) -> HuggingFaceLLM:
        if self._hf_llm: return self._hf_llm
        model_path = self.config.llama_model_dir
        if not model_path or not os.path.exists(model_path): raise ValueError("LLAMA_3.1_8B_INSTRUCT_DIR not found")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._hf_llm = HuggingFaceLLM(
            model_name=model_path, tokenizer_name=model_path, device_map="auto",
            model_kwargs={"token": self.config.hf_token, "dtype": torch.bfloat16},
            generate_kwargs={"temperature": 0.1, "do_sample": True, "eos_token_id": tokenizer.convert_tokens_to_ids(";")},
        )
        return self._hf_llm

    def load_openai_llm(self) -> OpenAI:
        if self._openai_llm: return self._openai_llm
        if not self.config.openai_api_key: raise ValueError("OPENAI_API_KEY not set")
        self._openai_llm = OpenAI(api_key=self.config.openai_api_key, model=self.config.openai_model, temperature=0.0)
        return self._openai_llm

    def load_reranker(self) -> SentenceTransformerRerank:
        if self._reranker: return self._reranker
        if not self.config.reranker_model_dir: raise ValueError("Reranker model directory not set")
        self._reranker = SentenceTransformerRerank(model=self.config.reranker_model_dir, top_n=self.config.rerank_top_n)
        return self._reranker

# --------------------
# Ingestion Pipeline
# --------------------

class GraphIngestionPipeline:
    def __init__(self, config: ConfigManager, models: ModelRegistry):
        self.config = config
        self.models = models

    # Check for already ingested files to avoid duplicates
    def get_ingested_files(self) -> set:
        """Retrieves a set of source file paths that have already been ingested."""
        query = "MATCH (n) WHERE n.source IS NOT NULL RETURN DISTINCT n.source AS source"
        try:
            results = self.config.query_graph(query)
            return {record['source'] for record in results if record.get('source')}
        except Exception as e:
            print(f"Warning: Could not check existing files. Error: {e}")
            return set()
        
    def ingest_from_directory(self, directory_path: str, suffixes: tuple = (".txt",)):
        """
        Loads documents from a directory, checks if they are already ingested,
        and ingests only the new ones.
        """
        if not os.path.isdir(directory_path):
            print(f"Directory '{directory_path}' not found.")
            return

        print("Checking for existing documents...")
        ingested_files = self.get_ingested_files()
        
        docs_to_ingest = []
        
        for root, _, files in os.walk(directory_path):
            for f in files:
                if f.lower().endswith(suffixes):
                    fp = os.path.join(root, f)
                    
                    # Check if file is already ingested
                    if fp in ingested_files:
                        print(f"File {f} has been ingested.")
                        continue
                        
                    # Load text if not ingested
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                            text = fh.read()
                            docs_to_ingest.append(Document(text=text, metadata={"source": fp}))
                    except Exception as e:
                        print(f"Error reading file {fp}: {e}")

        gc.collect()

        if not docs_to_ingest:
            print("No new documents to ingest.")
            return

        print(f"Starting ingestion for {len(docs_to_ingest)} new documents...")
        self.build_index(docs_to_ingest)

    def merge_duplicate_nodes(self):
        """Merges nodes with the same 'name' property, taken from your notebook."""
        print("Merging duplicate nodes...")
        query = """
        MATCH (n:entity)
        WITH n.name AS name, COLLECT(n) AS nodes
        WHERE SIZE(nodes) > 1
        CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard'})
        YIELD node
        RETURN node
        """
        try:
            results = self.config.query_graph(query)
            print(f"Merged {len(results)} sets of duplicate nodes.")
        except Exception as e:
            print(f"Node merging failed. Ensure APOC plugin is installed on Neo4j. Error: {e}")

    def build_index(self, documents: List[Document], use_llm: str = "openai"):
        store = self.config.get_store()
        llm = self.models.load_openai_llm() if use_llm == "openai" else self.models.load_hf_llm()

        print("Checking for existing documents...")
        ingested_files = self.get_ingested_files()
        
        docs_to_ingest = []
        for doc in documents:
            source = doc.metadata.get("source")
            # Check if the file path exists in the graph's source properties
            if source and source in ingested_files:
                print(f"File {os.path.basename(source)} has been ingested.")
            else:
                docs_to_ingest.append(doc)
        
        if not docs_to_ingest:
            print("All documents have already been ingested. Skipping index build.")
            return

        print(f"Starting index build for {len(docs_to_ingest)} new documents...")
        
        PropertyGraphIndex.from_documents(
            documents=docs_to_ingest, 
            property_graph_store=store, 
            llm=llm, 
            show_progress=True
        )
        
        print("Index build complete.")
        self.merge_duplicate_nodes()

# --------------
# Query Pipeline
# --------------

def parse_cypher_query(text: str) -> str:
    """
    Robustly parses a Cypher query from LLM output.
    """
    # Find standard markdown code blocks
    match = re.search(r"``````", text, re.IGNORECASE | re.DOTALL)

    if match:
        text = match.group(1)
    else:
        # Fallback: Look for the start of a Cypher query
        # This handles cases where the LLM forgets the code blocks or malforms them
        cypher_keywords = ["MATCH", "CALL", "CREATE", "MERGE"]
        for keyword in cypher_keywords:
            keyword_match = re.search(fr"({keyword}\s+.*)", text, re.IGNORECASE | re.DOTALL)
            if keyword_match:
                text = keyword_match.group(1)
                break
    
    # Remove any trailing markdown code blocks (the source of your current error)
    # If the fallback captured "MATCH ... ```", this removes the "```
    text = re.sub(r"```.*$", "", text, flags=re.DOTALL)

    # Clean whitespace
    text = text.strip()

    # Clean surrounding quotes (e.g., if LLM returned "MATCH ...")
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    return text

class CustomTextToCypherRetriever(BaseRetriever):
    """Custom retriever that implements the logic from your notebook."""
    def __init__(self, config: ConfigManager, models: ModelRegistry, use_llm: str):
        self._config = config
        self._graph_store = config.get_store()
        self._llm = models.load_openai_llm() if use_llm == "openai" else models.load_hf_llm()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get dynamic schema and generate the query
        schema = self._graph_store.get_schema()

        few_shot_examples = """

        Examples:
        ---------------------
        Question: What is HTX's vision?
        Cypher Query: MATCH (g:Grouping {name: "Where We Want To Go"})-[:HAS_CONCEPT]->(c:Concept {name: "Vision"})-[:HAS_DESCRIPTION]->(d:Description) RETURN d.name AS description


        Question: What is HTX and who does it serve?
        Cypher Query: MATCH (c:Concept {name: "Mission"})-[:HAS_DESCRIPTION]->(d:Description) RETURN d.name AS mission


        Question: What does C4I mean in HTX's work?
        Cypher Query: MATCH (c:Concept {name: "C4I"})-[:HAS_DESCRIPTION]->(d:Description) RETURN d.name AS definition

        Question: What's featured at Home Team Festival?
        Cypher Query: MATCH (c:Concept)-[:HAS_DESCRIPTION]->(d:Description) WHERE toLower(c.name) CONTAINS toLower('what's') RETURN d.name AS description
        ---------------------
        """
        
        CYPHER_GENERATION_TEMPLATE = f"""
        You are an expert Neo4j developer.
        Your task is to translate a user's question into a valid Cypher query based on the provided schema.
        You must use ONLY the nodes, relationships, and properties in the schema.

        ---

        Schema:
        {schema}

        ---

        IMPORTANT INSTRUCTIONS:
        First, think step-by-step about how to answer the question.
        1.  Identify the key entities and values from the user's question.
        2.  Map these entities to the corresponding node labels and properties in the schema.
        3.  Identify the relationship or action being asked about.
        4.  Construct a path using the relationships from the schema that connects the identified nodes.
        5.  Finally, based on your plan, generate the Cypher query.

        ---

        Examples to guide you in formulating Cypher queries:
        {few_shot_examples}

        ---

        Question: {{question}}

        ---

        THOUGHT PROCESS:
        The user is asking for the definition of "C4I".
        1.  The key entity is "C4I".
        2.  I can map this to a node: `(:Concept {{name: 'C4I'}})`.
        3.  The user is asking for its "definition" or "description".
        4.  The schema shows that a `Concept` is linked to its `Description` via the path `(:Concept)-[:HAS_DESCRIPTION]->(:Description)`.
        5.  Therefore, I will start from the `Concept` node for "C4I", follow the `HAS_DESCRIPTION` relationship, and return the name of the `Description` node.

        ---

        Cypher Query:

        """

        cypher_prompt = PromptTemplate(CYPHER_GENERATION_TEMPLATE)
        response_obj = self._llm.predict(cypher_prompt, schema=schema, question=query_bundle.query_str)
        
        # Parse and execute the query
        generated_query = parse_cypher_query(str(response_obj))
        print(f"\nGenerated and Cleaned Cypher Query:\n{generated_query}")
        
        try:
            retrieved_data = self._config.query_graph(generated_query)
        except Exception as e:
            print(f"Error executing generated Cypher: {e}")
            return []

        # Format results into nodes
        nodes = [NodeWithScore(node=Node(text=str(d)), score=1.0) for d in retrieved_data]
        return nodes

class GraphQueryPipeline:
    def __init__(self, config: ConfigManager, models: ModelRegistry):
        self.config = config
        self.models = models

    def create_query_engine(self, use_llm: str = "openai") -> RetrieverQueryEngine:
        custom_retriever = CustomTextToCypherRetriever(self.config, self.models, use_llm)
        reranker = self.models.load_reranker()
        node_postprocessors = [reranker]

        response_synthesizer = get_response_synthesizer(
            llm=self.models.load_openai_llm(),
        )

        return RetrieverQueryEngine(
            retriever=custom_retriever,
            node_postprocessors=node_postprocessors,
            response_synthesizer=response_synthesizer,
        )

    def query(self, question: str, use_llm: str = "openai") -> Any:
        engine = self.create_query_engine(use_llm=use_llm)
        return engine.query(question)


# -------------------------
# Minimal ingestion helpers
# -------------------------

def load_text_documents_from_dir(path: str, suffixes: tuple = (".txt",)) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(path): return docs
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(suffixes):
                fp = os.path.join(root, f)
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                docs.append(Document(text=text, metadata={"source": fp}))
    gc.collect()
    return docs
