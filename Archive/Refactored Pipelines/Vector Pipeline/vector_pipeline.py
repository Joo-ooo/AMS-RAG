import os
import re
import gc
import glob
import json
from datetime import datetime
from pypdf import PdfReader
from docx import Document
from typing import List, Dict

import torch
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path
from dotenv import load_dotenv, find_dotenv
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
    Document,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank, FixedRecencyPostprocessor, SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from chromadb import HttpClient

class VerboseHyDE(HyDEQueryTransform):
    """
    A custom HyDE transformer that prints the generated hypothetical document
    to the console for debugging purposes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hyde_query = None

    def _run(self, query_bundle: QueryBundle, metadata=None) -> QueryBundle:
        new_bundle = super()._run(query_bundle, metadata)
        
        print(f"\n[HyDE Debug] Input: {query_bundle.query_str}")
        if new_bundle.custom_embedding_strs:
            self.last_hyde_query = new_bundle.custom_embedding_strs[0]
        else:
            self.last_hyde_query = "No HyDE query generated"
            
        print(f"[HyDE Debug] Stored Query: {self.last_hyde_query[:50]}...")
            
        return new_bundle

# --- HELPER: Metadata Extractor ---
def extract_creation_date(filepath: str) -> dict:
    """Extracts creation date from PDF/DOCX internal metadata."""
    meta = {"publication_date": "Unknown", "publication_year": "Unknown"}
    filename = os.path.basename(filepath)
    
    # Try to find a year in the filename (e.g., '2021', '2025')
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        year = int(year_match.group(1))
        meta["publication_year"] = year
        # Default to Jan 1st of that year for sorting purposes
        meta["publication_date"] = f"{year}-01-01" 
        return meta
    
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(filepath, strict=False)
            if reader.metadata and "/CreationDate" in reader.metadata:
                # Format: D:YYYYMMDD...
                raw = reader.metadata["/CreationDate"][2:10]
                dt = datetime.strptime(raw, "%Y%m%d")
                meta["publication_date"] = dt.strftime("%Y-%m-%d")
                meta["publication_year"] = dt.year
        elif ext == ".docx":
            doc = Document(filepath)
            if doc.core_properties.created:
                dt = doc.core_properties.created
                meta["publication_date"] = dt.strftime("%Y-%m-%d")
                meta["publication_year"] = dt.year
    except Exception as e:
        pass # Fail silently on metadata to keep ingestion running
    return meta

class ConfigManager:
    """Manages environment variables and file paths."""
    def __init__(self, env_path: str = None):
        dotenv_path = env_path if env_path else find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded .env from {dotenv_path}")
        else:
            print("Warning: .env file not found. Relying on environment variables.")
        
        self.project_root = os.path.dirname(dotenv_path)
        self.data_directory = self._get_path_from_env('VECTOR_DATASET_DIR')
        self.benchmark_filepath = self._get_path_from_env('VECTOR_BENCHMARK_DATASET_DIR')
        self.docstore_directory = self._get_path_from_env('DOCSTORE_DIR')

        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = int(os.getenv('CHROMA_PORT', 8000))

        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        print(f"Project root set to: {self.project_root}")
        print(f"Data directory set to: {self.data_directory}")

    def _get_path_from_env(self, env_var: str) -> str:
        relative_path = os.getenv(env_var)
        if not relative_path:
            raise ValueError(f"Environment variable '{env_var}' not set in .env file.")
        return os.path.join(self.project_root, relative_path)

class ModelProvider:
    """Initializes and provides access to all models."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.llm = self._init_llm()
        self.ocr_model, self.ocr_processor = self._init_ocr_model()
        self.embed_model = self._init_embedding_model()
        self.reranker = self._init_reranker()
        
        # Set global settings for LlamaIndex
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        print("Global LlamaIndex settings configured.")

    # --- Running inside docker container ---
    def _init_llm(self) -> OpenAILike:
        """Initializes the LLM by connecting to the vLLM server."""
        
        # Get the API Base URL from environment variables
        api_base = os.getenv("VLLM_API_BASE")
        if not api_base:
            raise ValueError("VLLM_API_BASE environment variable is not set.")

        model_name = "/models/Llama-3.1-8B-Instruct"

        # Initialize the OpenAILike client
        llm = OpenAILike(
            model=model_name,
            api_base=api_base,
            api_key="EMPTY",  # vLLM requires a dummy key
            is_chat_model=True,
            context_window=4096,
            temperature=0.1,
            max_tokens=512,
        )
        
        print(f"Connected to vLLM at {api_base} with model: {model_name}")
        return llm

    def _init_ocr_model(self):
        """Initializes the Nanonets OCR model from a local directory."""
        model_path = os.getenv("NANONETS_OCR_S_DIR")
        if not model_path or not os.path.isdir(model_path):
            print(f"Warning: OCR model directory not found at path: {model_path}. OCR will be disabled.")
            return None, None

        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.bfloat16
            )
            print(f"Nanonets OCR model loaded from: {model_path}")
            return model, processor
        except Exception as e:
            print(f"Failed to load Nanonets OCR model from {model_path}. Error: {e}")
            return None, None
            
    def _init_embedding_model(self) -> HuggingFaceEmbedding:
        """Initializes the sentence embedding model from a local directory."""
        model_path = os.getenv("MULTILINGUAL_E5_LARGE_DIR")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Embedding model directory not found at path: {model_path}")

        embed_model = HuggingFaceEmbedding(
            model_name=model_path, 
            token=self.config.hf_token,
            # device="cpu"
        )
        print(f"Embedding model loaded from: {model_path}")
        return embed_model

    def _init_reranker(self) -> SentenceTransformerRerank:
        """Initializes the reranker model from a local directory."""
        model_path = os.getenv("BGE_RERANKER_V2_M_DIR")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Reranker model directory not found at path: {model_path}")

        reranker = SentenceTransformerRerank(
            model=model_path, 
            top_n=5,
        )
        print(f"Reranker model loaded from: {model_path}")
        return reranker

class DocumentIngestor:
    """Handles PDF ingestion and OCR processing."""
    
    NANONETS_PROMPT = "Extract the text from the above document... Watermarks should be wrapped in brackets." # Truncated for brevity

    def __init__(self, models: ModelProvider):
        if not models.ocr_model or not models.ocr_processor:
            raise ValueError("OCR model is not available in ModelProvider.")
        self.models = models
        self.ocr_model = models.ocr_model
        self.ocr_processor = models.ocr_processor
        self.docstore_path = models.config.docstore_directory
    
    def get_new_file_paths(self, data_directory: str) -> List[str]:
        """
        Compares filenames in the directory against the persisted docstore metadata.
        Returns a list of file paths that need to be processed.
        """
        # Get all PDF paths from disk
        all_pdf_paths = glob.glob(os.path.join(data_directory, "*.pdf"))
        print(f"Found {len(all_pdf_paths)} PDFs in directory.")

        docstore_file = os.path.join(self.docstore_path, "docstore.json")
        
        # If no DB exists, everything is new
        if not os.path.exists(docstore_file):
            print("No existing docstore found. All files are new.")
            return all_pdf_paths

        try:
            # Load index
            storage_context = StorageContext.from_defaults(persist_dir=self.docstore_path)
            docs_dict = storage_context.docstore.docs
            
            # Collect set of EXISTING FILENAMES from metadata
            existing_filenames = set()
            for node in docs_dict.values():
                fname = node.metadata.get("file_name") or node.metadata.get("filename")
                if fname:
                    existing_filenames.add(fname)

        except Exception as e:
            print(f"Error loading existing index/docstore: {e}. Assuming all files new.")
            return all_pdf_paths

        # Filter: Keep file if its BASENAME is not in our known list
        new_paths = []
        for path in all_pdf_paths:
            filename = os.path.basename(path) # e.g. "annual_report.pdf"
            
            if filename not in existing_filenames:
                new_paths.append(path)
            else:
                print(f"Skipping {filename} (Already ingested)")
                pass
        
        return new_paths

    def _batch_ocr_on_pages(self, images: List, page_batch_size: int = 8) -> List[str]:
        """Processes a list of page images in batches using the OCR model."""
        all_texts = []
        for i in tqdm(range(0, len(images), page_batch_size), desc="OCR on Pages"):
            batch_images = images[i:i + page_batch_size]
            content = []
            for _ in batch_images:
                content.append({"type": "image"})
                content.append({"type": "text", "text": self.NANONETS_PROMPT})
            
            messages = [{"role": "user", "content": content}]
            text = self.ocr_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.ocr_processor(text=text, images=batch_images, padding=True, return_tensors="pt").to(self.ocr_model.device)
            
            output_ids = self.ocr_model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            generated_ids = [output_id[len(input_id):] for input_id, output_id in zip(inputs.input_ids, output_ids)]
            batch_outputs = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            all_texts.extend(batch_outputs)
            del batch_images, inputs, output_ids, generated_ids
            gc.collect()
        return all_texts

    def ingest_pdfs_from_directory(self, data_directory: str, specific_files: List[str] = None) -> List[Document]:
        """Ingests all PDFs from a directory, performs OCR, and returns Document objects."""

        # Check for files to process
        if specific_files is None:
            pdf_files = self.get_new_file_paths(data_directory) 
        else:
            pdf_files = specific_files

        if not pdf_files:
            print("✅ No new files to ingest.")
            return []


        print(f"Starting OCR/Ingestion for {len(pdf_files)} files...")

        page_batch_size = 8 
        all_documents = []

        for filepath in tqdm(pdf_files, desc="Processing Files"):
            
            if not os.path.isabs(filepath):
                 filepath = os.path.join(data_directory, os.path.basename(filepath))

            filename = os.path.basename(filepath)

            try:
                images_from_pdf = convert_from_path(filepath, dpi=150)
                page_texts = self._batch_ocr_on_pages(images_from_pdf, page_batch_size)
                full_text_content = "\n".join(page_texts)

                if full_text_content.strip():
                    abs_path = os.path.abspath(filepath)

                    # Extract date metadata
                    date_meta = extract_creation_date(filepath)
                
                    # Combine metadata
                    combined_metadata = {
                        "filepath": abs_path, 
                        "file_name": filename,
                        **date_meta # Merges publication_date and publication_year
                    }

                    document = Document(
                        text=full_text_content,
                        metadata=combined_metadata,
                        id_=abs_path,
                        text_template="Context Source:\n{metadata_str}\n----------------\n\n{content}"
                    )
                    # Ensure LLM is allowed to see these keys
                    document.excluded_llm_metadata_keys = ["filepath"] 
                    
                    all_documents.append(document)
                else:
                    print(f"- WARNING: No content was extracted from {filename}")
                
                del images_from_pdf, page_texts
                gc.collect()

            except Exception as e:
                print(f"- FAILED to process {filename}. Error: {e}")
                continue
        
        print(f"--- Ingestion complete. Successfully loaded {len(all_documents)} documents. ---")
        return all_documents


class DualPathEnsembleRetriever(BaseRetriever):
    """Custom retriever that runs BM25 and semantic retrieval in parallel and merges results."""
    def __init__(
            self, bm25_retriever: BM25Retriever, 
            vector_retriever: VectorIndexRetriever, 
            bm25_weight: float = 2.0):
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever
        self._bm25_weight = bm25_weight
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        bm25_hits = self._bm25_retriever.retrieve(query_bundle)
        vector_hits = self._vector_retriever.retrieve(query_bundle)

        for node in bm25_hits:
            node.score *= self._bm25_weight

        combined_hits: Dict[str, NodeWithScore] = {}
        for hit in bm25_hits + vector_hits:
            if hit.node.node_id not in combined_hits or hit.score > combined_hits[hit.node.node_id].score:
                combined_hits[hit.node.node_id] = hit
        
        sorted_hits = sorted(list(combined_hits.values()), key=lambda x: x.score, reverse=True)
        return sorted_hits

class RAGPipeline:
    """Orchestrates the entire RAG pipeline from ingestion to querying."""
    def __init__(self, config: ConfigManager, models: ModelProvider):
        self.config = config
        self.models = models
        self.nodes = None
        self.query_engine = None
        self.ingestor = DocumentIngestor(models)

    def setup_pipeline(self, documents: List[Document], chunk_size=2048, chunk_overlap=512):
        """Builds the index, retrievers, and query engine."""
        print("--- Setting up RAG pipeline ---")
        
        PERSIST_DIR = self.config.docstore_directory
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        db_client = HttpClient(host=self.config.chroma_host, port=self.config.chroma_port)
        chroma_collection = db_client.get_or_create_collection("vector_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # recency_processor = RecencyPostprocessor()
        
        vector_index = None

        if os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
            try:
                print(f"Loading existing Docstore/Index from {PERSIST_DIR}...")
                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
                vector_index = load_index_from_storage(storage_context)
                print("Index loaded successfully!")
                
                existing_filenames = set()
                for node in vector_index.docstore.docs.values():
                    print(node.metadata.get("file_name"), node.metadata.get("publication_date"))
                    fname = node.metadata.get("file_name") or node.metadata.get("filename")
                    if fname:
                        existing_filenames.add(fname)

                # Filter incoming documents by filename
                new_docs = []
                if documents:
                    for doc in documents:
                        # Assume doc.metadata['file_name'] was set by ingestor
                        doc_filename = doc.metadata.get("file_name")
                        if doc_filename and doc_filename not in existing_filenames:
                            new_docs.append(doc)
                        elif not doc_filename:
                            # If no filename metadata, default to adding it to be safe
                            new_docs.append(doc)
                
                # Update index if new docs are present
                if new_docs:
                    print(f"Adding {len(new_docs)} new documents to the existing index...")
                    new_nodes = node_parser.get_nodes_from_documents(new_docs)
                    vector_index.docstore.add_documents(new_nodes) # Add to docstore
                    vector_index.insert_nodes(new_nodes)          # Insert vectors
                    vector_index.storage_context.persist(persist_dir=PERSIST_DIR) # Persist changes
                    print("New documents added and index persisted.")
                else:
                    print("No new documents to add to the index.")
                    
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                vector_index = None

        if vector_index is None:
            print("Creating NEW index from scratch...")
            
            # Parse Documents into Nodes
            nodes = node_parser.get_nodes_from_documents(documents)
            
            # Create Storage Context (Connecting VectorStore + Local Docstore)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            storage_context.docstore.add_documents(nodes)
            
            # Create Index with the nodes explicitly
            vector_index = VectorStoreIndex(
                nodes, 
                storage_context=storage_context,
            )
            
            # Now that storage_context has the nodes, this will write them to docstore.json
            print("Persisting Docstore to local disk...")
            vector_index.storage_context.persist(persist_dir=PERSIST_DIR)

        print("Initializing Retrievers...")
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
        
        all_nodes = list(vector_index.docstore.docs.values())
        
        if all_nodes:
            print(f"Initializing BM25 with {len(all_nodes)} nodes...")
            bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=10)
            dual_retriever = DualPathEnsembleRetriever(bm25_retriever, vector_retriever)
        else:
            print("WARNING: Docstore is empty. Skipping BM25.")
            dual_retriever = vector_retriever

        print("Retriever strategy initialized.")

        # Response Synthesizer
        custom_prompt = PromptTemplate(
            "You are an expert drafting a formal response based on Singapore government official documents.\n"
            "Your goal is to provide a precise answer based ONLY on the provided context.\n\n"
            
            "### CRITICAL CONFLICT CHECK:\n"
            "1. You must analyze the context for conflicting information (e.g., Source A says 'X' but Source B says 'Y').\n"
            "2. IF A CONFLICT EXISTS: You must explicitly state: 'I am unable to answer this question due to conflicting contexts found in [Source A] and [Source B].'\n"
            "3. IF NO CONFLICT: Answer the question directly and concisely.\n\n"
            
            "### CONTEXT:\n"
            "{context_str}\n\n"
            
            "### QUESTION:\n"
            "{query_str}\n\n"
            
            "### ANSWER:\n"
        )
        response_synthesizer = get_response_synthesizer(text_qa_template=custom_prompt)

        reranker = self.models.reranker
        
        # Base Query Engine (Ensemble + Reranker + Recency)
        ensemble_query_engine = RetrieverQueryEngine(
            retriever=dual_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                reranker, # Rank by relevance
            ]
        )
        
        # Advanced Query Engine with HyDE
        hyde_prompt = PromptTemplate(
            "You are an expert drafting a formal response based on Singapore government official documents."
            "Write a concise, precise and factual passage that directly answers the user's question using the appropriate domain terminology.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Acronyms: Automatically expand or include standard Singapore government acronyms relevant to the topic "
            "(e.g., if the user asks about 'scams', include 'Anti-Scam Centre (ASC)'; if 'housing', use 'BTO'/'SHB'; if 'finance', use 'MAS'/'TRM').\n"
            "2. Specificity: Do not be vague. Invent plausible-sounding but specific details (like 'Annex A' or 'Section 4') to help vector search match the structure of real documents.\n\n"
            "Question: {context_str}\n"
            "Report Excerpt:"
        )
        self.hyde_transform = VerboseHyDE(llm=self.models.llm, include_original=True, hyde_prompt=hyde_prompt)
        
        # self.query_engine = TransformQueryEngine(
        #     ensemble_query_engine,
        #     query_transform=self.hyde_transform
        # )
        self.query_engine = ensemble_query_engine
        print("Advanced Query Engine with HyDE transform is ready.")

    def is_database_empty(self) -> bool:
        """Checks if the ChromaDB collection has any data."""
        try:
            print(f"Connecting to ChromaDB at {self.config.chroma_host}:{self.config.chroma_port}...")
            db_client = HttpClient(host=self.config.chroma_host, port=self.config.chroma_port)
            collection = db_client.get_or_create_collection("vector_collection")
            count = collection.count()
            print(f"Found {count} existing documents in ChromaDB.")
            return count == 0
        except Exception as e:
            print(f"Could not connect to ChromaDB to check status: {e}")
            return True # Assume empty if we can't connect

    def _post_process_answer(self, raw_answer: str, original_query: str) -> str:
        """Cleans and rephrases the final answer to be conversational."""
        answer_core = str(raw_answer).strip().rstrip('.')
        if not answer_core:
            return "Sorry, I couldn't extract an answer from the context."

        rephrase_prompt_text = (
            f"You are a helpful assistant. Rephrase the following answer to be natural and precise. "
            f"YOU MUST cite the publication date or year of the information if it is mentioned in the context. "
            f"If no date is found, state 'Date unknown'.\n\n"
            f"Original Question: {original_query}\n"
            f"Extracted Answer: {answer_core}\n\n"
            f"Response:"
        )
        
        rephrased_raw = self.models.llm.complete(rephrase_prompt_text, max_new_tokens=64)
        s = ' '.join(str(rephrased_raw).strip().split())
        
        # Ensure the core answer is in the final output
        if answer_core not in s:
            s = f"The answer is {answer_core}."
        
        # Ensure it's a single sentence
        idx = s.find('.')
        s = s[:idx+1].strip() if idx != -1 else s + '.'
        
        return s

    def query(self, query_text: str) -> dict:
        """Executes a query against the pipeline and returns the response and source nodes."""
        if not self.query_engine:
            raise RuntimeError("Pipeline is not set up. Please run `setup_pipeline` first.")
            
        print(f"Executing query: {query_text}")
        response = self.query_engine.query(query_text)
        
        final_answer = self._post_process_answer(str(response), query_text)
        
        return {
            "answer": final_answer,
            "source_nodes": response.source_nodes,
        }